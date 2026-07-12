"""The embedding-codebook encoder — third encoder for the sensitivity appendix.

A fixed, versioned embedder maps each message to a vector; a k-means codebook,
fit ONCE on a calibration pool and then frozen, maps vectors to symbols. Two
spec decisions are enforced in code, not prose:

  - **D-CB (fit once, pooled).** ``fit`` runs exactly once; a second call
    raises. Encoding before fitting raises. The calibration pool is sorted and
    de-duplicated before fitting, so the codebook is independent of data
    arrival order — same pool, same instrument, always (E5 determinism).
  - **The embedder is not a subject.** The embedder is injected and its
    identity string enters ``config_id``; the spec requires it not be one of
    the backbones under test (the instrument must not share structure with its
    subject) — enforced at wiring time, recorded here.

k-means is implemented locally (seeded k-means++ init, Lloyd iterations,
deterministic tie-breaking) rather than imported, so determinism is owned by
this module and the dependency surface stays flat.

Symbols: 0..k-1 are codebook clusters; k is the reserved symbol for
non-message or empty-text events (so n_symbols = k + 1 and every event gets a
symbol, keeping streams aligned across encoders).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from base import Encoder
from streams import AgentStream


def _kmeans(x: np.ndarray, k: int, seed: int, n_iter: int) -> np.ndarray:
    """Seeded k-means++ init + Lloyd, deterministic; returns (k, dim) centroids."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n < k:
        raise ValueError(f"calibration pool has {n} distinct texts < k = {k}")

    # k-means++ initialization.
    centroids = np.empty((k, x.shape[1]))
    centroids[0] = x[rng.integers(n)]
    d2 = np.sum((x - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        total = d2.sum()
        if total <= 0.0:
            centroids[i:] = centroids[0]
            break
        probs = d2 / total
        centroids[i] = x[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((x - centroids[i]) ** 2, axis=1))

    # Lloyd iterations; argmin breaks ties toward the lower index (numpy default).
    for _ in range(n_iter):
        assign = np.argmin(((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
        new = centroids.copy()
        for j in range(k):
            members = x[assign == j]
            if members.shape[0] > 0:
                new[j] = members.mean(axis=0)
        if np.allclose(new, centroids):
            break
        centroids = new
    return centroids


class CodebookEncoder(Encoder):
    name = "embedding-codebook"
    version = "v1"

    def __init__(self, embedder: Callable[[str], np.ndarray], embedder_id: str,
                 k: int = 8, seed: int = 0, n_iter: int = 50) -> None:
        self._embedder = embedder
        self._embedder_id = embedder_id
        self._k = k
        self._seed = seed
        self._n_iter = n_iter
        self._centroids: np.ndarray | None = None
        self.n_symbols = k + 1                     # + reserved non-message/empty symbol

    @property
    def params(self) -> dict:
        return {"embedder_id": self._embedder_id, "k": self._k, "seed": self._seed,
                "n_iter": self._n_iter}

    @property
    def fitted(self) -> bool:
        return self._centroids is not None

    def fit(self, calibration_texts) -> "CodebookEncoder":
        """Fit the codebook once, on the pooled calibration set (D-CB), then freeze."""
        if self.fitted:
            raise RuntimeError(
                "codebook already fitted — D-CB prohibits refits; a new instrument "
                "requires a new encoder (version bump)"
            )
        pool = sorted({t for t in (str(t) for t in calibration_texts) if t.strip()})
        x = np.stack([np.asarray(self._embedder(t), dtype=float) for t in pool])
        self._centroids = _kmeans(x, self._k, self._seed, self._n_iter)
        return self

    def _symbol(self, kind: str, payload: dict) -> int:
        if kind != "message":
            return self._k
        text = str(payload.get("text", ""))
        if not text.strip():
            return self._k
        v = np.asarray(self._embedder(text), dtype=float)
        return int(np.argmin(np.sum((self._centroids - v) ** 2, axis=1)))

    def encode_stream(self, stream: AgentStream) -> tuple[int, ...]:
        if not self.fitted:
            raise RuntimeError("codebook not fitted — fit on the pooled calibration set first")
        return tuple(self._symbol(e.kind, e.payload) for e in stream.events)
