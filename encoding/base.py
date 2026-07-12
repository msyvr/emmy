"""Encoder base: the E1/E2 contract and the Gate-E utilization tool.

The contract every encoder in the battery satisfies:

  - **Per-stream by construction (E1).** ``encode_stream`` takes ONE agent's
    ``AgentStream`` and nothing else. The signature is the guarantee; the test
    suite additionally verifies behaviorally that another agent's stream cannot
    influence the output (guarding future implementations that might sneak
    shared state in).
  - **One symbol per event.** Alignment across agents (matched-turn slices) is
    the estimator layer's job, using the events' turn indices.
  - **Declared and frozen (E2).** An encoder's identity is (name, version,
    parameters); ``config_id`` hashes them so a run's config can pin exactly
    which instrument produced its symbols. Changing anything changes the id —
    a different id is a different instrument.
  - **Deterministic (E5).** Same stream in, same symbols out, always. Fitted
    components freeze after fitting (see ``codebook``).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod

import numpy as np

from streams import AgentStream


def config_id(name: str, version: str, params: dict) -> str:
    """Stable 12-hex identity for an encoder configuration (E2)."""
    canonical = json.dumps([name, version, sorted(params.items())], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class Encoder(ABC):
    """One symbol per event, from one agent's emitted stream alone."""

    name: str
    version: str
    n_symbols: int

    @abstractmethod
    def encode_stream(self, stream: AgentStream) -> tuple[int, ...]:
        """Symbols aligned to ``stream.events`` (one per event)."""

    @property
    @abstractmethod
    def params(self) -> dict:
        """The declared parameters that, with name/version, fix the instrument."""

    @property
    def config_id(self) -> str:
        return config_id(self.name, self.version, self.params)


def entropy_bits(symbols, n_symbols: int) -> float:
    """Plug-in entropy of the empirical symbol distribution, in bits.

    Gate E check 3 (utilization): an encoder that collapses to a near-constant
    symbol carries nothing; the check demands entropy above a declared minimum
    on pilot data.
    """
    symbols = np.asarray(symbols, dtype=int)
    if symbols.size == 0:
        return 0.0
    counts = np.bincount(symbols, minlength=n_symbols).astype(float)
    p = counts / counts.sum()
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))
