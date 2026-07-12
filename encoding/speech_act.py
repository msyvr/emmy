"""The speech-act encoder — primary for message streams (pre-sweep spec §1).

A pinned single-stream judge labels each message *in isolation* into a K = 6
taxonomy: propose / agree / disagree / inform / request / other. Seeing one
message at a time is what keeps it E1-compliant — a judge shown two streams can
manufacture the correlation it reports (the joint-encoder prohibition).

The judge is injected as a callable text -> label so the encoder logic is
testable without a model. Pinning is the caller's responsibility and part of
the instrument identity: the ``judge_id`` string (model + prompt version +
temperature, e.g. "qwen2.5-7b/speech-act-v1/T0") enters ``config_id``, so a
different judge is a different instrument. Judge label noise is measured, not
assumed away — that is E7's probe-set calibration, run per setup.

``keyword_judge`` is a deterministic rule-based judge: the test double, and a
legitimately cheap baseline encoder in its own right.

Non-message events map to "other" (one symbol per event keeps streams aligned
across encoders; the action-type encoder is the instrument for actions).
"""

from __future__ import annotations

from typing import Callable

from base import Encoder
from streams import AgentStream

TAXONOMY = ("propose", "agree", "disagree", "inform", "request", "other")
PROMPT_VERSION = "speech-act-v1"
PROMPT_TEMPLATE = (
    "Classify the following single message from a multi-agent conversation.\n"
    "Answer with exactly one word from: propose, agree, disagree, inform, request, other.\n"
    "\nMessage:\n{text}\n\nLabel:"
)


def keyword_judge(text: str) -> str:
    """Deterministic rule-based judge: test double + cheap baseline."""
    t = text.strip().lower()
    if not t:
        return "other"
    if t.endswith("?") or t.startswith(("please", "can you", "could you", "would you")):
        return "request"
    if t.startswith(("i suggest", "we should", "let's", "how about", "i propose")):
        return "propose"
    if t.startswith(("i agree", "agreed", "yes", "sounds good", "ok", "okay")):
        return "agree"
    if t.startswith(("i disagree", "no,", "no.", "i don't think", "that won't")):
        return "disagree"
    return "inform"


class SpeechActEncoder(Encoder):
    name = "speech-act"
    version = PROMPT_VERSION
    n_symbols = len(TAXONOMY)

    def __init__(self, judge: Callable[[str], str], judge_id: str) -> None:
        """``judge`` maps one rendered message to a label; ``judge_id`` pins its identity."""
        self._judge = judge
        self._judge_id = judge_id
        self._index = {label: i for i, label in enumerate(TAXONOMY)}

    @property
    def params(self) -> dict:
        return {"taxonomy": ",".join(TAXONOMY), "judge_id": self._judge_id}

    def _label(self, kind: str, payload: dict) -> str:
        if kind != "message":
            return "other"
        raw = self._judge(str(payload.get("text", "")))
        label = str(raw).strip().lower().split()[0] if str(raw).strip() else "other"
        return label if label in self._index else "other"

    def encode_stream(self, stream: AgentStream) -> tuple[int, ...]:
        return tuple(self._index[self._label(e.kind, e.payload)] for e in stream.events)
