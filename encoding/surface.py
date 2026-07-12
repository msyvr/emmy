"""The surface encoder — the discriminant control (pre-sweep spec §1).

Deterministic, unfitted, and deliberately shallow: message length bins crossed
with a has-code flag. It is *expected* to carry little construct signal — its
role in the battery is discriminant (a metric that reads coordination off
surface statistics alone is suspect), so all bin boundaries are declared
constants, not fitted quantities (which also makes E5 trivial: nothing to
leak, nothing to freeze).

K = 8 taxonomy:

    empty / short / short_code / mid / mid_code / long / long_code / non_message
"""

from __future__ import annotations

from base import Encoder
from streams import AgentStream

TAXONOMY = (
    "empty",
    "short",
    "short_code",
    "mid",
    "mid_code",
    "long",
    "long_code",
    "non_message",
)

LENGTH_BINS = (80, 320)  # chars: short < 80 <= mid < 320 <= long
CODE_MARKERS = ("```", "def ", "import ", "{", "};")


class SurfaceEncoder(Encoder):
    name = "surface"
    version = "v1"
    n_symbols = len(TAXONOMY)

    def __init__(self) -> None:
        self._index = {label: i for i, label in enumerate(TAXONOMY)}

    @property
    def params(self) -> dict:
        return {"length_bins": LENGTH_BINS, "code_markers": ",".join(CODE_MARKERS)}

    def _label(self, kind: str, payload: dict) -> str:
        if kind != "message":
            return "non_message"
        text = str(payload.get("text", ""))
        if not text.strip():
            return "empty"
        if len(text) < LENGTH_BINS[0]:
            size = "short"
        elif len(text) < LENGTH_BINS[1]:
            size = "mid"
        else:
            size = "long"
        has_code = any(marker in text for marker in CODE_MARKERS)
        return f"{size}_code" if has_code else size

    def encode_stream(self, stream: AgentStream) -> tuple[int, ...]:
        return tuple(self._index[self._label(e.kind, e.payload)] for e in stream.events)
