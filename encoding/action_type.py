"""The action-type encoder — the workhorse (pre-sweep spec §1, primary).

Structured actions are already near-symbolic; this encoder is a deterministic
total map from events to a fixed K = 8 taxonomy:

    tool_search / tool_compute / tool_file / tool_other
    msg_direct / msg_broadcast / terminate / other

Every event maps to exactly one symbol (total function: unknown tools land in
``tool_other``, unknown kinds in ``other``), so the encoder is deterministic by
construction and needs no fitting.
"""

from __future__ import annotations

from base import Encoder
from streams import AgentStream

TAXONOMY = (
    "tool_search",
    "tool_compute",
    "tool_file",
    "tool_other",
    "msg_direct",
    "msg_broadcast",
    "terminate",
    "other",
)

# Declared tool-class map; extend deliberately (a new entry is a new encoder
# version — the config id changes with the table).
TOOL_CLASSES = {
    "search": "tool_search",
    "web": "tool_search",
    "browse": "tool_search",
    "lookup": "tool_search",
    "compute": "tool_compute",
    "calc": "tool_compute",
    "python": "tool_compute",
    "code": "tool_compute",
    "file": "tool_file",
    "read": "tool_file",
    "write": "tool_file",
    "edit": "tool_file",
}

BROADCAST_RECIPIENTS = (None, "", "all", "*")


class ActionTypeEncoder(Encoder):
    name = "action-type"
    version = "v1"
    n_symbols = len(TAXONOMY)

    def __init__(self) -> None:
        self._index = {label: i for i, label in enumerate(TAXONOMY)}

    @property
    def params(self) -> dict:
        return {"taxonomy": ",".join(TAXONOMY), "tools": ",".join(sorted(TOOL_CLASSES))}

    def _label(self, kind: str, payload: dict) -> str:
        if kind == "action":
            if payload.get("terminate"):
                return "terminate"
            tool = str(payload.get("tool", "")).lower()
            return TOOL_CLASSES.get(tool, "tool_other")
        if kind == "message":
            to = payload.get("to")
            return "msg_broadcast" if to in BROADCAST_RECIPIENTS else "msg_direct"
        return "other"

    def encode_stream(self, stream: AgentStream) -> tuple[int, ...]:
        return tuple(self._index[self._label(e.kind, e.payload)] for e in stream.events)
