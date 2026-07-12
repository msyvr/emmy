"""The stream data model for the encoding layer (pre-sweep spec, §1).

The load-bearing definition, fixed here so no implementer re-decides it: an
agent's stream is the agent's *emitted* actions and messages only — never its
received context or its view of the shared transcript. An implementer who
encodes "agent A's turn as it appears in the transcript" has already mixed
streams, and the per-stream soundness guarantee (E1: per-stream encoders cannot
manufacture dependence) is void. Exogenous episode facts (task specification,
instance id, configuration, seeds) live on the episode, not in any stream —
they are what the conditioning variable s may be encoded from (E6), and nothing
else is.

v1 assumes synchronous rounds: at most one event per agent per turn, and the
turn index is the time step (within-round ordering is declared out of scope in
the spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    """One emitted event: an action (tool call, artifact edit, terminate) or a message.

    ``kind`` is "action" or "message". ``payload`` carries the event's fields —
    for actions e.g. {"tool": "search", ...} or {"terminate": True}; for
    messages e.g. {"text": "...", "to": "agent_b" | None}.
    """

    turn: int
    kind: str
    payload: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("action", "message"):
            raise ValueError(f"unknown event kind: {self.kind!r}")
        if self.turn < 0:
            raise ValueError(f"negative turn: {self.turn}")


@dataclass(frozen=True)
class AgentStream:
    """One agent's emitted events, in turn order (at most one event per turn, v1)."""

    agent: str
    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        turns = [e.turn for e in self.events]
        if turns != sorted(turns):
            raise ValueError(f"events out of turn order for agent {self.agent!r}")
        if len(set(turns)) != len(turns):
            raise ValueError(f"more than one event per turn for agent {self.agent!r} (v1 is synchronous)")


@dataclass(frozen=True)
class Episode:
    """One episode: per-agent emitted streams plus the exogenous facts.

    ``exogenous`` is the only legal source for the conditioning variable s
    (E6): task specification, instance id, configuration, episode seed. State
    the agents jointly produced does not belong here — that is the collider
    phantom's lesson, calibrated in phase0.
    """

    episode_id: str
    exogenous: dict
    streams: dict[str, AgentStream]
