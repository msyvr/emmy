"""Known-answer and contract tests for the encoding layer.

The two contract tests every encoder must pass are Gate E checks 1–2 made
executable: E1 isolation (another agent's stream cannot influence this agent's
symbols — verified behaviorally, not just by signature) and E5 determinism
(same stream twice, identical symbols). Run directly
(``uv run python encoding/test_encoding.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from action_type import ActionTypeEncoder, TAXONOMY as ACTION_TAXONOMY
from base import entropy_bits
from codebook import CodebookEncoder
from speech_act import SpeechActEncoder, keyword_judge
from streams import AgentStream, Episode, Event
from surface import SurfaceEncoder

# --- fixtures ---------------------------------------------------------------


def toy_embedder(text: str) -> np.ndarray:
    """Deterministic toy embedder: crude but separable features."""
    return np.array([
        float(len(text)),
        float(text.count("a")),
        float(text.count("?")),
        float("```" in text),
    ])


CALIBRATION_TEXTS = [
    "aaaa" * 10, "aaa" * 12, "aaaaa" * 9,                       # a-heavy cluster
    "let us plan?", "what next?", "which file?",                # short questions
    "```python\nprint(1)\n```" * 3, "```js\nx=1\n```" * 4,      # code-ish
    "the quick brown fox jumps over the lazy dog " * 5,         # long prose
]


def build_encoders():
    cb = CodebookEncoder(toy_embedder, "toy-embedder/v1", k=4, seed=0).fit(CALIBRATION_TEXTS)
    return [
        ActionTypeEncoder(),
        SurfaceEncoder(),
        SpeechActEncoder(keyword_judge, "keyword_judge/v1"),
        cb,
    ]


def sample_stream(agent: str = "a") -> AgentStream:
    return AgentStream(agent, (
        Event(0, "message", {"text": "I suggest we split the task.", "to": "all"}),
        Event(1, "action", {"tool": "search", "query": "prior art"}),
        Event(2, "message", {"text": "what did you find?", "to": "b"}),
        Event(3, "action", {"tool": "quantum_widget"}),
        Event(4, "message", {"text": ""}),
        Event(5, "action", {"terminate": True}),
    ))


def sample_episode() -> Episode:
    other = AgentStream("b", (
        Event(0, "message", {"text": "agreed, sounds good", "to": "a"}),
        Event(1, "action", {"tool": "write", "path": "notes.md"}),
    ))
    return Episode("ep-0", {"task": "toy", "seed": 7}, {"a": sample_stream(), "b": other})


# --- contract tests: every encoder ------------------------------------------


def test_e1_isolation_behavioral():
    # Replace agent b's stream entirely; agent a's symbols must be bit-identical.
    episode = sample_episode()
    garbage = AgentStream("b", tuple(
        Event(t, "message", {"text": f"GARBAGE {t}" * 50, "to": "a"}) for t in range(6)
    ))
    mutated = Episode(episode.episode_id, episode.exogenous, {"a": episode.streams["a"], "b": garbage})
    for enc in build_encoders():
        before = enc.encode_stream(episode.streams["a"])
        after = enc.encode_stream(mutated.streams["a"])
        assert before == after, f"E1 violated by {enc.name}"


def test_e5_determinism():
    stream = sample_stream()
    for enc in build_encoders():
        assert enc.encode_stream(stream) == enc.encode_stream(stream), f"nondeterministic: {enc.name}"


def test_one_symbol_per_event_and_in_range():
    stream = sample_stream()
    for enc in build_encoders():
        symbols = enc.encode_stream(stream)
        assert len(symbols) == len(stream.events)
        assert all(0 <= s < enc.n_symbols for s in symbols)


def test_config_id_stable_and_parameter_sensitive():
    a1, a2 = ActionTypeEncoder(), ActionTypeEncoder()
    assert a1.config_id == a2.config_id
    j1 = SpeechActEncoder(keyword_judge, "keyword_judge/v1")
    j2 = SpeechActEncoder(keyword_judge, "some-model/T0")
    assert j1.config_id != j2.config_id      # different judge -> different instrument


# --- known answers: action-type ----------------------------------------------


def test_action_type_known_answers():
    enc = ActionTypeEncoder()
    idx = {label: i for i, label in enumerate(ACTION_TAXONOMY)}
    symbols = enc.encode_stream(sample_stream())
    assert symbols[0] == idx["msg_broadcast"]    # to: "all"
    assert symbols[1] == idx["tool_search"]
    assert symbols[2] == idx["msg_direct"]       # to: "b"
    assert symbols[3] == idx["tool_other"]       # unknown tool
    assert symbols[5] == idx["terminate"]


# --- known answers: surface ---------------------------------------------------


def test_surface_known_answers():
    enc = SurfaceEncoder()
    stream = AgentStream("a", (
        Event(0, "message", {"text": ""}),
        Event(1, "message", {"text": "short one"}),
        Event(2, "message", {"text": "x" * 100}),
        Event(3, "message", {"text": "```code```" + "x" * 400}),
        Event(4, "action", {"tool": "search"}),
    ))
    labels = [enc._label(e.kind, e.payload) for e in stream.events]
    assert labels == ["empty", "short", "mid", "long_code", "non_message"]


# --- known answers: speech-act -------------------------------------------------


def test_speech_act_known_answers_and_invalid_label():
    enc = SpeechActEncoder(keyword_judge, "keyword_judge/v1")
    stream = AgentStream("a", (
        Event(0, "message", {"text": "I suggest we split the task."}),
        Event(1, "message", {"text": "Agreed, sounds good."}),
        Event(2, "message", {"text": "What did you find?"}),
        Event(3, "message", {"text": "The results are in the file."}),
        Event(4, "action", {"tool": "search"}),
    ))
    labels = [enc._label(e.kind, e.payload) for e in stream.events]
    assert labels == ["propose", "agree", "request", "inform", "other"]

    bad_judge = SpeechActEncoder(lambda text: "HALLUCINATED_LABEL", "bad/v0")
    assert bad_judge._label("message", {"text": "anything"}) == "other"


# --- codebook: freeze semantics, determinism, separability ---------------------


def test_codebook_freeze_semantics():
    enc = CodebookEncoder(toy_embedder, "toy-embedder/v1", k=4)
    try:
        enc.encode_stream(sample_stream())
        raise AssertionError("encode before fit must raise")
    except RuntimeError:
        pass
    enc.fit(CALIBRATION_TEXTS)
    try:
        enc.fit(CALIBRATION_TEXTS)
        raise AssertionError("refit must raise (D-CB)")
    except RuntimeError:
        pass


def test_codebook_order_independence():
    a = CodebookEncoder(toy_embedder, "toy-embedder/v1", k=4, seed=0).fit(CALIBRATION_TEXTS)
    b = CodebookEncoder(toy_embedder, "toy-embedder/v1", k=4, seed=0).fit(
        list(reversed(CALIBRATION_TEXTS)) + CALIBRATION_TEXTS[:2]   # shuffled + duplicates
    )
    stream = sample_stream()
    assert a.encode_stream(stream) == b.encode_stream(stream)


def test_codebook_separates_toy_clusters():
    enc = CodebookEncoder(toy_embedder, "toy-embedder/v1", k=4, seed=0).fit(CALIBRATION_TEXTS)
    code_msg = Event(0, "message", {"text": "```python\ny=2\n```" * 3})
    question = Event(1, "message", {"text": "shall we proceed?"})
    a_heavy = Event(2, "message", {"text": "aaaa" * 11})
    s = enc.encode_stream(AgentStream("a", (code_msg, question, a_heavy)))
    assert len(set(s)) == 3                        # three different clusters
    assert all(sym < enc._k for sym in s)          # none fell in the reserved symbol
    assert enc.encode_stream(AgentStream("a", (Event(0, "action", {"tool": "x"}),)))[0] == enc._k


# --- utilization helper ---------------------------------------------------------


def test_entropy_bits():
    assert entropy_bits([0, 0, 0, 0], 4) == 0.0
    assert abs(entropy_bits([0, 1, 2, 3], 4) - 2.0) < 1e-12
    assert entropy_bits([], 4) == 0.0


# --- stream model validation ----------------------------------------------------


def test_stream_model_rejects_malformed():
    try:
        Event(0, "telepathy", {})
        raise AssertionError("unknown kind must raise")
    except ValueError:
        pass
    try:
        AgentStream("a", (Event(1, "message", {}), Event(0, "message", {})))
        raise AssertionError("out-of-order turns must raise")
    except ValueError:
        pass
    try:
        AgentStream("a", (Event(0, "message", {}), Event(0, "action", {})))
        raise AssertionError("two events on one turn must raise (v1 synchronous)")
    except ValueError:
        pass


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} encoding tests passed.")


if __name__ == "__main__":
    _run_all()
