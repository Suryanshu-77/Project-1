from typing import Any

from tierwise.affinity import AffinityTable, prefix_boundaries, split_prefix
from tierwise.schema import ChatRequest
from tierwise.tokens import estimate_output, estimate_tokens


def _req(messages: list[dict[str, Any]], **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate({"messages": messages, **extra})


def _conversation(last: str, system: str = "You are helpful.") -> ChatRequest:
    return _req(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": last},
        ]
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_estimate_tokens_chars_div_4_ceil() -> None:
    assert estimate_tokens(_req([{"role": "user", "content": "x" * 9}])) == 3


def test_estimate_output_defaults_512() -> None:
    assert estimate_output(_req([{"role": "user", "content": "x"}])) == 512
    assert estimate_output(_req([{"role": "user", "content": "x"}], max_tokens=50)) == 50


def test_prefix_excludes_last_user_turn() -> None:
    first, tokens = split_prefix(_conversation("question A"))
    second, _ = split_prefix(_conversation("a completely different question B"))
    assert first is not None and first == second
    assert tokens > 0


def test_prefix_changes_with_system_or_tools() -> None:
    base, _ = split_prefix(_conversation("q"))
    other_system, _ = split_prefix(_conversation("q", system="You are terse."))
    data = _conversation("q").model_dump()
    data["tools"] = [{"type": "function", "function": {"name": "f"}}]
    with_tools, _ = split_prefix(ChatRequest.model_validate(data))
    assert len({base, other_system, with_tools}) == 3


def test_single_user_message_has_no_prefix() -> None:
    assert split_prefix(_req([{"role": "user", "content": "hi"}])) == (None, 0)


def test_incumbent_respects_ttl() -> None:
    clock = FakeClock()
    table = AffinityTable(clock=clock)
    table.record("p", "model-a")
    clock.now = 299
    assert table.incumbent("p", lambda _: 300) == "model-a"
    clock.now = 301
    assert table.incumbent("p", lambda _: 300) is None


def test_incumbent_is_most_recent_warm_model() -> None:
    clock = FakeClock()
    table = AffinityTable(clock=clock)
    table.record("p", "model-a")
    clock.now = 10
    table.record("p", "model-b")
    assert table.incumbent("p", lambda _: 300) == "model-b"
    assert set(table.warm_models("p")) == {"model-a", "model-b"}


def test_none_ttl_never_warm() -> None:
    table = AffinityTable(clock=FakeClock())
    table.record("p", "model-a")
    assert table.incumbent("p", lambda _: None) is None
    assert table.incumbent(None, lambda _: 300) is None


def test_lru_eviction() -> None:
    table = AffinityTable(max_entries=2, clock=FakeClock())
    for prefix in ("p1", "p2", "p3"):
        table.record(prefix, "m")
    assert table.warm_models("p1") == {}
    assert set(table.warm_models("p3")) == {"m"}


def test_continued_conversation_shares_a_boundary_with_the_previous_prompt() -> None:
    turn1 = _req(
        [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "q1"}]
    )
    turn2 = _req(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
    )
    served_prompt, served_tokens = prefix_boundaries(turn1)[-1]
    earlier = prefix_boundaries(turn2)[:-1]
    assert (served_prompt, served_tokens) in earlier


def test_longest_warm_boundary_and_chain_incumbent() -> None:
    clock = FakeClock()
    table = AffinityTable(clock=clock)
    bounds = [("h1", 10), ("h2", 25), ("h3", 40)]
    table.record("h2", "model-a")
    assert table.longest_warm(bounds, "model-a", 300) == 25
    assert table.longest_warm(bounds, "model-b", 300) == 0
    assert table.chain_incumbent(bounds, lambda _: 300) == "model-a"
    clock.now = 400
    assert table.chain_incumbent(bounds, lambda _: 300) is None


def test_record_and_warm_by_effort() -> None:
    table = AffinityTable(clock=FakeClock())
    table.record("p", "model-a", "low")
    # effort ignored: any recorded effort of the model is warm
    assert table.is_warm("p", "model-a", 300, effort="high", effort_matters=False)
    # effort part of the cache key: only the same effort is warm
    assert table.is_warm("p", "model-a", 300, effort="low", effort_matters=True)
    assert not table.is_warm("p", "model-a", 300, effort="high", effort_matters=True)
    assert not table.is_warm("p", "model-a", 300, effort=None, effort_matters=True)
    assert table.incumbent("p", lambda _: 300) == "model-a"
