from typing import Any

import pytest

from tierwise.affinity import AffinityTable, split_prefix
from tierwise.registry import ModelSpec, Registry
from tierwise.router import RoutingDecision
from tierwise.router.modes import parse_target
from tierwise.router.router import RoutingResult, required_caps
from tierwise.schema import ChatRequest
from tierwise.selector import NoEligibleModel, estimated_cost, quality_bar, select

ALL_PROVIDERS = {"openai", "anthropic", "gemini", "ollama"}


def spec(
    model_id: str, provider: str, quality: float, price_in: float, price_out: float, **kw: Any
) -> ModelSpec:
    fields: dict[str, Any] = {
        "id": model_id,
        "provider": provider,
        "price_in": price_in,
        "price_out": price_out,
        "context": 200_000,
        "caps": ["tools", "vision", "json"],
        "quality": quality,
        "tier": "moderate",
        **kw,
    }
    return ModelSpec.model_validate(fields)


REGISTRY = Registry(
    baseline_model="top",
    models=[
        spec("local", "ollama", 0.30, 0, 0, caps=["json"], context=8_000, tier="basic"),
        spec("cheap", "openai", 0.40, 0.1, 0.4, caps=["tools", "json"], tier="basic",
             price_cache_read=0.01, cache_ttl_s=300),
        spec("mid", "anthropic", 0.70, 1.0, 5.0, price_cache_read=0.1, cache_ttl_s=300),
        spec("gem", "gemini", 0.85, 2.0, 12.0, context=1_000_000, tier="flagship"),
        spec("top", "anthropic", 0.95, 4.0, 20.0, tier="flagship", price_cache_read=0.2,
             cache_ttl_s=300),
    ],
)  # fmt: skip


def _req(content: Any = "hi", model: str = "auto", **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    )


def _result(req: ChatRequest, difficulty: float = 0.5, confidence: float = 1.0) -> RoutingResult:
    target = parse_target(req.model)
    decision = (
        RoutingDecision(difficulty=difficulty, confidence=confidence, source="rules")
        if target.kind == "auto"
        else None
    )
    return RoutingResult(target, decision, req, required_caps(req))


def _ids(models: list[ModelSpec]) -> list[str]:
    return [m.id for m in models]


def test_quality_bar_formula_and_modes() -> None:
    assert quality_bar(0.5, 1.0, "balanced") == pytest.approx(0.625)
    assert quality_bar(0.5, 1.0, "cheap") == pytest.approx(0.525)
    assert quality_bar(0.5, 1.0, "quality") == pytest.approx(0.725)
    assert quality_bar(0.5, 0.1, "balanced") == pytest.approx(0.725)
    assert quality_bar(1.0, 1.0, "quality") == 1.0


def test_cheapest_model_clearing_bar_first() -> None:
    selection = select(_result(_req(), 0.1), REGISTRY, ALL_PROVIDERS, AffinityTable())
    assert selection.candidates[0].id == "cheap"  # bar 0.365: local (0.30) misses it
    assert selection.reason == "rules"
    assert selection.bar == pytest.approx(0.365)


def test_failover_tail_order() -> None:
    selection = select(_result(_req(), 0.1), REGISTRY, ALL_PROVIDERS, AffinityTable())
    assert _ids(selection.candidates) == ["cheap", "mid", "gem", "top", "local"]


def test_nothing_clears_bar_falls_back_by_quality() -> None:
    selection = select(
        _result(_req(model="auto:quality"), 1.0), REGISTRY, ALL_PROVIDERS, AffinityTable()
    )
    assert selection.reason == "fallback"
    assert _ids(selection.candidates) == ["top", "gem", "mid", "cheap", "local"]


def test_filters_caps_and_context_and_provider() -> None:
    image = [
        {"type": "text", "text": "x" * 40_000},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
    ]
    selection = select(
        _result(_req(image), 0.1), REGISTRY, ALL_PROVIDERS - {"gemini"}, AffinityTable()
    )
    assert _ids(selection.candidates) == ["mid", "top"]  # no vision: cheap, local; no gemini


def test_no_vision_model_raises_no_eligible() -> None:
    image = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
    with pytest.raises(NoEligibleModel) as info:
        select(_result(_req(image)), REGISTRY, {"openai", "ollama"}, AffinityTable())
    assert "vision" in info.value.message


def test_oversized_prompt_raises_no_eligible() -> None:
    with pytest.raises(NoEligibleModel):
        select(_result(_req("x" * 4_000_000)), REGISTRY, ALL_PROVIDERS, AffinityTable())


def test_tier_target_cheapest_in_tier() -> None:
    selection = select(
        _result(_req(model="tier:flagship")), REGISTRY, ALL_PROVIDERS, AffinityTable()
    )
    assert _ids(selection.candidates) == ["gem", "top"]
    assert (selection.reason, selection.bar) == ("override", None)


def test_empty_tier_raises_no_eligible() -> None:
    with pytest.raises(NoEligibleModel):
        select(_result(_req(model="tier:moderate")), REGISTRY, {"openai"}, AffinityTable())


def test_warm_cache_makes_incumbent_cheaper() -> None:
    top, gem = REGISTRY.get("top"), REGISTRY.get("gem")
    assert top is not None and gem is not None
    warm_top = estimated_cost(top, est_input=100_000, est_output=500, warm_prefix_tokens=99_000)
    cold_gem = estimated_cost(gem, est_input=100_000, est_output=500, warm_prefix_tokens=0)
    assert warm_top < cold_gem


def _conversation(last: str) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "You are a careful assistant. " * 50},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": last},
            ]
        }
    )


def test_affinity_keeps_incumbent_within_10_percent() -> None:
    near = Registry(
        baseline_model="a",
        models=[
            spec("a", "anthropic", 0.9, 1.00, 1.0, cache_ttl_s=300),  # no cache discount
            spec("b", "openai", 0.9, 0.95, 1.0),
        ],
    )
    req = _conversation("next question")
    table = AffinityTable()
    prefix, _ = split_prefix(req)
    assert prefix is not None
    table.record(prefix, "a")
    selection = select(_result(req, 0.5), near, ALL_PROVIDERS, table)
    assert _ids(selection.candidates)[0] == "a"
    assert selection.reason == "affinity"


def test_affinity_ignored_when_incumbent_fails_bar_or_filter() -> None:
    req = _conversation("next question")
    table = AffinityTable()
    prefix, _ = split_prefix(req)
    assert prefix is not None
    table.record(prefix, "cheap")  # quality 0.40 cannot clear a 0.82 bar
    selection = select(_result(req, 0.8), REGISTRY, ALL_PROVIDERS, table)
    assert selection.candidates[0].id != "cheap"
    assert selection.reason == "rules"


def test_stale_affinity_for_missing_model_ignored() -> None:
    req = _conversation("next question")
    table = AffinityTable()
    prefix, _ = split_prefix(req)
    assert prefix is not None
    table.record(prefix, "ghost-model-removed-from-registry")
    selection = select(_result(req, 0.1), REGISTRY, ALL_PROVIDERS, table)
    assert selection.reason == "rules"
    assert selection.prefix_hash == prefix


def test_continued_conversation_keeps_incumbent() -> None:
    near = Registry(
        baseline_model="a",
        models=[
            spec("a", "anthropic", 0.9, 1.00, 1.0, cache_ttl_s=300, price_cache_read=0.1),
            spec("b", "openai", 0.9, 0.95, 1.0),
        ],
    )
    system = {"role": "system", "content": "You are a careful assistant. " * 200}
    turn1 = ChatRequest.model_validate({"messages": [system, {"role": "user", "content": "q1"}]})
    table = AffinityTable()
    first = select(_result(turn1, 0.5), near, ALL_PROVIDERS, table)
    assert first.candidates[0].id == "b" and first.prompt_hash is not None
    table.record(first.prompt_hash, "a")  # pretend "a" served turn 1
    turn2 = ChatRequest.model_validate(
        {
            "messages": [
                system,
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ]
        }
    )
    second = select(_result(turn2, 0.5), near, ALL_PROVIDERS, table)
    assert second.candidates[0].id == "a"
    assert second.reason == "affinity"
