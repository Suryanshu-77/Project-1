from typing import Any

import pytest

from tierwise.affinity import AffinityTable, split_prefix
from tierwise.options import Option, expand
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


def _ids(models: list[Option]) -> list[str]:
    return [o.spec.id for o in models]


def test_quality_bar_formula_and_modes() -> None:
    assert quality_bar(0.5, 1.0, "balanced") == pytest.approx(0.625)
    assert quality_bar(0.5, 1.0, "cheap") == pytest.approx(0.525)
    assert quality_bar(0.5, 1.0, "quality") == pytest.approx(0.725)
    assert quality_bar(0.5, 0.1, "balanced") == pytest.approx(0.725)
    assert quality_bar(1.0, 1.0, "quality") == 1.0


def test_cheapest_model_clearing_bar_first() -> None:
    selection = select(_result(_req(), 0.1), REGISTRY, ALL_PROVIDERS, AffinityTable())
    assert selection.candidates[0].spec.id == "cheap"  # bar 0.365: local (0.30) misses it
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
    assert selection.candidates[0].spec.id != "cheap"
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
    assert first.candidates[0].spec.id == "b" and first.prompt_hash is not None
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
    assert second.candidates[0].spec.id == "a"
    assert second.reason == "affinity"


def _efforts(default: str, **levels: tuple[float, int]) -> dict[str, Any]:
    return {
        "default": default,
        "levels": {k: {"quality": q, "thinking_tokens": t} for k, (q, t) in levels.items()},
    }


EFFORT_REGISTRY = Registry(
    baseline_model="strong",
    models=[
        spec("strong", "anthropic", 0.90, 1.0, 6.0, cache_ttl_s=300, price_cache_read=0.1,
             effort_in_cache_key=True,
             efforts=_efforts("high", low=(-0.04, 100), medium=(-0.02, 500), high=(0.0, 3000))),
        spec("mid", "openai", 0.80, 0.5, 5.0, cache_ttl_s=300, price_cache_read=0.05,
             efforts=_efforts("medium", low=(-0.04, 100), medium=(0.0, 500), high=(0.03, 3000))),
    ],
)  # fmt: skip


def _pick(req: ChatRequest, difficulty: float, table: AffinityTable | None = None) -> Option:
    selection = select(
        _result(req, difficulty), EFFORT_REGISTRY, ALL_PROVIDERS, table or AffinityTable()
    )
    return selection.candidates[0]


def test_easy_request_picks_lowest_clearing_effort() -> None:
    option = _pick(_req("hi"), 0.0)  # bar 0.30: everything clears
    assert (option.spec.id, option.effort) == ("mid", "low")


def test_strong_model_low_effort_beats_mid_model_high_effort_when_cheaper() -> None:
    # bar 0.30 + 0.65*0.75 = 0.7875; penalties x1.5 at this difficulty -> mid low 0.74 fails,
    # mid medium 0.80 clears; strong low 0.90 - 0.06 = 0.84 clears
    # mid medium: (512 + 500) x $5 = 5060; strong low: (512 + 100) x $6 = 3672 -> strong low
    option = _pick(_req("hi"), 0.75)
    assert (option.spec.id, option.effort) == ("strong", "low")


def test_cost_includes_thinking_tokens() -> None:
    selection = select(_result(_req("hi"), 0.0), EFFORT_REGISTRY, ALL_PROVIDERS, AffinityTable())
    efforts = [o.effort for o in selection.candidates if o.spec.id == "mid"]
    assert efforts == ["low", "medium", "high"]  # same model, cost grows with thinking


def test_ties_prefer_lower_effort() -> None:
    flat = Registry(
        baseline_model="m",
        models=[spec("m", "openai", 0.80, 1.0, 1.0,
                     efforts=_efforts("medium", low=(0.0, 0), medium=(0.0, 0)))],
    )  # fmt: skip
    selection = select(_result(_req("hi"), 0.0), flat, ALL_PROVIDERS, AffinityTable())
    assert [o.effort for o in selection.candidates] == ["low", "medium"]


def test_client_effort_restricts_options() -> None:
    req = _req("hi", reasoning_effort="high")
    selection = select(_result(req, 0.0), EFFORT_REGISTRY, ALL_PROVIDERS, AffinityTable())
    assert {o.effort for o in selection.candidates} == {"high"}
    assert all(o.send for o in selection.candidates)


def test_client_effort_none_eligible_raises_naming_effort() -> None:
    req = _req("hi", reasoning_effort="minimal")
    with pytest.raises(NoEligibleModel) as info:
        select(_result(req, 0.0), EFFORT_REGISTRY, ALL_PROVIDERS, AffinityTable())
    assert "minimal" in info.value.message


def test_tier_target_picks_cheapest_option_in_tier() -> None:
    tiered = Registry(
        baseline_model="strong",
        models=[m.model_copy(update={"tier": "flagship"}) for m in EFFORT_REGISTRY.models],
    )
    req = _req("hi", model="tier:flagship")
    selection = select(_result(req), tiered, ALL_PROVIDERS, AffinityTable())
    assert (selection.candidates[0].spec.id, selection.candidates[0].effort) == ("mid", "low")


def _conv(last: str) -> ChatRequest:
    system = {"role": "system", "content": "You are a careful assistant. " * 400}
    return ChatRequest.model_validate(
        {"messages": [system, {"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1"},
                      {"role": "user", "content": last}]}
    )  # fmt: skip


def test_effort_change_is_cold_when_effort_in_cache_key() -> None:
    req = _conv("q2")
    prefix, _ = split_prefix(req)
    assert prefix is not None
    table = AffinityTable()
    table.record(prefix, "strong", "low")
    selection = select(_result(req, 0.9), EFFORT_REGISTRY, ALL_PROVIDERS, table)
    costs = {o.effort: o for o in selection.candidates if o.spec.id == "strong"}
    strong = EFFORT_REGISTRY.get("strong")
    assert strong is not None
    # only the low option sees the warm prefix; high pays the full history again
    warm = estimated_cost(strong, 1000, 512, 1000)
    cold = estimated_cost(strong, 1000, 512, 0)
    assert warm < cold
    assert table.longest_warm([(prefix, 900)], "strong", 300, "low", True) == 900
    assert table.longest_warm([(prefix, 900)], "strong", 300, None, True) == 0
    assert set(costs) == {"low", "medium", "high"}


def test_effort_change_stays_warm_when_not_in_cache_key() -> None:
    table = AffinityTable()
    table.record("p", "mid", "low")
    assert table.longest_warm([("p", 900)], "mid", 300, "high", False) == 900


def test_affinity_keeps_incumbent_model_with_cheapest_clearing_effort() -> None:
    req = _conv("q2")
    prefix, _ = split_prefix(req)
    assert prefix is not None
    table = AffinityTable()
    table.record(prefix, "strong", "low")  # strong served this conversation at low effort
    selection = select(_result(req, 0.6), EFFORT_REGISTRY, ALL_PROVIDERS, table)
    assert (selection.candidates[0].spec.id, selection.candidates[0].effort) == ("strong", "low")
    assert selection.reason == "affinity"


def _labeled(label: str) -> list[str]:
    import json
    from pathlib import Path

    path = Path(__file__).parent / "router" / "labeled_prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["prompt"] for row in rows if row["label"] == label]


def _shipped_pick(prompt: str) -> Option:
    from tierwise.registry import load_registry
    from tierwise.router.router import route

    registry = load_registry("config/registry.yaml")
    req = ChatRequest.model_validate({"messages": [{"role": "user", "content": prompt}]})
    return select(route(req), registry, ALL_PROVIDERS, AffinityTable()).candidates[0]


def test_shipped_registry_hard_prompts_get_medium_or_above() -> None:
    low = [(p[:40], o.spec.id, o.effort) for p in _labeled("hard")
           if (o := _shipped_pick(p)).effort is not None and o.rank < 2]  # fmt: skip
    assert not low, f"hard prompts at low effort: {low}"


def test_shipped_registry_easy_prompts_get_lowest_level_of_chosen_model() -> None:
    wrong = []
    for prompt in _labeled("easy"):
        option = _shipped_pick(prompt)
        efforts = option.spec.efforts
        if efforts is not None:
            lowest = min(
                efforts.levels,
                key=lambda lv: ["minimal", "low", "medium", "high", "xhigh", "max"].index(lv),
            )
            if option.effort != lowest:
                wrong.append((prompt[:40], option.spec.id, option.effort))
    assert not wrong, f"easy prompts not at their model's lowest effort: {wrong}"


def test_low_effort_penalty_scales_with_difficulty() -> None:
    from tierwise.selector import effective_quality

    [low, medium, high] = expand(EFFORT_REGISTRY.get("mid"))  # type: ignore[arg-type]
    # registry shift for mid@low is -0.04 at the reference difficulty 0.5
    assert effective_quality(low, 0.5) == pytest.approx(0.76)
    assert effective_quality(low, 0.25) == pytest.approx(0.78)  # half the penalty
    assert effective_quality(low, 1.0) == pytest.approx(0.72)  # double the penalty
    assert effective_quality(low, 0.0) == pytest.approx(0.80)  # trivial: no penalty
    assert effective_quality(medium, 1.0) == pytest.approx(0.80)  # default: no shift
    assert effective_quality(high, 0.1) == pytest.approx(0.83)  # bonuses don't scale
