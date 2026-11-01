"""CI gate: labeled hard prompts must never be routed to a basic-tier model."""

import json
from pathlib import Path

import pytest

from tierwise.affinity import AffinityTable
from tierwise.config import Settings
from tierwise.registry import load_registry
from tierwise.router.learned import LearnedRouter, load_learned
from tierwise.router.router import route
from tierwise.schema import ChatRequest
from tierwise.selector import select

ROOT = Path(__file__).parents[2]
PROVIDERS = {"anthropic", "openai", "gemini", "ollama"}


def _hard_prompts() -> list[str]:
    path = ROOT / "tests" / "router" / "labeled_prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r["prompt"] for r in rows if r["label"] == "hard"]


async def _basic_picks(learned: LearnedRouter | None) -> list[tuple[str, str]]:
    registry = load_registry(ROOT / "config" / "registry.yaml")
    picks = []
    for prompt in _hard_prompts():
        req = ChatRequest.model_validate({"messages": [{"role": "user", "content": prompt}]})
        option = select(await route(req, learned), registry, PROVIDERS, AffinityTable()).candidates[
            0
        ]
        if option.spec.tier == "basic":
            picks.append((prompt[:50], option.spec.id))
    return picks


async def test_hard_prompts_never_route_to_basic_tier_rules() -> None:
    assert len(_hard_prompts()) == 10
    misroutes = await _basic_picks(None)
    assert not misroutes, f"hard prompts sent to a basic model: {misroutes}"


async def test_hard_prompts_never_route_to_basic_tier_learned() -> None:
    learned = load_learned(Settings(_env_file=None, router_dir=str(ROOT / "models" / "router")))  # type: ignore[call-arg]
    if learned is None:
        pytest.skip("learned router artifacts not present (CI)")
    try:
        misroutes = await _basic_picks(learned)
    finally:
        learned.close()
    assert not misroutes, f"hard prompts sent to a basic model: {misroutes}"
