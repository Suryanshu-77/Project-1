import json
from pathlib import Path
from typing import Any

import pytest

from tierwise.router.features import Features, extract_features
from tierwise.router.rules import rules_decision, score
from tierwise.schema import ChatRequest

LABELED = Path(__file__).with_name("labeled_prompts.jsonl")


def _req(messages: list[dict[str, Any]], **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate({"messages": messages, **extra})


def _labeled() -> list[tuple[str, str, float]]:
    rows = [json.loads(line) for line in LABELED.read_text(encoding="utf-8").splitlines()]
    scored = []
    for row in rows:
        difficulty = score(extract_features(_req([{"role": "user", "content": row["prompt"]}])))
        scored.append((row["prompt"], row["label"], difficulty))
    return scored


def _band(difficulty: float) -> str:
    if difficulty < 0.4:
        return "easy"
    return "hard" if difficulty > 0.6 else "medium"


def test_features_counts_code_math_keywords() -> None:
    text = "Prove this for production:\n```python\nx = 1\n```\nwhere $x^2 = 4$"
    f = extract_features(_req([{"role": "user", "content": text}]))
    assert f.code_blocks >= 1
    assert f.has_math
    assert f.reasoning_hits == 1  # prove
    assert f.high_stakes_hits == 1  # production


def test_features_only_last_user_message_for_keywords() -> None:
    f = extract_features(
        _req(
            [
                {"role": "user", "content": "security audit please"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "thanks"},
            ]
        )
    )
    assert (f.high_stakes_hits, f.easy_hits, f.user_turns) == (0, 1, 2)


def test_features_tools_images_turns() -> None:
    content = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
    ]
    f = extract_features(
        _req(
            [{"role": "user", "content": content}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
    )
    assert (f.has_images, f.has_tools, f.user_turns) == (True, True, 1)
    assert f.est_tokens > 0


def test_score_is_clamped() -> None:
    huge = Features(
        est_tokens=10**7,
        user_turns=500,
        code_blocks=100,
        has_math=True,
        high_stakes_hits=50,
        reasoning_hits=50,
        easy_hits=0,
        has_tools=True,
        has_images=True,
    )
    tiny = Features(0, 1, 0, False, 0, 0, 50, False, False)
    assert 0.0 <= score(tiny) <= score(huge) <= 1.0


def test_rules_decision_confidence_formula() -> None:
    decision = rules_decision(_req([{"role": "user", "content": "hi"}]))
    assert decision.source == "rules"
    assert decision.confidence == pytest.approx(min(1.0, 2 * abs(decision.difficulty - 0.5)))
    assert "est_tokens" in decision.signals


def test_labeled_set_routes_sensibly() -> None:
    scored = _labeled()
    wrong_easy = [(p[:50], round(d, 2)) for p, label, d in scored if label == "easy" and d >= 0.4]
    wrong_hard = [(p[:50], round(d, 2)) for p, label, d in scored if label == "hard" and d <= 0.6]
    assert not wrong_easy, f"easy prompts scored too high: {wrong_easy}"
    assert not wrong_hard, f"hard prompts scored too low: {wrong_hard}"
    accuracy = sum(_band(d) == label for _, label, d in scored) / len(scored)
    assert accuracy >= 0.8, (
        f"accuracy {accuracy:.0%}: {[(p[:40], lab, round(d, 2)) for p, lab, d in scored]}"
    )


def test_no_hard_prompt_scores_easy() -> None:
    misroutes = [(p[:50], round(d, 2)) for p, label, d in _labeled() if label == "hard" and d < 0.4]
    assert not misroutes, f"hard prompts would go to the cheapest tier: {misroutes}"
