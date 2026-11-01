import pytest

from evaluation.scorers import hybrid_scores, request_of, rules_scores
from tierwise.router import RoutingDecision
from tierwise.router.rules import rules_decision
from tierwise.schema import ChatRequest

PROMPTS = ["hi, thanks", "explain the CAP theorem", "debug this production outage step by step"]


class FakeLearned:
    async def decide(self, req: ChatRequest) -> RoutingDecision:
        return RoutingDecision(difficulty=0.8, confidence=0.1, source="learned")


def test_rules_scores_match_rules_router() -> None:
    expected = [rules_decision(request_of(p)).difficulty for p in PROMPTS]
    assert rules_scores(PROMPTS).tolist() == pytest.approx(expected)


def test_hybrid_matches_route() -> None:
    prompts = ["hi, thanks", "explain the CAP theorem"]
    scores, low_confidence = hybrid_scores(FakeLearned(), prompts)  # type: ignore[arg-type]
    assert scores[0] == pytest.approx(rules_decision(request_of("hi, thanks")).difficulty)
    assert scores[1] == pytest.approx(0.8)
    assert low_confidence == 0.5


def test_hybrid_without_learned_is_rules() -> None:
    scores, _ = hybrid_scores(None, PROMPTS)
    assert scores.tolist() == pytest.approx(rules_scores(PROMPTS).tolist())
