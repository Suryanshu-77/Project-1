"""Rules router: turn request features into a difficulty score, with no model calls."""

from dataclasses import asdict

from tierwise.router import RoutingDecision
from tierwise.router.features import Features, extract_features
from tierwise.schema import ChatRequest

# How much each signal pushes difficulty up (positive) or down (negative).
# Tune these against tests/router/labeled_prompts.jsonl.
WEIGHTS: dict[str, float] = {
    "base": 0.45,
    "per_1k_tokens": 0.05,
    "per_user_turn": 0.02,
    "code": 0.20,
    "math": 0.25,
    "high_stakes": 0.25,
    "reasoning": 0.15,
    "easy": -0.20,
    "tools": 0.15,
    "images": 0.10,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score(features: Features) -> float:
    """Difficulty in [0, 1]: 0 means any model will do, 1 means it needs the best one."""
    total = WEIGHTS["base"]

    # Scale continuous signals
    total += (features.est_tokens / 1000.0) * WEIGHTS["per_1k_tokens"]
    total += features.user_turns * WEIGHTS["per_user_turn"]

    # Cap keyword hit counts at 2 to prevent domination
    total += min(features.code_blocks, 2) * WEIGHTS["code"]
    total += min(features.high_stakes_hits, 2) * WEIGHTS["high_stakes"]
    total += min(features.reasoning_hits, 2) * WEIGHTS["reasoning"]
    total += min(features.easy_hits, 2) * WEIGHTS["easy"]

    # Booleans just add their weight when true
    if features.has_math:
        total += WEIGHTS["math"]
    if features.has_tools:
        total += WEIGHTS["tools"]
    if features.has_images:
        total += WEIGHTS["images"]

    return _clamp(total)


def rules_decision(req: ChatRequest) -> RoutingDecision:
    features = extract_features(req)
    difficulty = score(features)
    confidence = min(1.0, 2 * abs(difficulty - 0.5))
    return RoutingDecision(
        difficulty=difficulty, confidence=confidence, source="rules", signals=asdict(features)
    )
