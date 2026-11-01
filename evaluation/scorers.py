"""Routing scores for the eval sweep: higher means "send this prompt to the strong model"."""

import asyncio
from collections.abc import Sequence

import numpy as np

from tierwise.router.learned import LearnedRouter
from tierwise.router.router import route
from tierwise.router.rules import rules_decision
from tierwise.schema import ChatRequest
from tierwise.selector import LOW_CONFIDENCE


def request_of(prompt: str) -> ChatRequest:
    return ChatRequest.model_validate({"messages": [{"role": "user", "content": prompt}]})


def rules_scores(prompts: Sequence[str]) -> np.ndarray:
    return np.array([rules_decision(request_of(p)).difficulty for p in prompts])


def learned_scores(router: LearnedRouter, prompts: Sequence[str], batch: int = 64) -> np.ndarray:
    """The classifier's raw P(strong model needed)."""
    reqs = [request_of(p) for p in prompts]
    return np.concatenate(
        [router.score_batch(reqs[i : i + batch]) for i in range(0, len(reqs), batch)]
    )


def hybrid_scores(
    router: LearnedRouter | None, prompts: Sequence[str]
) -> tuple[np.ndarray, float]:
    """What production does: route() difficulty (rules, or learned when rules are unsure).

    Also returns the share of prompts whose decision is low-confidence, which is when the
    selector raises the quality bar.
    """

    async def run() -> list[tuple[float, float]]:
        decisions = []
        for p in prompts:
            decision = (await route(request_of(p), router)).decision
            assert decision is not None  # auto routing always decides
            decisions.append((decision.difficulty, decision.confidence))
        return decisions

    pairs = asyncio.run(run())
    difficulty = np.array([d for d, _ in pairs])
    low = sum(c < LOW_CONFIDENCE for _, c in pairs) / len(pairs)
    return difficulty, float(low)
