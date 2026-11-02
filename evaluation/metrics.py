"""Cost vs. quality for a two-model router: threshold sweep, baselines and the headline metric.

Each prompt has a cheap and a strong answer with known quality and cost. A router gives every
prompt a score; at threshold t, prompts with score >= t go to the strong model. Sweeping t over
every distinct score traces the router's cost/quality curve, from all-cheap to all-strong.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalRow:
    prompt: str
    q_cheap: float
    q_strong: float
    cost_cheap: float
    cost_strong: float


@dataclass(frozen=True)
class Point:
    threshold: float
    frac_strong: float
    cost: float  # mean USD per prompt
    quality: float  # mean quality of the chosen answers
    cost_frac: float  # cost / all-strong cost
    quality_frac: float  # quality / all-strong quality; can exceed 1


def _columns(rows: Sequence[EvalRow]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.array([getattr(r, name) for r in rows], dtype=np.float64)
        for name in ("q_cheap", "q_strong", "cost_cheap", "cost_strong")
    )


def sweep(rows: Sequence[EvalRow], scores: np.ndarray) -> list[Point]:
    scores = np.asarray(scores, dtype=np.float64)
    if not rows:
        raise ValueError("no rows to evaluate")
    if len(rows) != len(scores):
        raise ValueError(f"{len(rows)} rows but {len(scores)} scores")
    q_cheap, q_strong, c_cheap, c_strong = _columns(rows)
    n = len(rows)
    strong_quality, strong_cost = q_strong.mean(), c_strong.mean()

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    # switching prompt i from cheap to strong adds these deltas; prefix sums give every cut
    dq = np.concatenate([[0.0], np.cumsum((q_strong - q_cheap)[order])])
    dc = np.concatenate([[0.0], np.cumsum((c_strong - c_cheap)[order])])
    # a cut is valid only after the last prompt of a tie group (ties route strong together)
    last_of_group = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    cuts = np.concatenate([[0], last_of_group, [n]])
    thresholds = [float("inf"), *(float(sorted_scores[k - 1]) for k in cuts[1:])]

    base_q, base_c = q_cheap.sum(), c_cheap.sum()
    points = []
    for k, threshold in zip(cuts, thresholds, strict=True):
        quality = (base_q + dq[k]) / n
        cost = (base_c + dc[k]) / n
        points.append(
            Point(
                threshold=threshold,
                frac_strong=k / n,
                cost=float(cost),
                quality=float(quality),
                cost_frac=float(cost / strong_cost) if strong_cost else 0.0,
                quality_frac=float(quality / strong_quality) if strong_quality else 0.0,
            )
        )
    return sorted(points, key=lambda p: (p.cost, p.frac_strong))


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def cost_saved_at(points: list[Point], target: float = 0.95) -> float | None:
    """Share of the all-strong cost saved at the operating point that reaches `target` quality.

    `points` comes from `sweep` (sorted by cost). Returns None when no point reaches `target`.
    """
    for i, p_above in enumerate(points):
        # prefix sums can leave all-strong at 0.9999999999999998, missing a target of 1.0
        if p_above.quality_frac >= target - 1e-9:
            # If it lands on the target (within rounding), or it's the very first point
            if i == 0 or p_above.quality_frac <= target:
                return 1.0 - p_above.cost_frac

            p_below = points[i - 1]

            # Interpolate cost_frac linearly
            dq = p_above.quality_frac - p_below.quality_frac
            dc = p_above.cost_frac - p_below.cost_frac

            t = (target - p_below.quality_frac) / dq
            interp_cost = p_below.cost_frac + t * dc

            return 1.0 - interp_cost

    return None


def misroutes(rows: Sequence[EvalRow], scores: np.ndarray, threshold: float) -> int:
    """Prompts the cheap model failed (and the strong one didn't) that were still routed cheap."""
    q_cheap, q_strong, _, _ = _columns(rows)
    routed_cheap = np.asarray(scores, dtype=np.float64) < threshold
    return int(np.sum(routed_cheap & (q_cheap < 0.5) & (q_strong >= 0.5)))
