import warnings

import numpy as np
import pytest

from evaluation.metrics import EvalRow, Point, cost_saved_at, misroutes, random_scores, sweep


def _rows() -> list[EvalRow]:
    return [
        EvalRow(prompt=f"p{i}", q_cheap=q, q_strong=1.0, cost_cheap=1.0, cost_strong=10.0)
        for i, q in enumerate([1.0, 1.0, 0.0, 0.0])
    ]


PERFECT = np.array([0.0, 0.0, 1.0, 1.0])


def _point(cost_frac: float, quality_frac: float) -> Point:
    return Point(threshold=0.0, frac_strong=0.0, cost=cost_frac, quality=quality_frac,
                 cost_frac=cost_frac, quality_frac=quality_frac)  # fmt: skip


def test_sweep_endpoints_are_all_cheap_and_all_strong() -> None:
    points = sweep(_rows(), PERFECT)
    first, last = points[0], points[-1]
    assert (first.frac_strong, first.cost, first.quality) == (0.0, 1.0, 0.5)
    assert (last.frac_strong, last.cost, last.quality_frac, last.cost_frac) == (1.0, 10.0, 1.0, 1.0)


def test_perfect_scores_reach_full_quality_at_half_strong() -> None:
    mid = [p for p in sweep(_rows(), PERFECT) if p.frac_strong == 0.5]
    assert len(mid) == 1
    assert mid[0].quality_frac == 1.0
    assert mid[0].cost == pytest.approx(5.5)


def test_constant_scores_give_two_points() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        points = sweep(_rows(), np.full(4, 0.3))
    assert [p.frac_strong for p in points] == [0.0, 1.0]


def test_ties_route_strong() -> None:
    points = sweep(_rows(), np.array([0.5, 0.5, 0.0, 0.0]))
    assert any(p.threshold == 0.5 and p.frac_strong == 0.5 for p in points)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        sweep(_rows(), np.zeros(3))


def test_empty_rows_raise() -> None:
    with pytest.raises(ValueError):
        sweep([], np.zeros(0))


def test_cost_saved_at_perfect_router() -> None:
    # interpolated between all-cheap (cost 0.10, quality 0.5) and the perfect cut (0.55, 1.0):
    # quality 0.95 is 90% of the way, at cost 0.10 + 0.9 * 0.45 = 0.505
    assert cost_saved_at(sweep(_rows(), PERFECT), 0.95) == pytest.approx(1 - 0.505)


def test_cost_saved_at_exact_hit_uses_that_point() -> None:
    points = [_point(0.1, 0.5), _point(0.55, 1.0), _point(1.0, 1.0)]
    assert cost_saved_at(points, 1.0) == pytest.approx(0.45)


def test_cost_saved_at_unreachable_returns_none() -> None:
    assert cost_saved_at([_point(0.1, 0.5), _point(0.5, 0.9)], 0.95) is None


def test_cost_saved_at_never_beats_a_qualifying_point() -> None:
    rng = np.random.default_rng(1)
    rows = [
        EvalRow(f"p{i}", float(rng.random()), 1.0, 1.0, 10.0) for i in range(200)
    ]
    points = sweep(rows, rng.random(200))
    cheapest = min(p.cost_frac for p in points if p.quality_frac >= 0.95)
    saved = cost_saved_at(points, 0.95)
    assert saved is not None and saved >= 1 - cheapest - 1e-12


def test_random_scores_deterministic() -> None:
    np.testing.assert_array_equal(random_scores(10), random_scores(10))
    assert random_scores(10).shape == (10,)


def test_misroutes_counts_cheap_routed_failures() -> None:
    assert misroutes(_rows(), PERFECT, 0.5) == 0
    assert misroutes(_rows(), np.zeros(4), 0.5) == 2
