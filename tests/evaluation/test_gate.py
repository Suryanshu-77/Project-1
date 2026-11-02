import numpy as np
import pytest

from evaluation.gate import GateResult, evaluate_classifier, quality_target, should_promote
from evaluation.metrics import EvalRow


def _rows(needs_strong: list[bool]) -> list[EvalRow]:
    return [EvalRow(f"p{i}", 0.0 if hard else 1.0, 1.0, 0.1, 1.0)
            for i, hard in enumerate(needs_strong)]  # fmt: skip


def test_evaluate_classifier_scores_rows() -> None:
    rows = _rows([True, False, True, False])
    p_rows = np.array([0.9, 0.1, 0.8, 0.2])
    result = evaluate_classifier(p_rows, rows, critical_misroutes=2)
    assert result.critical_misroutes == 2
    assert result.auc == 1.0  # every hard prompt scored above every easy one
    assert result.cost_saved is not None and result.cost_saved > 0.4


def test_auc_is_none_with_one_class() -> None:
    result = evaluate_classifier(np.array([0.2, 0.4]), _rows([False, False]), 0)
    assert result.auc is None


def test_random_scores_save_less_than_perfect_ones() -> None:
    needs = [i % 3 == 0 for i in range(60)]
    perfect = evaluate_classifier(np.array([float(h) for h in needs]), _rows(needs), 0)
    noise = evaluate_classifier(np.random.default_rng(0).random(60), _rows(needs), 0)
    assert perfect.cost_saved is not None
    assert noise.cost_saved is None or noise.cost_saved < perfect.cost_saved


def _easy_rows(n: int = 200) -> list[EvalRow]:
    """SPROUT-like traffic: the cheap model fails 1 prompt in 20, so all-cheap reaches 95%."""
    return [EvalRow(f"p{i}", 0.0 if i % 20 == 0 else 1.0, 1.0, 0.08, 1.0) for i in range(n)]


def test_target_separates_perfect_from_random_on_easy_traffic() -> None:
    rows = _easy_rows()
    perfect = evaluate_classifier(np.array([1.0 - r.q_cheap for r in rows]), rows, 0)
    noise = evaluate_classifier(np.random.default_rng(0).random(len(rows)), rows, 0)
    constant = evaluate_classifier(np.zeros(len(rows)), rows, 0)
    assert perfect.cost_saved is not None and noise.cost_saved is not None
    assert perfect.cost_saved > noise.cost_saved + 0.1  # at 95% all three would tie
    assert perfect.cost_saved > (constant.cost_saved or 0.0)
    assert perfect.target == noise.target  # same rows, same target


@pytest.mark.parametrize("rows", [
    _easy_rows(), _rows([True, False] * 10),
    [EvalRow("same", 1.0, 1.0, 0.1, 1.0)] * 5,  # cheap already matches strong everywhere
    [EvalRow("better", 1.0, 0.5, 0.1, 1.0)] * 5,  # cheap beats strong: all-cheap is above 1
])  # fmt: skip
def test_target_is_reachable(rows: list[EvalRow]) -> None:
    target = quality_target(rows)
    assert 0.0 < target <= 1.0  # all-strong is quality 1.0, so a target <= 1 always has a value
    assert evaluate_classifier(np.zeros(len(rows)), rows, 0).cost_saved is not None


GOOD = GateResult(cost_saved=0.50, critical_misroutes=0, auc=0.80)


def test_rejects_any_critical_misroute() -> None:
    better_but_unsafe = GateResult(cost_saved=0.70, critical_misroutes=1, auc=0.90)
    promote, reason = should_promote(GOOD, better_but_unsafe)
    assert not promote and reason


def test_rejects_worse_cost_saved() -> None:
    promote, reason = should_promote(GOOD, GateResult(0.40, 0, 0.85))
    assert not promote and reason


def test_promotes_strictly_better_and_safe() -> None:
    promote, reason = should_promote(GOOD, GateResult(0.55, 0, 0.82))
    assert promote and reason


@pytest.mark.parametrize("champion_saved", [None, 0.5])
def test_none_cost_saved_counts_as_worst(champion_saved: float | None) -> None:
    champion = GateResult(champion_saved, 0, 0.7)
    promote, _ = should_promote(champion, GateResult(None, 0, 0.9))
    if champion_saved is not None:
        assert not promote  # can't reach 95% quality at all: worse than a champion that can
    promote, _ = should_promote(GateResult(None, 0, 0.7), GateResult(0.3, 0, 0.7))
    assert promote  # reaching the target beats never reaching it


def test_tie_is_deterministic() -> None:
    first = should_promote(GOOD, GOOD)
    assert first == should_promote(GOOD, GOOD) and first[1]


def test_decide_rejects_a_classifier_that_calls_more_hard_prompts_easy() -> None:
    from evaluation.gate import decide

    champion = GateResult(0.50, 0, 0.80, classifier_misroutes=2)
    louder = GateResult(0.90, 0, 0.95, classifier_misroutes=3)  # the rules hide it in routing
    promote, reason = decide(champion, louder)
    assert not promote and "3" in reason and "2" in reason


def test_decide_defers_to_should_promote_otherwise() -> None:
    from evaluation.gate import decide

    champion = GateResult(0.50, 0, 0.80, classifier_misroutes=2)
    better = GateResult(0.60, 0, 0.80, classifier_misroutes=2)
    assert decide(champion, better) == should_promote(champion, better)
