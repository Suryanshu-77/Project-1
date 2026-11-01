"""Champion/challenger gate for the flywheel retrain (spec §5.2).

Both classifiers are scored on the same rows: how much of the all-strong cost they save at 95%
of its quality (the curve over every threshold), and how many of the labeled hard prompts
their chosen threshold would send to the cheap side.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evaluation.metrics import EvalRow, cost_saved_at, sweep


@dataclass(frozen=True)
class GateResult:
    cost_saved_95: float | None  # None: no threshold reaches 95% of all-strong quality
    critical_misroutes: int
    auc: float | None  # None when the rows hold a single class


def _auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """ROC-AUC as the chance a random positive outscores a random negative (ties count half)."""
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def evaluate_classifier(
    p_rows: np.ndarray, rows: Sequence[EvalRow], p_critical: np.ndarray, threshold: float
) -> GateResult:
    p_rows = np.asarray(p_rows, dtype=np.float64)
    needs_strong = np.array([row.q_cheap < 0.5 <= row.q_strong for row in rows])
    return GateResult(
        cost_saved_95=cost_saved_at(sweep(rows, p_rows), 0.95),
        critical_misroutes=int((np.asarray(p_critical) < threshold).sum()),
        auc=_auc(p_rows, needs_strong),
    )


def _pct(value: float | None) -> str:
    return "None" if value is None else f"{value * 100:.1f}%"


def should_promote(champion: GateResult, challenger: GateResult) -> tuple[bool, str]:
    """Decide whether to promote the challenger over the champion based on their results.

    Safety (critical misroutes) is the absolute prerequisite. Among safe models, the one
    that saves more money at the target quality wins. Ties are broken by AUC.
    """
    if challenger.critical_misroutes > 0:
        return False, f"rejected: {challenger.critical_misroutes} critical misroutes"

    champ_cost = -math.inf if champion.cost_saved_95 is None else champion.cost_saved_95
    chall_cost = -math.inf if challenger.cost_saved_95 is None else challenger.cost_saved_95

    if chall_cost < champ_cost:
        # None (never reaches 95% quality) is displayed as "None", not -inf
        chall, champ = _pct(challenger.cost_saved_95), _pct(champion.cost_saved_95)
        return False, f"rejected: cost saved ({chall}) worse than champion ({champ})"

    if chall_cost > champ_cost:
        champ = _pct(champion.cost_saved_95)
        return True, f"promoted: cost saved {chall_cost * 100:.1f}% vs {champ}"

    # Cost is tied. Use AUC as a tiebreaker.
    champ_auc = -math.inf if champion.auc is None else champion.auc
    chall_auc = -math.inf if challenger.auc is None else challenger.auc

    if chall_auc > champ_auc:
        champ_shown = "None" if champion.auc is None else f"{champ_auc:.3f}"
        return True, f"promoted on tie: AUC {chall_auc:.3f} vs {champ_shown}"

    return False, "rejected on tie: no improvement in cost or AUC"
