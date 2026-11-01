"""Benchmark tierwise's routers on cost vs. quality.

Usage: uv run python -m evaluation.run_eval --dataset {heldout,sprout} [--limit N]
Writes docs/benchmark/<dataset>.json (aggregates only) and docs/benchmark/<dataset>_pareto.png.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.datasets import load_gpt4_test, load_sprout
from evaluation.metrics import EvalRow, Point, cost_saved_at, misroutes, random_scores, sweep
from evaluation.scorers import hybrid_scores, learned_scores, rules_scores

OUT_DIR = Path("docs/benchmark")
TARGET = 0.95
# SPROUT's cheap model alone reaches 95% of the strong one, so stricter targets are reported too
TARGETS = (0.95, 0.98, 0.99)
MAX_STORED_POINTS = 200  # the JSON keeps a thinned curve; metrics use every point
PAIRS = {"heldout": ("Mixtral-8x7B", "GPT-4"), "sprout": ("gpt-4o-mini", "gpt-4o")}


@dataclass
class Report:
    dataset: str
    n: int
    dropped: int
    cheap: str
    strong: str
    baselines: dict[str, dict[str, float]]
    routers: dict[str, dict[str, Any]] = field(default_factory=dict)
    low_confidence_share: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _single_threshold(points: list[Point], target: float) -> Point | None:
    """The cheapest point a single fixed threshold reaches at `target` quality."""
    qualifying = [p for p in points if p.quality_frac >= target]
    return min(qualifying, key=lambda p: p.cost_frac) if qualifying else None


def _thin(points: list[Point], limit: int = MAX_STORED_POINTS) -> list[Point]:
    """At most limit + 1 points, evenly spaced along the curve, endpoints included."""
    if len(points) <= limit + 1:
        return points
    keep = sorted({round(i * (len(points) - 1) / limit) for i in range(limit + 1)})
    return [points[i] for i in keep]


def _auc(points: list[Point]) -> float:
    """Area under quality_frac over cost_frac: higher means more quality per dollar."""
    xs = [p.cost_frac for p in points]
    ys = [p.quality_frac for p in points]
    return float(np.trapezoid(ys, xs))


def evaluate(
    rows: Sequence[EvalRow],
    scorers: dict[str, np.ndarray],
    dataset: str,
    cheap: str,
    strong: str,
    dropped: int = 0,
    low_confidence_share: float | None = None,
) -> Report:
    """All-strong is always at quality_frac 1.0, so for TARGET <= 1 every router has a value."""
    q_cheap = float(np.mean([r.q_cheap for r in rows]))
    q_strong = float(np.mean([r.q_strong for r in rows]))
    c_cheap = float(np.mean([r.cost_cheap for r in rows]))
    c_strong = float(np.mean([r.cost_strong for r in rows]))
    report = Report(
        dataset=dataset,
        n=len(rows),
        dropped=dropped,
        cheap=cheap,
        strong=strong,
        baselines={
            "all_cheap": {"cost": c_cheap, "quality": q_cheap},
            "all_strong": {"cost": c_strong, "quality": q_strong},
        },
        low_confidence_share=low_confidence_share,
    )
    for name, scores in scorers.items():
        points = sweep(rows, scores)
        single = _single_threshold(points, TARGET)
        saved_single = None if single is None else 1 - single.cost_frac
        missed = None if single is None else misroutes(rows, scores, single.threshold)
        report.routers[name] = {
            "cost_saved_at_95": cost_saved_at(points, TARGET),  # interpolated (randomized mix)
            "cost_saved_at_95_single_threshold": saved_single,
            "misroutes_at_95": missed,
            "cost_saved_at": {f"{t:.2f}": cost_saved_at(points, t) for t in TARGETS},
            "auc_cost_quality": _auc(points),
            "points": [[p.cost_frac, p.quality_frac] for p in _thin(points)],
        }
    return report


def plot(report: Report, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    for name, result in report.routers.items():
        xs, ys = zip(*result["points"], strict=True)
        ax.plot(xs, ys, label=name, linewidth=1.6)
    cheap = report.baselines["all_cheap"]
    strong = report.baselines["all_strong"]
    ax.scatter([cheap["cost"] / strong["cost"]], [cheap["quality"] / strong["quality"]],
               marker="o", color="black", zorder=3, label=f"all {report.cheap}")  # fmt: skip
    ax.scatter([1.0], [1.0], marker="s", color="black", zorder=3, label=f"all {report.strong}")
    ax.axhline(TARGET, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel(f"cost (fraction of all-{report.strong})")
    ax.set_ylabel(f"quality (fraction of all-{report.strong})")
    ax.set_title(f"tierwise routers on {report.dataset} (n={report.n})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _fmt(value: float | None) -> str:
    return "not reached" if value is None else f"{value:.1%}"


def main(dataset: str, limit: int | None) -> int:
    from tierwise.config import Settings
    from tierwise.router.learned import load_learned

    router = load_learned(Settings())
    if router is None:
        print("learned router unavailable: run training.export_onnx (see the warning above)")
        return 1
    if dataset == "heldout":
        rows, dropped = load_gpt4_test()[:limit], 0
    else:
        rows, dropped = load_sprout(limit=limit)
    prompts = [r.prompt for r in rows]
    print(f"{dataset}: {len(rows)} prompts ({dropped} dropped); scoring...", flush=True)
    hybrid, low_confidence = hybrid_scores(router, prompts)
    scorers = {
        "random": random_scores(len(rows)),
        "rules": rules_scores(prompts),
        "learned": learned_scores(router, prompts),
        "hybrid": hybrid,
    }
    router.close()
    cheap, strong = PAIRS[dataset]
    report = evaluate(rows, scorers, dataset, cheap, strong, dropped, low_confidence)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{dataset}.json").write_text(json.dumps(report.to_json(), indent=1) + "\n")
    plot(report, OUT_DIR / f"{dataset}_pareto.png")

    header = "".join(f"{f'saved@{t:.0%}':>12}" for t in TARGETS)
    print(f"\n{'router':<10}{header}{'single@95%':>12}{'misroutes':>11}{'AUC':>8}")
    for name, r in report.routers.items():
        saved = "".join(f"{_fmt(v):>12}" for v in r["cost_saved_at"].values())
        print(f"{name:<10}{saved}{_fmt(r['cost_saved_at_95_single_threshold']):>12}"
              f"{r['misroutes_at_95']!s:>11}{r['auc_cost_quality']:>8.3f}")  # fmt: skip
    print(f"low-confidence share (hybrid): {low_confidence:.1%}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(PAIRS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    sys.exit(main(args.dataset, args.limit))
