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
    from matplotlib.ticker import PercentFormatter

    # (color, line width, label); unknown router names keep matplotlib's defaults
    style = {"random": ("#9AA0A6", 1.4, "random"), "rules": ("#F28E2B", 1.6, "rules only"),
             "learned": ("#4E79A7", 1.8, "learned only"),
             "hybrid": ("#2CA02C", 3.0, "hybrid (deployed)")}  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    for name, result in report.routers.items():
        xs, ys = zip(*result["points"], strict=True)
        color, width, label = style.get(name, (None, 1.6, name))
        ax.plot(xs, ys, label=label, color=color, linewidth=width, zorder=3 if width > 2 else 2)
    cheap = report.baselines["all_cheap"]
    strong = report.baselines["all_strong"]
    ax.scatter([cheap["cost"] / strong["cost"]], [cheap["quality"] / strong["quality"]],
               marker="o", color="#333333", zorder=4, label=f"all {report.cheap}")  # fmt: skip
    ax.scatter([1.0], [1.0], marker="s", color="#333333", zorder=4, label=f"all {report.strong}")
    ax.axhline(TARGET, linestyle="--", color="#9AA0A6", linewidth=1, zorder=1)
    ax.text(0.01, TARGET, f" {TARGET:.0%} quality target", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=8.5, color="#5F6368")  # fmt: skip
    saved = report.routers.get("hybrid", {}).get("cost_saved_at_95")
    # the headline: where the deployed router crosses the target, unless all-cheap already
    # reaches it (then every router "saves" the same, and the number says nothing)
    if saved is not None and cheap["quality"] / strong["quality"] < TARGET:
        ax.scatter([1 - saved], [TARGET], s=60, color="#2CA02C", edgecolor="white", zorder=5)
        ax.annotate(f"deployed router: {saved:.0%} cheaper\nthan all-{report.strong}",
                    (1 - saved, TARGET), xytext=(24, -46), textcoords="offset points",
                    fontsize=9, color="#1B5E20",
                    arrowprops={"arrowstyle": "-", "color": "#2CA02C"})  # fmt: skip
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f"cost, as a share of always using {report.strong}")
    ax.set_ylabel(f"quality, as a share of {report.strong}'s")
    shown = {"heldout": "held-out prompts", "sprout": "SPROUT"}.get(report.dataset, report.dataset)
    ax.set_title(f"Cost vs quality on {shown} (n={report.n:,})", loc="left",
                 fontsize=12, fontweight="bold")  # fmt: skip
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
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
