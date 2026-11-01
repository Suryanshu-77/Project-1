import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.metrics import EvalRow
from evaluation.run_eval import evaluate, plot
from tests.evaluation.test_metrics import PERFECT, _rows


def test_evaluate_report_shape() -> None:
    report = evaluate(
        _rows(), {"perfect": PERFECT, "constant": np.full(4, 0.5)}, "toy", "c", "s"
    )
    perfect, constant = report.routers["perfect"], report.routers["constant"]
    assert perfect["cost_saved_at_95"] == pytest.approx(0.495)  # interpolated
    assert perfect["cost_saved_at_95_single_threshold"] == pytest.approx(0.45)
    assert perfect["misroutes_at_95"] == 0
    # a constant score can only pick all-cheap or all-strong; interpolating mixes them at random
    assert constant["cost_saved_at_95_single_threshold"] == pytest.approx(0.0)
    assert constant["cost_saved_at_95"] == pytest.approx(1 - (0.1 + 0.9 * 0.9))
    assert report.baselines["all_cheap"]["quality"] == 0.5
    assert report.n == 4
    json.dumps(report.to_json())


def test_strong_worse_on_some_prompts() -> None:
    rows = [EvalRow("a", 1.0, 0.0, 1.0, 10.0), EvalRow("b", 0.0, 1.0, 1.0, 10.0)]
    report = evaluate(rows, {"perfect": np.array([0.0, 1.0])}, "toy", "c", "s")
    best = max(q for _, q in report.routers["perfect"]["points"])
    assert best == pytest.approx(2.0)  # routing beats all-strong; not clamped
    assert isinstance(report.routers["perfect"]["cost_saved_at_95"], float)


def test_plot_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    report = evaluate(_rows(), {"perfect": PERFECT}, "toy", "c", "s")
    path = tmp_path / "p.png"
    plot(report, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_reports_several_targets() -> None:
    report = evaluate(_rows(), {"perfect": PERFECT}, "toy", "c", "s")
    saved = report.routers["perfect"]["cost_saved_at"]
    assert set(saved) == {"0.95", "0.98", "0.99"}
    assert saved["0.95"] == pytest.approx(report.routers["perfect"]["cost_saved_at_95"])


def test_stored_points_are_thinned_but_keep_endpoints() -> None:
    rng = np.random.default_rng(0)
    rows = [EvalRow(f"p{i}", float(rng.random()), 1.0, 1.0, 10.0) for i in range(1000)]
    report = evaluate(rows, {"random": rng.random(1000)}, "toy", "c", "s")
    points = report.routers["random"]["points"]
    assert len(points) <= 201
    assert points[0][0] == pytest.approx(0.1) and points[-1] == pytest.approx([1.0, 1.0])
