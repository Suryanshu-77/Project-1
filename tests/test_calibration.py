import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tierwise import calibration
from tierwise.calibration import (
    CANDIDATE_CUTS,
    CalibrationError,
    CutStats,
    choose_cut,
    cut_stats,
    fetch_reference,
    read_rows,
    reference_file,
    row_pid,
)  # fmt: skip
from training.prepare_data import normalize

CHEAP, STRONG = "cheap-model", "strong-model"


def _row(i: int, q_cheap: float, q_strong: float) -> dict[str, Any]:
    return {"key": f"k{i}", "prompt": f"Prompt {i}",
            CHEAP: {"response": f"cheap {i}", "score": q_cheap},
            STRONG: {"response": f"strong {i}", "score": q_strong}}  # fmt: skip


ROWS = {("a.parquet", 5): _row(5, 0.0, 1.0), ("a.parquet", 9): _row(9, 1.0, 1.0),
        ("b.parquet", 2): _row(2, 0.0, 0.0)}  # fmt: skip


def _ref(rows: dict[tuple[str, int], dict[str, Any]] = ROWS) -> dict[str, Any]:
    pairs = [{"file": file, "row": i, "key": r["key"], "pid": row_pid(r["prompt"])}
             for (file, i), r in rows.items()]  # fmt: skip
    return {"dataset": "ds", "revision": "abc", "cheap": CHEAP, "strong": STRONG,
            "files": {}, "pairs": pairs}  # fmt: skip


def _fake(rows: dict[tuple[str, int], dict[str, Any]], calls: list[Any] | None = None):
    def read(ref: dict[str, Any], file: str, wanted: list[int]) -> list[dict[str, Any]]:
        if calls is not None:
            calls.append((file, wanted))
        return [rows[(file, i)] for i in wanted if (file, i) in rows]

    return read


def test_row_pid_matches_training_normalize() -> None:
    prompt = "  Mixed CASE\n and   spaces "
    assert row_pid(prompt) == hashlib.sha256(normalize(prompt).encode()).hexdigest()[:16]


def test_fetch_reference_returns_pairs_with_gold_in_ref_order() -> None:
    pairs = fetch_reference(_ref(), _fake(ROWS))
    assert [p.gold for p in pairs] == [True, False, False]  # (0,1) (1,1) (0,0)
    assert (pairs[0].prompt, pairs[0].cheap, pairs[0].strong) == ("Prompt 5", "cheap 5", "strong 5")
    assert pairs[0].pid == row_pid("Prompt 5")


def test_fetch_reference_reads_each_file_once() -> None:
    calls: list[Any] = []
    fetch_reference(_ref(), _fake(ROWS, calls))
    assert sorted(calls) == [("a.parquet", [5, 9]), ("b.parquet", [2])]


def test_fetch_reference_key_mismatch_fails() -> None:
    moved = {**ROWS, ("a.parquet", 9): {**ROWS[("a.parquet", 9)], "key": "other"}}
    with pytest.raises(CalibrationError, match="a.parquet row 9 does not match"):
        fetch_reference(_ref(), _fake(moved))


def test_fetch_reference_pid_mismatch_fails() -> None:
    moved = {**ROWS, ("a.parquet", 9): {**ROWS[("a.parquet", 9)], "prompt": "something else"}}
    with pytest.raises(CalibrationError, match="does not match"):
        fetch_reference(_ref(), _fake(moved))


def test_fetch_reference_missing_answer_fails() -> None:
    broken = {**ROWS, ("b.parquet", 2): {**ROWS[("b.parquet", 2)], CHEAP: {"response": None}}}
    with pytest.raises(CalibrationError, match="does not match"):
        fetch_reference(_ref(), _fake(broken))


def test_fetch_reference_short_read_fails() -> None:
    short = {k: v for k, v in ROWS.items() if k != ("a.parquet", 9)}
    with pytest.raises(CalibrationError, match="missing 1 of 3"):
        fetch_reference(_ref(), _fake(short))


def test_read_rows_takes_rows_from_a_checked_parquet(tmp_path: Path, monkeypatch) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.Table.from_pylist([_row(i, 1.0, 0.0) | {"other": "x"} for i in range(5)])
    pq.write_table(table, tmp_path / "a.parquet")
    got: list[tuple[str, str]] = []

    def download(url: str, target: Path, sha256: str, max_bytes: int) -> Path:
        got.append((url, sha256))
        return tmp_path / "a.parquet"

    monkeypatch.setattr(calibration, "download_file", download)
    ref = _ref() | {"files": {"a.parquet": {"sha256": "s", "size": 10}}}
    rows = read_rows(ref, "a.parquet", [3, 1], cache=tmp_path)
    assert [r["key"] for r in rows] == ["k3", "k1"] and "other" not in rows[0]
    assert got == [("https://huggingface.co/datasets/ds/resolve/abc/a.parquet", "s")]


def test_read_rows_without_pyarrow_says_how_to_install(monkeypatch) -> None:
    import builtins

    real = builtins.__import__

    def no_pyarrow(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("pyarrow"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    with pytest.raises(CalibrationError, match=r"tierwise\[calibrate\]"):
        read_rows(_ref(), "a.parquet", [1])


def test_packaged_reference_is_400_text_free_pairs_at_a_pinned_revision() -> None:
    ref = reference_file()
    assert len(ref["pairs"]) == 400 and len(ref["revision"]) == 40
    assert all(set(p) == {"file", "row", "key", "pid"} for p in ref["pairs"])
    assert {p["file"] for p in ref["pairs"]} <= set(ref["files"])
    assert all(set(f) == {"sha256", "size"} for f in ref["files"].values())
    assert (ref["cheap"], ref["strong"]) == ("openai-gpt-4o-mini", "openai-gpt-4o")


DATA = Path(__file__).parent / "data"


def _stats(cut: int, rate: float, f1: float) -> CutStats:
    return CutStats(cut=cut, rate=rate, precision=f1, recall=f1, f1=f1)


def test_cut_stats_counts() -> None:
    s = cut_stats([(1, 4), (1, 4), (3, 4), (5, 5)], [True, False, True, False], cut=2)
    assert (s.cut, s.rate, s.precision, s.recall) == (2, 0.5, 0.5, 0.5)
    assert s.f1 == pytest.approx(0.5)


def test_cut_stats_with_no_labels_has_zero_precision() -> None:
    s = cut_stats([(4, 4), (5, 5)], [True, False], cut=2)
    assert (s.rate, s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0, 0.0)


def test_choose_cut_drops_rates_outside_the_band() -> None:
    """Reference rate 10%: only rates from 5% to 20% are eligible, whatever their F1."""
    stats = [_stats(2, 0.04, 0.9), _stats(3, 0.08, 0.5), _stats(4, 0.25, 0.9)]
    assert choose_cut(stats, 0.10) == stats[1]


def test_choose_cut_band_edges_are_eligible() -> None:
    stats = [_stats(2, 0.05, 0.4), _stats(3, 0.20, 0.6)]
    assert choose_cut(stats, 0.10) == stats[1]


def test_choose_cut_best_f1_wins() -> None:
    stats = [_stats(2, 0.09, 0.6), _stats(3, 0.12, 0.7), _stats(4, 0.15, 0.5)]
    assert choose_cut(stats, 0.10) == stats[1]


def test_choose_cut_tie_goes_to_smaller_cut() -> None:
    stats = [_stats(3, 0.12, 0.7), _stats(2, 0.09, 0.7)]
    assert choose_cut(stats, 0.10) == stats[1]


def test_choose_cut_none_when_nothing_eligible() -> None:
    assert choose_cut([_stats(2, 0.01, 0.9), _stats(4, 0.5, 0.9)], 0.10) is None


def test_haiku_pilot_reproduces_protocol_3() -> None:
    """Claude Haiku's 400 stored pilot judgments: Protocol 3 chose c = 2 from exactly these."""
    rows = json.loads((DATA / "haiku_pilot_scores.json").read_text(encoding="utf-8"))
    scores, gold = [(c, s) for c, s, _ in rows], [bool(g) for *_, g in rows]
    stats = [cut_stats(scores, gold, c) for c in CANDIDATE_CUTS]
    best = choose_cut(stats, sum(gold) / len(gold))
    assert best is not None and best.cut == 2
    assert round(best.rate, 3) == 0.045 and round(best.precision, 2) == 0.94
    assert round(best.recall, 2) == 0.65
