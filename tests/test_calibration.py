import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from tierwise import calibration
from tierwise.__main__ import main
from tierwise.calibration import (
    CANDIDATE_CUTS,
    CalibrationError,
    CutStats,
    Judged,
    RefPair,
    choose_cut,
    cut_stats,
    fetch_reference,
    judge_reference,
    load_cuts,
    read_rows,
    reference_file,
    row_pid,
    save_cut,
)  # fmt: skip
from tierwise.providers.openai import OpenAIProvider
from tierwise.registry import ModelSpec
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
REPO = Path(__file__).parents[1]


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


JUDGE = ModelSpec.model_validate(
    {"id": "judge-x", "provider": "openai", "price_in": 10.0, "price_out": 10.0,
     "context": 100_000, "caps": ["json"], "quality": 0.5, "tier": "basic"}  # fmt: skip
)


def _ref_pairs(n: int, positives: int = 1) -> list[RefPair]:
    return [RefPair(pid=f"p{i}", prompt=f"q{i}", cheap=f"cheap {i}", strong=f"strong {i}",
                    gold=i < positives) for i in range(n)]  # fmt: skip


def _judge_provider(scores: dict[str, int], replies: dict[str, str] | None = None,
                    calls: list[Any] | None = None) -> OpenAIProvider:  # fmt: skip
    """A judge scoring each answer by its text ("cheap 3" -> scores["cheap 3"], default 5)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        text = body["messages"][-1]["content"]
        found = re.search(r"Answer A:\n(.*?)\n\nAnswer B:\n(.*?)\n\n", text, re.DOTALL)
        assert found is not None
        a, b = found.groups()
        default = json.dumps({"a": scores.get(a, 5), "b": scores.get(b, 5)})
        content = (replies or {}).get(a, default)
        return httpx.Response(200, json={
            "id": "x", "model": body["model"], "created": 1,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
        })  # fmt: skip

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIProvider(client, "sk-test")


async def test_judge_reference_scores_each_pair() -> None:
    calls: list[Any] = []
    provider = _judge_provider({"cheap 0": 1, "cheap 1": 4}, calls=calls)
    judged = await judge_reference(_ref_pairs(2), provider, JUDGE, max_usd=1.0)
    assert judged.scores == {"p0": (1, 5), "p1": (4, 5)}  # (cheap, strong), whatever the order
    assert (judged.unscored, judged.stopped_by_cap) == (0, False)
    assert judged.spent_usd == pytest.approx(0.04)  # 2 calls x 2000 tokens at $10/M
    assert calls[0]["model"] == "judge-x" and calls[0]["temperature"] == 0
    assert "user: q0" in calls[0]["messages"][-1]["content"]  # the label path's transcript


async def test_calibrate_counts_unparseable_as_unscored() -> None:
    replies = {"cheap 1": "B is better", "strong 1": '{"a": 4}'}  # prose, or one score only
    judged = await judge_reference(_ref_pairs(3), _judge_provider({}, replies), JUDGE, max_usd=1.0)
    assert set(judged.scores) == {"p0", "p2"} and judged.unscored == 1
    assert judged.spent_usd == pytest.approx(0.06)  # the unscored call was still paid for


async def test_judge_reference_stops_at_cap() -> None:
    judged = await judge_reference(_ref_pairs(5), _judge_provider({}), JUDGE, max_usd=0.05)
    assert len(judged.scores) == 2 and judged.stopped_by_cap  # 0.02 + 0.02; a third: 0.06 > 0.05
    assert judged.spent_usd == pytest.approx(0.04)


def test_load_cuts_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_cuts(tmp_path / "none.json") == {}


@pytest.mark.parametrize("text", ["{not json", '{"j": {"cut": 7}}', '{"j": {"rate": 0.1}}',
                                  '["j"]', '{"j": {"cut": "2"}}'])  # fmt: skip
def test_load_cuts_rejects_bad_file(tmp_path: Path, text: str) -> None:
    path = tmp_path / "judge_cuts.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CalibrationError, match="judge_cuts.json"):
        load_cuts(path)


def test_save_cut_keeps_other_judges(tmp_path: Path) -> None:
    path = tmp_path / "judge_cuts.json"
    save_cut(path, "a", {"cut": 3})
    before = json.loads(path.read_text(encoding="utf-8"))["a"]
    save_cut(path, "b", {"cut": 2})
    save_cut(path, "b", {"cut": 4})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["a"] == before and data["b"] == {"cut": 4}
    assert load_cuts(path) == {"a": 3, "b": 4}


def test_save_cut_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "judge_cuts.json"
    save_cut(path, "a", {"cut": 3})

    def fail(*_: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(calibration.os, "replace", fail)
    with pytest.raises(OSError):
        save_cut(path, "a", {"cut": 2})
    assert load_cuts(path) == {"a": 3} and list(tmp_path.iterdir()) == [path]


def _cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pairs: list[RefPair],
         judged: Judged | None = None) -> list[Any]:  # fmt: skip
    """`tierwise flywheel calibrate` with the reference and the judge faked."""
    judge_calls: list[Any] = []

    async def judge(pairs: list[RefPair], provider: Any, spec: ModelSpec, max_usd: float) -> Judged:
        judge_calls.append((spec.id, max_usd))
        assert judged is not None
        return judged

    monkeypatch.setattr(calibration, "reference_file", lambda: {})
    monkeypatch.setattr(calibration, "fetch_reference", lambda ref: pairs)
    monkeypatch.setattr(calibration, "judge_reference", judge)
    monkeypatch.chdir(tmp_path)  # no developer .env: it could name another judge model
    for name in [n for n in os.environ if n.startswith("TIERWISE_") or n == "JUDGE_MODEL"]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("TIERWISE_REGISTRY", str(REPO / "config/registry.yaml"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # the default judge: the baseline model
    return judge_calls


def _judged(scores: dict[str, tuple[int, int]], unscored: int = 0) -> Judged:
    return Judged(scores=scores, unscored=unscored, spent_usd=0.12, stopped_by_cap=False)


def test_flywheel_calibrate_writes_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    pairs = _ref_pairs(4)  # p0 is SPROUT's one positive
    scores = {"p0": (1, 5), "p1": (5, 5), "p2": (4, 5), "p3": (3, 3)}
    calls = _cli(monkeypatch, tmp_path, pairs, _judged(scores))
    out = tmp_path / "cuts.json"
    assert main(["flywheel", "calibrate", "--cuts-file", str(out), "--max-usd", "0.3"]) == 0
    entry = json.loads(out.read_text(encoding="utf-8"))[calls[0][0]]
    assert calls[0][1] == 0.3 and entry["cut"] == 2 and entry["n"] == 4
    assert (entry["rate"], entry["reference_rate"], entry["spent_usd"]) == (0.25, 0.25, 0.12)
    assert {"precision", "recall", "f1", "ts"} <= set(entry)
    assert re.search(r"cut 2 \(rate 25\.0%, G 25\.0%", capsys.readouterr().out)


def test_flywheel_calibrate_dry_run_makes_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _cli(monkeypatch, tmp_path, _ref_pairs(400))
    out = tmp_path / "cuts.json"
    assert main(["flywheel", "calibrate", "--dry-run", "--cuts-file", str(out)]) == 0
    assert calls == [] and not out.exists() and "400 calls" in capsys.readouterr().out


def test_flywheel_calibrate_too_many_unscored_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = _ref_pairs(20)
    _cli(monkeypatch, tmp_path, pairs, _judged({p.pid: (5, 5) for p in pairs[2:]}, unscored=2))
    out = tmp_path / "cuts.json"
    assert main(["flywheel", "calibrate", "--cuts-file", str(out)]) == 1 and not out.exists()


def test_flywheel_calibrate_stopped_by_cap_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judged = Judged(scores={"p0": (1, 5)}, unscored=0, spent_usd=0.1, stopped_by_cap=True)
    _cli(monkeypatch, tmp_path, _ref_pairs(4), judged)
    out = tmp_path / "cuts.json"
    assert main(["flywheel", "calibrate", "--cuts-file", str(out)]) == 1 and not out.exists()


def test_flywheel_calibrate_no_eligible_cut_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _ref_pairs(4)
    _cli(monkeypatch, tmp_path, pairs, _judged({p.pid: (1, 5) for p in pairs}))  # labels them all
    out = tmp_path / "cuts.json"
    assert main(["flywheel", "calibrate", "--cuts-file", str(out)]) == 1 and not out.exists()
    err = capsys.readouterr().err
    assert "cut 2" in err and "100.0%" in err and "G 25.0%" in err


def test_flywheel_calibrate_reference_failure_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cli(monkeypatch, tmp_path, [])

    def broken(ref: Any) -> list[RefPair]:
        raise CalibrationError("missing 3 of 400 reference pairs")

    monkeypatch.setattr(calibration, "fetch_reference", broken)
    assert main(["flywheel", "calibrate", "--cuts-file", str(tmp_path / "c.json")]) == 1


def test_flywheel_calibrate_corrupt_cuts_file_fails_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand-edited cuts file must stop the run before ~400 judge calls are paid for."""
    calls = _cli(monkeypatch, tmp_path, _ref_pairs(4), _judged({"p0": (1, 5)}))
    out = tmp_path / "cuts.json"
    out.write_text("{oops", encoding="utf-8")
    assert main(["flywheel", "calibrate", "--cuts-file", str(out)]) == 1
    assert calls == [] and "cuts.json" in capsys.readouterr().err


def test_flywheel_calibrate_save_failure_still_reports_the_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _ref_pairs(4)
    scores = {"p0": (1, 5), "p1": (5, 5), "p2": (4, 5), "p3": (3, 3)}
    _cli(monkeypatch, tmp_path, pairs, _judged(scores))

    def read_only(*_: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(calibration, "save_cut", read_only)
    assert main(["flywheel", "calibrate", "--cuts-file", str(tmp_path / "c.json")]) == 1
    err = capsys.readouterr().err
    assert "cut 2" in err and "read-only file system" in err


def test_flywheel_calibrate_refuses_when_the_estimate_passes_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run sure to stop at --max-usd would spend the whole cap and save nothing."""
    calls = _cli(monkeypatch, tmp_path, _ref_pairs(400), _judged({}))
    out = tmp_path / "cuts.json"
    assert main(["flywheel", "calibrate", "--cuts-file", str(out), "--max-usd", "0.001"]) == 1
    assert calls == [] and not out.exists() and "--max-usd" in capsys.readouterr().err
