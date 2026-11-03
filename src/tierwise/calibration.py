"""Per-judge cut calibration (`tierwise flywheel calibrate`).

How harshly a judge grades decides where "the cheaper answer falls short" starts: at the shipped
cut of 4 Claude Haiku labeled 3.3x the reference rate and made routing worse, while a cut of 2 on
the same judgments helped (experiments/flywheel_local/PROTOCOL-3.md). So the cut is chosen per
judge: the judge scores a packaged reference set whose right labels are known, and the cut whose
labels agree with them best is saved for it.

The reference is SPROUT pairs, packaged as (file, row, key, pid) at a pinned revision only:
SPROUT has no stated license, so its prompts and answers are downloaded from Hugging Face here
(the `calibrate` extra reads the parquet files), never shipped.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tierwise.config import CACHE_ROUTER_DIR, _resolve
from tierwise.fetch import FetchError, download_file
from tierwise.flywheel import _transcript, ask, judge_messages, needs_strong, parse_scores
from tierwise.providers.base import Provider, ProviderError
from tierwise.registry import ModelSpec
from tierwise.reliability.retry import RetryPolicy
from tierwise.schema import ChatRequest, Message

CACHE = CACHE_ROUTER_DIR.parent / "sprout"  # ~/.cache/tierwise/sprout/<revision>/<file>
FILE_URL = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{file}"
INSTALL = "calibration reads SPROUT's parquet files: pip install 'tierwise[calibrate]'"

ReadRows = Callable[[dict[str, Any], str, list[int]], list[dict[str, Any]]]  # (ref, file, rows)


class CalibrationError(Exception):
    pass


@dataclass(frozen=True)
class RefPair:
    pid: str
    prompt: str
    cheap: str  # the cheaper model's answer
    strong: str  # the stronger model's answer
    gold: bool  # SPROUT's verdict: the cheaper answer fell short where the stronger didn't


def row_pid(prompt: str) -> str:
    """The prompt's id: training.prepare_data.normalize, hashed (the package can't import it)."""
    return hashlib.sha256(" ".join(prompt.lower().split()).encode()).hexdigest()[:16]


def reference_file() -> dict[str, Any]:
    path = _resolve(None, "config/calibration_ref.json", "calibration_ref.json")
    ref: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return ref


def read_rows(
    ref: dict[str, Any], file: str, rows: list[int], cache: Path = CACHE
) -> list[dict[str, Any]]:
    """Rows of one of the reference's parquet files, in the order asked, with only the columns
    calibration needs. The file is downloaded once at the pinned revision and sha256-checked."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CalibrationError(INSTALL) from exc
    meta = ref["files"][file]
    url = FILE_URL.format(dataset=ref["dataset"], revision=ref["revision"], file=file)
    try:
        path = download_file(url, cache / ref["revision"] / file, meta["sha256"], meta["size"])
    except FetchError as exc:
        raise CalibrationError(str(exc)) from exc
    table = pq.read_table(path, columns=["key", "prompt", ref["cheap"], ref["strong"]])
    taken: list[dict[str, Any]] = table.take([i for i in rows if i < table.num_rows]).to_pylist()
    return taken


def _pair(want: dict[str, Any], row: dict[str, Any], cheap: str, strong: str) -> RefPair:
    answers = [row.get(cheap) or {}, row.get(strong) or {}]
    usable = all(a.get("response") and a.get("score") is not None for a in answers)
    same = row.get("key") == want["key"] and row_pid(row.get("prompt") or "") == want["pid"]
    if not (same and usable):
        raise CalibrationError(f"{want['file']} row {want['row']} does not match the packaged "
                               "reference pair; regenerate calibration_ref.json")  # fmt: skip
    q_cheap, q_strong = (float(a["score"]) for a in answers)
    return RefPair(want["pid"], row["prompt"], answers[0]["response"], answers[1]["response"],
                   q_cheap < 0.5 <= q_strong)  # fmt: skip


def fetch_reference(ref: dict[str, Any], read: ReadRows = read_rows) -> list[RefPair]:
    """The reference pairs, read and checked one by one against the packaged key and pid: rows
    that aren't the packaged pairs must fail here, not calibrate on the wrong prompts."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    for want in ref["pairs"]:
        by_file.setdefault(want["file"], []).append(want)
    got: dict[tuple[str, int], RefPair] = {}
    for file, wants in by_file.items():
        rows = read(ref, file, [w["row"] for w in wants])
        for want, row in zip(wants, rows, strict=False):  # a short read is counted below
            got[(file, want["row"])] = _pair(want, row, ref["cheap"], ref["strong"])
    missing = len(ref["pairs"]) - len(got)
    if missing:
        raise CalibrationError(f"missing {missing} of {len(ref['pairs'])} reference pairs in "
                               "SPROUT's files")  # fmt: skip
    return [got[(p["file"], p["row"])] for p in ref["pairs"]]


CANDIDATE_CUTS = (2, 3, 4)  # needs strong = cheap < cut and strong >= GOOD_ENOUGH


@dataclass(frozen=True)
class CutStats:
    """How one cut's labels agree with the reference labels (SPROUT's verdicts)."""

    cut: int
    rate: float  # the share of pairs it labels "needs strong"
    precision: float  # of those, the share SPROUT agrees with (0.0 when it labels none)
    recall: float  # of SPROUT's positives, the share it finds
    f1: float


def cut_stats(scores: list[tuple[int, int]], gold: list[bool], cut: int) -> CutStats:
    labels = [needs_strong(cheap, strong, cut) for cheap, strong in scores]
    hits = sum(label and g for label, g in zip(labels, gold, strict=True))
    precision = hits / sum(labels) if any(labels) else 0.0
    recall = hits / sum(gold) if any(gold) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return CutStats(cut, sum(labels) / len(labels), precision, recall, f1)


def choose_cut(stats: list[CutStats], reference_rate: float) -> CutStats | None:
    """The cut to save for this judge, or None when no candidate is trustworthy.

    Protocol 3's rule: a cut is eligible only when its label rate is within half to twice the
    reference rate (inclusive); among the eligible ones the best F1 wins, a tie going to the
    smaller cut. None when nothing is eligible: then nothing is saved for the judge.
    """
    eligible = [s for s in stats if 0.5 * reference_rate <= s.rate <= 2.0 * reference_rate]

    if not eligible:
        return None

    return max(eligible, key=lambda s: (s.f1, -s.cut))


MAX_UNSCORED = 0.05  # more unscored pairs than this and the calibration is incomplete


@dataclass(frozen=True)
class Judged:
    scores: dict[str, tuple[int, int]]  # pid -> (cheap, strong)
    unscored: int
    spent_usd: float
    stopped_by_cap: bool


async def judge_reference(
    pairs: list[RefPair],
    provider: Provider,
    judge: ModelSpec,
    max_usd: float,
    policy: RetryPolicy = RetryPolicy(),  # noqa: B008 (frozen dataclass)
) -> Judged:
    """Score each reference pair with the judge, exactly as `flywheel label` does (the same
    transcript, rubric and seeded answer order), stopping before a call could pass max_usd."""
    scores: dict[str, tuple[int, int]] = {}
    unscored, spent, last = 0, 0.0, 0.0
    for pair in pairs:
        if spent + last > max_usd:  # the last call's cost estimates the next one's
            return Judged(scores, unscored, spent, stopped_by_cap=True)
        request = ChatRequest(model=judge.id, messages=[Message(role="user", content=pair.prompt)])
        messages, cheap_key = judge_messages(_transcript(request), pair.cheap, pair.strong,
                                             seed=pair.pid)  # fmt: skip
        grading = ChatRequest(model=judge.id, messages=messages, temperature=0)
        try:
            verdict, last = await ask(provider, judge, grading, policy)
            spent += last
            a, b = parse_scores(verdict.content or "")
        except (ProviderError, ValueError):
            unscored += 1
            continue
        scores[pair.pid] = (a, b) if cheap_key == "a" else (b, a)
    return Judged(scores, unscored, spent, stopped_by_cap=False)


def estimate_usd(pairs: list[RefPair], judge: ModelSpec) -> float:
    """What judging the pairs should cost: ~4 characters a token, plus the rubric, ~20 out."""
    tokens_in = sum((len(p.prompt) + len(p.cheap) + len(p.strong)) / 4 + 250 for p in pairs)
    return (tokens_in * judge.price_in + 20 * len(pairs) * judge.price_out) / 1e6


def _bad(path: Path, why: str) -> CalibrationError:
    return CalibrationError(f"{path} is not a valid cuts file ({why}): fix or delete it, then "
                            "rerun tierwise flywheel calibrate")  # fmt: skip


def _read_cuts(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _bad(path, str(exc)) from exc
    if not isinstance(data, dict):
        raise _bad(path, "not a JSON object")
    for judge_id, entry in data.items():
        cut = entry.get("cut") if isinstance(entry, dict) else None
        # bool is an int subclass: `true` must not pass as a cut of 1
        if not isinstance(cut, int) or isinstance(cut, bool) or cut not in CANDIDATE_CUTS:
            raise _bad(path, f"{judge_id!r} has no cut in {CANDIDATE_CUTS}")
    return data


def load_cuts(path: Path) -> dict[str, int]:
    """judge id -> its calibrated cut; {} when nothing was calibrated yet. A corrupt file is an
    error: training on labels from a garbled cut would be worse than not training."""
    return {judge_id: entry["cut"] for judge_id, entry in _read_cuts(path).items()}


def save_cut(path: Path, judge_id: str, entry: dict[str, Any]) -> None:
    """Replace one judge's entry, keeping the others; written atomically."""
    data = _read_cuts(path) | {judge_id: entry}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".cuts-",
                                      suffix=".part", delete=False)  # fmt: skip
    try:
        with tmp:
            json.dump(data, tmp, indent=2)
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
