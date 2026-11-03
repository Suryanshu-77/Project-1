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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tierwise.config import CACHE_ROUTER_DIR, _resolve
from tierwise.fetch import FetchError, download_file

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
