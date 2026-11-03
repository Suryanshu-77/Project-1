"""Write tierwise's packaged calibration reference and its test fixture, once (train group).

    uv run --group train python experiments/flywheel_local/make_reference.py

- config/calibration_ref.json (shipped as tierwise/data/): the Protocol 2 pilot's 400 pairs as
  SPROUT (file, row, key, pid) at a pinned revision, with each file's sha256 and size, and no
  text: `tierwise flywheel calibrate` downloads the files itself.
- tests/data/haiku_pilot_scores.json: [cheap, strong, gold] for the pilot pairs Claude Haiku
  scored (stage `pilot`, same_pair_v2), so a test can check calibration reproduces Protocol 3.
"""

import hashlib
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pairs import CHEAP, STRONG, pid  # noqa: E402
from samepair2 import DB, PILOT  # noqa: E402
from stage0 import build_pool, gold  # noqa: E402

from evaluation.datasets import SPROUT_REPO, load_sprout, row_from_sprout  # noqa: E402
from tierwise.calibration import fetch_reference  # noqa: E402
from training.prepare_data import normalize  # noqa: E402

REPO = HERE.parents[1]
REF = REPO / "config/calibration_ref.json"
FIXTURE = REPO / "tests/data/haiku_pilot_scores.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    train_rows, _ = load_sprout("train")
    pool = build_pool(train_rows, load_sprout("test")[0])[0][:PILOT]
    root = Path(snapshot_download(SPROUT_REPO, repo_type="dataset", allow_patterns=["*train*"]))
    revision = root.name  # snapshots/<commit sha>: the files at it never change
    want = {normalize(r.prompt) for r in pool}
    found: dict[str, dict[str, object]] = {}
    files: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("train-*.parquet")):
        name = path.relative_to(root).as_posix()
        table = pq.read_table(path, columns=["key", "prompt", CHEAP, STRONG])
        for i, rec in enumerate(table.to_pylist()):
            norm = normalize(rec["prompt"])
            if norm in want and norm not in found and row_from_sprout(rec, CHEAP, STRONG):
                found[norm] = {"file": name, "row": i, "key": rec["key"], "pid": pid(rec["prompt"])}
                files[name] = {"sha256": sha256(path), "size": path.stat().st_size}
    tree = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{SPROUT_REPO}/tree/{revision}/data", timeout=60))
    hub = {f["path"]: f["lfs"]["oid"] for f in tree if "lfs" in f}
    assert all(hub[name] == meta["sha256"] for name, meta in files.items()), "sha256 vs the Hub"
    ref = {"dataset": SPROUT_REPO, "revision": revision, "cheap": CHEAP, "strong": STRONG,
           "files": files, "pairs": [found[normalize(r.prompt)] for r in pool]}  # fmt: skip

    def local(ref: dict, file: str, rows: list[int]) -> list[dict]:
        return pq.read_table(root / file, columns=["key", "prompt", CHEAP, STRONG]).take(
            rows).to_pylist()  # fmt: skip

    checked = fetch_reference(ref, local)  # the shipped reader's own checks, on the local copy
    assert [p.pid for p in checked] == [pid(r.prompt) for r in pool]
    REF.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")

    db = sqlite3.connect(DB)
    got = {k: (c, s) for k, c, s in db.execute(
        "select key, cheap, strong from same_pair_v2 where stage = 'pilot' and cheap is not null")}
    rows = [[*got[f"label:{pid(r.prompt)}"], gold(r)] for r in pool
            if f"label:{pid(r.prompt)}" in got]  # fmt: skip
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"{len(checked)} pairs in {len(files)} files at {revision[:7]}, checked; G "
          f"{sum(p.gold for p in checked)}; fixture {len(rows)} rows, "
          f"G rate {sum(g for *_, g in rows) / len(rows):.1%}")


if __name__ == "__main__":
    main()
