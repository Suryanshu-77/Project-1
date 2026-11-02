"""Helpers shared by the same-pair judging scripts (v1: samepair.py, v2: samepair2.py)."""

import hashlib
import random
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from evaluation.datasets import SPROUT_REPO  # noqa: E402
from evaluation.metrics import EvalRow  # noqa: E402
from training.prepare_data import normalize  # noqa: E402

CHEAP, STRONG = "openai-gpt-4o-mini", "openai-gpt-4o"


def env(path: Path = Path(".env")) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
             if "=" in line and not line.lstrip().startswith("#"))  # fmt: skip
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


def pid(prompt: str) -> str:
    return hashlib.sha256(normalize(prompt).encode("utf-8")).hexdigest()[:16]


def answers(pool: list[EvalRow]) -> dict[str, tuple[str, str]]:
    """pid -> (gpt-4o-mini answer, gpt-4o answer), from the same SPROUT row load_sprout kept:
    the first one with both scores. Its scores must match the pool's."""
    import pyarrow.parquet as pq  # train group only
    from huggingface_hub import snapshot_download

    want = {normalize(r.prompt): r for r in pool}
    root = Path(snapshot_download(SPROUT_REPO, repo_type="dataset", allow_patterns=["*train*"]))
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("train-*.parquet")):
        for rec in pq.read_table(path, columns=["prompt", CHEAP, STRONG]).to_pylist():
            cheap, strong = rec.get(CHEAP), rec.get(STRONG)
            if not (cheap and strong) or None in (cheap.get("score"), strong.get("score")):
                continue  # load_sprout skipped this row too
            key = normalize(rec["prompt"])
            if key in want and pid(rec["prompt"]) not in out:
                row = want[key]
                scored = (float(cheap["score"]), float(strong["score"]))
                assert (row.q_cheap, row.q_strong) == scored, key[:60]
                out[pid(rec["prompt"])] = (cheap["response"] or "", strong["response"] or "")
    missing = [r.prompt[:60] for r in pool if pid(r.prompt) not in out]
    assert not missing, f"no SPROUT answers for {len(missing)} pool prompts: {missing[:3]}"
    return out


def flipped_seed(seed: str) -> str:
    """A seed that puts the two answers in the other order (judge_messages draws it)."""
    first = random.Random(seed).random() < 0.5
    return next(s for i in range(100) if (random.Random(s := f"{seed}#flip{i}").random() < 0.5)
                != first)  # fmt: skip
