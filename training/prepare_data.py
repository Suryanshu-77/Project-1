"""Labels for the learned router from routellm/gpt4_dataset (Apache-2.0).

Each row has a prompt and a GPT-4 judge's 1-5 grade of Mixtral's answer. A low grade means the
cheap model wasn't good enough, so the prompt needed a strong model.

Usage: uv run --group train python -m training.prepare_data [--limit 40000]
Writes training/data/{train,val,test}.jsonl with {"prompt", "label"}.
"""

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

DATASET = "routellm/gpt4_dataset"
STRONG_IF_SCORE_BELOW = 4  # Mixtral graded 1-3: the strong model was needed
DATA_DIR = Path("training/data")
Split = Literal["train", "val", "test"]


def label(mixtral_score: int) -> int:
    return 1 if mixtral_score < STRONG_IF_SCORE_BELOW else 0


def normalize(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _bucket(prompt: str) -> int:
    return int(hashlib.sha256(normalize(prompt).encode()).hexdigest()[:8], 16)


def split_of(prompt: str) -> Split:
    """By prompt hash: duplicates always land together, so no test prompt leaks into train."""
    bucket = _bucket(prompt) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def prepare(rows: Iterable[dict[str, Any]], limit: int | None) -> dict[Split, list[dict[str, Any]]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt, score = row.get("prompt"), row.get("mixtral_score")
        if not isinstance(prompt, str) or not prompt.strip() or score is None:
            continue
        unique.setdefault(normalize(prompt), {"prompt": prompt, "label": label(int(score))})
    kept = sorted(unique.items(), key=lambda kv: _bucket(kv[0]))  # hash order = a random sample
    if limit is not None:
        kept = kept[:limit]
    out: dict[Split, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for _, row in kept:
        out[split_of(row["prompt"])].append(row)
    return out


def _download() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download  # train group only

    rows: list[dict[str, Any]] = []
    for name in ("train.jsonl", "valid.jsonl"):
        path = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=name)
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def main(limit: int | None) -> None:
    splits = prepare(_download(), limit)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with open(DATA_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        positive = sum(r["label"] for r in rows) / max(len(rows), 1)
        print(f"{split}: {len(rows)} prompts, {positive:.1%} need a strong model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40_000)
    main(parser.parse_args().limit)
