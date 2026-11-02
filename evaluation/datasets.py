"""Eval datasets as EvalRows: a prompt plus the quality and cost of a cheap and a strong answer.

- Held-out split of routellm/gpt4_dataset (Apache-2.0): Mixtral-8x7B (cheap) vs GPT-4 (strong).
  The judge graded only Mixtral's answer, so the GPT-4 answer is the reference (quality 1.0).
- SPROUT (CARROT-LLM-Routing/SPROUT, no stated license): downloaded at eval time, never committed.
"""

import json
import math
from pathlib import Path
from typing import Any

from evaluation.metrics import EvalRow
from training.prepare_data import split_of

# USD per 1M tokens (input, output), list prices of the snapshots each dataset used:
# gpt-4: gpt-4-1106-preview, OpenAI pricing at release; mixtral-8x7b: Together AI;
# gpt-4o: gpt-4o-2024-08-06 and gpt-4o-mini, OpenAI pricing.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4": (10.0, 30.0),
    "mixtral-8x7b": (0.60, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}
SPROUT_REPO = "CARROT-LLM-Routing/SPROUT"
SPROUT_PRICE_KEY = {"openai-gpt-4o": "gpt-4o", "openai-gpt-4o-mini": "gpt-4o-mini"}
EVAL_FIELDS = ("mixtral_score", "prompt_chars", "gpt4_chars", "mixtral_chars")


def token_cost(model: str, tokens_in: float, tokens_out: float) -> float:
    price_in, price_out = PRICES[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def _tokens(chars: int) -> int:
    return math.ceil(chars / 4)


def row_from_gpt4(rec: dict[str, Any]) -> EvalRow:
    tokens_in = _tokens(rec["prompt_chars"])
    return EvalRow(
        prompt=rec["prompt"],
        q_cheap=(int(rec["mixtral_score"]) - 1) / 4,
        q_strong=1.0,
        cost_cheap=token_cost("mixtral-8x7b", tokens_in, _tokens(rec["mixtral_chars"])),
        cost_strong=token_cost("gpt-4", tokens_in, _tokens(rec["gpt4_chars"])),
    )


def load_gpt4_test(path: Path = Path("training/data/test.jsonl")) -> list[EvalRow]:
    rows = []
    with path.open(encoding="utf-8") as f:  # not splitlines(): prompts may hold U+2028
        lines = [line for line in f if line.strip()]
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if any(field not in rec for field in EVAL_FIELDS):
            raise ValueError(f"{path} predates M5; re-run `python -m training.prepare_data`")
        if split_of(rec["prompt"]) != "test":
            raise ValueError(f"{path} row {i} is not a test-split prompt (train/test leakage)")
        rows.append(row_from_gpt4(rec))
    return rows


def row_from_sprout(rec: dict[str, Any], cheap: str, strong: str) -> EvalRow | None:
    """None when either model's answer (or its score) is missing from the record."""
    answers = {name: rec.get(name) for name in (cheap, strong)}
    if any(a is None or a.get("score") is None for a in answers.values()):
        return None
    for name, answer in answers.items():
        if not 0.0 <= float(answer["score"]) <= 1.0:
            raise ValueError(f"SPROUT score {answer['score']} for {name} is outside [0, 1]")

    def cost(name: str) -> float:
        answer = answers[name]
        return token_cost(
            SPROUT_PRICE_KEY[name], answer["num_input_tokens"], answer["num_output_tokens"]
        )

    return EvalRow(
        prompt=rec["prompt"],
        q_cheap=float(answers[cheap]["score"]),
        q_strong=float(answers[strong]["score"]),
        cost_cheap=cost(cheap),
        cost_strong=cost(strong),
    )


def load_sprout(
    split: str = "test",
    cheap: str = "openai-gpt-4o-mini",
    strong: str = "openai-gpt-4o",
    limit: int | None = None,
) -> tuple[list[EvalRow], int]:
    """(rows, dropped). Downloads SPROUT's parquet files into the Hugging Face cache."""
    import pyarrow.parquet as pq  # train group only
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(SPROUT_REPO, repo_type="dataset", allow_patterns=[f"*{split}*"]))
    rows: list[EvalRow] = []
    dropped = 0
    # the cache may hold other splits too: read only this one's files
    for path in sorted(root.rglob(f"{split}-*.parquet")):
        for rec in pq.read_table(path, columns=["prompt", cheap, strong]).to_pylist():
            row = row_from_sprout(rec, cheap, strong)
            if row is None:
                dropped += 1
            else:
                rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows, dropped
    if not rows:
        raise ValueError(f"no SPROUT {split} rows found under {root}")
    return rows, dropped
