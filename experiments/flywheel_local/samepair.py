"""Stages 1 and 2 of PROTOCOL.md: Claude Haiku judges SPROUT's own gpt-4o-mini vs gpt-4o answers.

The pairs are the stage 0 pool, in its fixed order; the answer texts come from SPROUT, so
nothing is generated. Every call is stored in the `same_pair` table of flywheel.db: re-running
never pays twice, and each stage stops at its hard cap.

    uv run --group retrain --group train python experiments/flywheel_local/samepair.py pilot
"""

import argparse
import asyncio
import hashlib
import json
import random
import sqlite3
import sys
import time
from functools import partial
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent

from rejudge import env  # noqa: E402
from stage0 import build_pool, gold  # noqa: E402

from evaluation.datasets import SPROUT_REPO, load_sprout  # noqa: E402
from evaluation.metrics import EvalRow  # noqa: E402
from tierwise.flywheel import _transcript, judge_messages, parse_score  # noqa: E402
from tierwise.providers.base import ProviderError  # noqa: E402
from tierwise.providers.openai import OpenAIProvider  # noqa: E402
from tierwise.reliability.retry import RetryPolicy, with_retries  # noqa: E402
from tierwise.schema import ChatRequest  # noqa: E402
from training.prepare_data import normalize  # noqa: E402

CHEAP, STRONG = "openai-gpt-4o-mini", "openai-gpt-4o"
PILOT, CHECKS = 400, 20
PRICE_IN, PRICE_OUT = 0.10, 0.50  # per 1M tokens: claude-haiku-5.5 on AICredits


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


async def judge(jobs: list[tuple[str, str, str, str, str]], cap_usd: float) -> None:
    """jobs: (key, prompt, cheap answer, strong answer, seed). Stores each result as it lands."""
    cfg = env()
    db = sqlite3.connect(HERE / "flywheel.db")
    db.execute("create table if not exists same_pair (key text primary key, score integer, "
               "error text, cost_usd real, input_tokens integer, ts real)")  # fmt: skip
    done = {k for (k,) in db.execute("select key from same_pair")}
    spent = db.execute("select coalesce(sum(cost_usd), 0) from same_pair").fetchone()[0]
    todo = [j for j in jobs if j[0] not in done]
    print(f"{len(todo)} calls to make ({len(jobs) - len(todo)} already stored); "
          f"spent so far ${spent:.4f}, cap ${cap_usd:.2f}", flush=True)  # fmt: skip
    lock, gate = asyncio.Lock(), asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=120) as client:
        provider = OpenAIProvider(client, cfg["JUDGE_API_KEY"], cfg["JUDGE_BASE_URL"])

        async def one(key: str, prompt: str, cheap: str, strong: str, seed: str) -> None:
            nonlocal spent
            async with gate:
                if spent >= cap_usd:
                    return
                request = ChatRequest.model_validate(
                    {"messages": [{"role": "user", "content": prompt}]})  # fmt: skip
                messages = judge_messages(_transcript(request), cheap, strong, seed)
                grading = ChatRequest(model=cfg["JUDGE_MODEL"], temperature=0, messages=messages)
                score, error, cost, tokens = None, None, 0.0, 0
                try:
                    call = partial(provider.chat, grading, cfg["JUDGE_MODEL"])
                    reply = await with_retries(call, RetryPolicy())
                    tokens = reply.usage.input_tokens
                    cost = (tokens * PRICE_IN + reply.usage.output_tokens * PRICE_OUT) / 1e6
                    score = parse_score(reply.content or "")
                except (ProviderError, ValueError) as exc:
                    error = str(exc)[:300]
                async with lock:
                    spent += cost
                    db.execute("insert or replace into same_pair values (?, ?, ?, ?, ?, ?)",
                               (key, score, error, cost, tokens, time.time()))  # fmt: skip
                    db.commit()

        await asyncio.gather(*(one(*j) for j in todo))
    print(f"spent ${spent:.4f} in all", flush=True)
    if spent >= cap_usd:
        print("stopped at the cap: re-run to finish (stored calls are not repeated)")


def pilot(cap_usd: float, dry_run: bool) -> None:
    train_rows, _ = load_sprout("train")
    test_rows, _ = load_sprout("test")
    pool = build_pool(train_rows, test_rows)[0][:PILOT]
    texts = answers(pool)
    jobs = []
    for r in pool:
        cheap, strong = texts[pid(r.prompt)]
        jobs.append((f"label:{pid(r.prompt)}", r.prompt, cheap, strong, pid(r.prompt)))
    for r in pool[:CHECKS]:
        p, (_, strong) = pid(r.prompt), texts[pid(r.prompt)]
        jobs.append((f"identical:{p}", r.prompt, strong, strong, p))  # gpt-4o against itself
        cheap = texts[p][0]
        jobs.append((f"flip:{p}", r.prompt, cheap, strong, flipped_seed(p)))  # A and B swapped
    if dry_run:  # build everything, call nothing
        flips = [j for j in jobs if j[0].startswith("flip:")]
        swapped = all((random.Random(j[4]).random() < 0.5) != (random.Random(j[0][5:]).random()
                      < 0.5) for j in flips)  # fmt: skip
        chars = np.array([len(j[1]) + len(j[2]) + len(j[3]) for j in jobs])
        print(f"dry run: {len(jobs)} jobs ({PILOT} labels, {CHECKS} identical, {CHECKS} flips); "
              f"every flip swaps the order: {swapped}; empty answers: "
              f"{sum(not j[2] or not j[3] for j in jobs)}; ~{chars.mean() / 4 + 250:.0f} input "
              f"tokens per call, ~${len(jobs) * 0.0003:.2f} at the first run's $0.0003 per call")
        return
    asyncio.run(judge(jobs, cap_usd))

    db = sqlite3.connect(HERE / "flywheel.db")
    got = dict(db.execute("select key, score from same_pair where score is not null").fetchall())
    errors = db.execute("select count(*) from same_pair where score is null").fetchone()[0]
    spent = db.execute("select sum(cost_usd), avg(input_tokens) from same_pair").fetchone()
    labeled = [r for r in pool if f"label:{pid(r.prompt)}" in got]
    scores = np.array([got[f"label:{pid(r.prompt)}"] for r in labeled])
    y = np.array([gold(r) for r in labeled])
    identical = [got.get(f"identical:{pid(r.prompt)}") for r in pool[:CHECKS]]
    flips = [(got.get(f"label:{pid(r.prompt)}"), got.get(f"flip:{pid(r.prompt)}"))
             for r in pool[:CHECKS]]  # fmt: skip
    ident_ok = sum(s is not None and s >= 4 for s in identical)
    flip_ok = sum(a is not None and b is not None and abs(a - b) <= 1 for a, b in flips)
    auc = float(roc_auc_score(y, -scores))
    rates = {c: float(np.mean(scores < c)) for c in (2, 3, 4, 5)}
    cutoff = min(rates, key=lambda c: (abs(rates[c] - y.mean()), c))
    checks = {"identical >= 4": (ident_ok, 18), "order flip within 1": (flip_ok, 17)}
    go = ident_ok >= 18 and flip_ok >= 17 and auc >= 0.70

    print(f"\npilot: {len(labeled)}/{PILOT} pairs scored, {errors} calls failed; "
          f"${spent[0]:.4f} spent, {spent[1]:.0f} input tokens per call on average")  # fmt: skip
    print(f"score counts: {dict(sorted(zip(*np.unique(scores, return_counts=True), strict=True)))}")
    for name, (value, need) in checks.items():
        print(f"  {name}: {value}/{CHECKS} (need {need})  {'ok' if value >= need else 'FAIL'}")
    print(f"  AUC of Haiku's score against SPROUT's verdict (G, {y.mean():.1%} positive): "
          f"{auc:.3f} (need 0.70)  {'ok' if auc >= 0.70 else 'FAIL'}")  # fmt: skip
    shown = ", ".join(f"score<{c}: {v:.1%}" for c, v in rates.items())
    print(f"  positive rate by cutoff: {shown}")
    print(f"  cutoff fixed for stage 2: needs strong = score < {cutoff}")
    print(f"\ndecision: {'GO to stage 2' if go else 'NO-GO: stop, stage 2 is not run'}")
    out = {"scored": len(labeled), "errors": errors, "spent_usd": spent[0], "auc": auc,
           "identical_ok": ident_ok, "flip_ok": flip_ok, "rates": rates, "cutoff": cutoff,
           "g_rate": float(y.mean()), "go": go}  # fmt: skip
    (HERE / "stage1_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["pilot"])
    parser.add_argument("--cap-usd", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true", help="build the jobs, make no calls")
    args = parser.parse_args()
    pilot(args.cap_usd, args.dry_run)


if __name__ == "__main__":
    main()
