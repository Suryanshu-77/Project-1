"""Re-judge the stored answer pairs with a stronger, paid judge; nothing is regenerated.

Same prompt, same two answers, same seeded A/B order as collect.py's mistral:7b judge, so the
judge is the only thing that changes. Scores go to a `rejudge` table next to the samples;
re-running skips pairs already scored, and the run stops at a hard dollar cap.

Settings come from the repo's .env (any OpenAI-compatible endpoint):
    JUDGE_API_KEY=...   JUDGE_BASE_URL=https://api.aicredits.in/v1
    JUDGE_MODEL=anthropic/claude-haiku-5.5
Run from the repo root:
    uv run python experiments/flywheel_local/rejudge.py --limit 1      # one call, to check
    uv run python experiments/flywheel_local/rejudge.py --cap-usd 0.50
"""

import argparse
import asyncio
import json
import sqlite3
import time
from functools import partial
from pathlib import Path

import httpx

from tierwise.flywheel import _transcript, judge_messages, parse_score
from tierwise.providers.base import ProviderError
from tierwise.providers.openai import OpenAIProvider
from tierwise.registry import load_registry
from tierwise.reliability.retry import RetryPolicy, with_retries
from tierwise.schema import ChatRequest

HERE = Path(__file__).parent
REGISTRY = load_registry(HERE / "registry.yaml")
DB = HERE / "flywheel.db"


def env(path: Path = Path(".env")) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
             if "=" in line and not line.lstrip().startswith("#"))  # fmt: skip
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


def pairs(db: sqlite3.Connection, judge: str) -> list[tuple[str, str, str, str]]:
    """(sample id, transcript, cheap answer, strong answer) for every labeled sample not yet
    scored by `judge`, oriented exactly as tierwise.flywheel._label does."""
    done = {r[0] for r in db.execute("select sample_id from rejudge where judge = ?", (judge,))}
    out = []
    for sid, request_json, routed, routed_answer, contrast, contrast_answer in db.execute(
        "select id, request_json, routed_model, routed_answer, contrast_model, contrast_answer "
        "from shadow_samples where status = 'labeled' order by ts"
    ):
        if sid in done:
            continue
        spec = REGISTRY.get(contrast)
        assert spec is not None and REGISTRY.get(routed) is not None
        cheap_is_routed = spec.tier == "flagship" or spec.id == REGISTRY.baseline_model
        cheap, strong = ((routed_answer, contrast_answer) if cheap_is_routed
                         else (contrast_answer, routed_answer))  # fmt: skip
        request = ChatRequest.model_validate_json(request_json)
        out.append((sid, _transcript(request), cheap, strong))
    return out


async def run(limit: int | None, cap_usd: float, price_in: float, price_out: float,
              workers: int) -> None:  # fmt: skip
    cfg = env()
    judge = cfg["JUDGE_MODEL"]
    db = sqlite3.connect(DB)
    db.execute("create table if not exists rejudge (sample_id text, judge text, score integer, "
               "error text, cost_usd real, ts real, primary key (sample_id, judge))")  # fmt: skip
    todo = pairs(db, judge)[:limit]
    print(f"{judge}: {len(todo)} pairs to score, cap ${cap_usd:.2f}", flush=True)
    spent, scored, failed = 0.0, 0, 0
    lock, gate = asyncio.Lock(), asyncio.Semaphore(workers)

    async with httpx.AsyncClient(timeout=120) as client:
        provider = OpenAIProvider(client, cfg["JUDGE_API_KEY"], cfg["JUDGE_BASE_URL"])

        async def one(sid: str, transcript: str, cheap: str, strong: str) -> None:
            nonlocal spent, scored, failed
            async with gate:
                if spent >= cap_usd:
                    return
                messages = judge_messages(transcript, cheap, strong, sid)
                request = ChatRequest(model=judge, temperature=0, messages=messages)
                score, error, cost = None, None, 0.0
                try:
                    reply = await with_retries(partial(provider.chat, request, judge),
                                               RetryPolicy())  # fmt: skip
                    usage = reply.usage
                    cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6
                    score = parse_score(reply.content or "")
                except (ProviderError, ValueError) as exc:
                    error = str(exc)[:300]
                async with lock:
                    spent += cost
                    scored, failed = scored + (score is not None), failed + (score is None)
                    db.execute("insert or replace into rejudge values (?, ?, ?, ?, ?, ?)",
                               (sid, judge, score, error, cost, time.time()))  # fmt: skip
                    db.commit()
                    if error and failed <= 3:
                        print(f"  failed {sid}: {error}", flush=True)
                    if (scored + failed) % 50 == 0:
                        print(f"  {scored + failed}/{len(todo)}  ${spent:.4f}", flush=True)

        await asyncio.gather(*(one(*p) for p in todo))
    total = db.execute("select count(score), sum(cost_usd) from rejudge where judge = ?",
                       (judge,)).fetchone()  # fmt: skip
    print(f"this run: scored {scored}, failed {failed}, ${spent:.4f}; "
          f"all runs: {total[0]} scored, ${total[1] or 0:.4f}")
    if spent >= cap_usd:
        print(f"stopped at the ${cap_usd:.2f} cap; re-run to continue")
    print(json.dumps({"judge": judge, "scores": dict(db.execute(
        "select score, count(*) from rejudge where judge = ? and score is not null "
        "group by score", (judge,)).fetchall())}))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="score at most this many pairs")
    parser.add_argument("--cap-usd", type=float, default=0.50, help="stop spending here")
    # AICredits lists claude-haiku-5.5 at $0.10 in / $0.50 out per 1M tokens (before markup)
    parser.add_argument("--price-in", type=float, default=0.10)
    parser.add_argument("--price-out", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.cap_usd, args.price_in, args.price_out, args.workers))


if __name__ == "__main__":
    main()
