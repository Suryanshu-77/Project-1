"""Mixed traffic through the compose stack for a few minutes, so the Grafana dashboard has a
realistic picture: easy and hard prompts, streams, routing modes, overrides, multi-turn chats
(cache affinity), unknown models (4xx) and a low-rpm key that gets rate-limited (429).

Usage (stack up first; the two FAKE_* knobs add model latency and upstream failures):
    FAKE_DELAY_S=0.05 FAKE_FAIL_RATE=0.15 docker compose -f deploy/docker-compose.yml up -d --build
    uv run python scripts/traffic.py --minutes 15
"""

import argparse
import asyncio
import random
import time
from collections import Counter
from typing import Any

import httpx
from compose_smoke import GATEWAY, create_key, wait_for

EASY = [
    "hi, thanks",
    "What is the capital of France?",
    "Translate 'good morning' into Spanish.",
    "Give me a synonym for 'happy'.",
    "What's 12 times 9?",
    "Summarize in one line: the meeting moved to Thursday.",
]
HARD = [
    "Design a multi-tenant SaaS billing architecture and compare the trade-offs step by step.",
    "Debug this production outage: checkout latency spiked after a migration.",
    "Prove that the square root of 2 is irrational.",
    "Review this security-sensitive auth flow for token replay and session fixation risks.",
    "Design a sharding strategy for a 50 TB Postgres database with zero downtime.",
    "Derive the gradient of softmax cross-entropy and explain numerical stability.",
]
FOLLOW_UPS = ["Can you expand on that?", "What are the risks?", "Give me a concrete example."]


def _payload(rng: random.Random) -> tuple[str, dict[str, Any]]:
    """(kind, request body) for one request, drawn from the workload mix."""
    prompt = rng.choice(EASY if rng.random() < 0.55 else HARD)
    body: dict[str, Any] = {"model": "auto", "temperature": 0, "stream": rng.random() < 0.25,
                            "messages": [{"role": "user", "content": prompt}]}  # fmt: skip
    roll = rng.random()
    if roll < 0.12:
        body["model"] = rng.choice(["auto:cheap", "auto:quality"])
        return "mode", body
    if roll < 0.20:
        body["model"] = rng.choice(["tier:basic", "tier:moderate", "small", "large"])
        return "override", body
    if roll < 0.22:
        body["model"] = "no-such-model"
        return "bad model", body
    return "auto", body


async def _chat(client: httpx.AsyncClient, auth: dict[str, str], body: dict[str, Any]) -> int:
    resp = await client.post(f"{GATEWAY}/v1/chat/completions", json=body, headers=auth)
    await resp.aread()
    return resp.status_code


async def worker(client: httpx.AsyncClient, auth: dict[str, str], seed: int, until: float,
                 counts: Counter[str]) -> None:  # fmt: skip
    rng = random.Random(seed)
    while time.monotonic() < until:
        if rng.random() < 0.12:  # a short conversation: later turns stay on the warm model
            history = [{"role": "user", "content": rng.choice(HARD)}]
            for turn in range(3):
                body = {"model": "auto", "temperature": 0, "messages": history}
                status = await _chat(client, auth, body)
                counts[f"chat {status}"] += 1
                history += [{"role": "assistant", "content": "Here is a detailed answer."},
                            {"role": "user", "content": FOLLOW_UPS[turn]}]  # fmt: skip
        else:
            kind, body = _payload(rng)
            counts[f"{kind} {await _chat(client, auth, body)}"] += 1
        await asyncio.sleep(rng.uniform(0.5, 2.0))


async def bursts(client: httpx.AsyncClient, auth: dict[str, str], until: float,
                 counts: Counter[str]) -> None:  # fmt: skip
    """A key with --rpm 20 that now and then sends a burst: the overflow gets 429."""
    while time.monotonic() < until:
        await asyncio.sleep(random.uniform(40, 80))
        body = {"model": "auto", "messages": [{"role": "user", "content": "ping"}]}
        statuses = await asyncio.gather(*(_chat(client, auth, body) for _ in range(30)))
        counts.update(f"burst {s}" for s in statuses)


async def run(minutes: float, workers: int) -> None:
    main_key = {"Authorization": f"Bearer {create_key('traffic', '--allow-shadow')}"}
    small_key = {"Authorization": f"Bearer {create_key('bursty', '--rpm', '20')}"}
    async with httpx.AsyncClient(timeout=60) as client:
        live = lambda: httpx.get(f"{GATEWAY}/v1/models", headers=main_key).is_success  # noqa: E731
        wait_for("traffic key live", live, 10)
        until = time.monotonic() + 60 * minutes
        counts: Counter[str] = Counter()
        tasks = [worker(client, main_key, seed, until, counts) for seed in range(workers)]
        tasks.append(bursts(client, small_key, until, counts))
        progress = asyncio.ensure_future(asyncio.gather(*tasks))
        while not progress.done():
            await asyncio.sleep(60)
            print(f"{sum(counts.values())} requests: {dict(sorted(counts.items()))}", flush=True)
        await progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    wait_for("gateway /healthz", lambda: httpx.get(f"{GATEWAY}/healthz").is_success, 120)
    asyncio.run(run(args.minutes, args.workers))


if __name__ == "__main__":
    main()
