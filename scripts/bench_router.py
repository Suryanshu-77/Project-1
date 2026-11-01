"""Learned-router latency: p50/p95 of one encoder + classifier pass, single request, on CPU.

Usage: uv run python scripts/bench_router.py   (exits 1 if p50 >= 10 ms)
"""

import json
import statistics
import sys
import time
from pathlib import Path

from tierwise.config import Settings
from tierwise.router.learned import load_learned
from tierwise.schema import ChatRequest

WARMUP, RUNS, BUDGET_MS = 20, 500, 10.0


def main() -> int:
    router = load_learned(Settings())
    if router is None:
        print("learned router unavailable (see the warning above)")
        return 1
    test = Path("training/data/test.jsonl")
    prompts = (
        [json.loads(line)["prompt"] for line in test.open(encoding="utf-8")][:RUNS]
        if test.exists()
        else ["Explain the CAP theorem with an example."] * RUNS
    )
    requests = [
        ChatRequest.model_validate({"messages": [{"role": "user", "content": p}]}) for p in prompts
    ]
    for req in requests[:WARMUP]:
        router._score_sync(req)
    times = []
    for i in range(RUNS):
        start = time.perf_counter()
        router._score_sync(requests[i % len(requests)])
        times.append((time.perf_counter() - start) * 1000)
    router.close()
    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18]
    print(f"{RUNS} runs: p50 {p50:.2f} ms, p95 {p95:.2f} ms (budget p50 < {BUDGET_MS} ms)")
    return 0 if p50 < BUDGET_MS else 1


if __name__ == "__main__":
    sys.exit(main())
