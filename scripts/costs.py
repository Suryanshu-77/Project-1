"""Cost report from the request log: what each model cost vs. the baseline model.

Usage: uv run python scripts/costs.py   (reads TIERWISE_DB_URL, default sqlite+aiosqlite:///tierwise.db)
"""

import asyncio
from collections import defaultdict

from tierwise.config import Settings
from tierwise.store import RequestLog


async def report(db_url: str) -> None:
    log = RequestLog(db_url)
    await log.start()
    await log.stop()
    records = await log.fetch_all()

    rows: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    unpriced = 0
    for r in records:
        if r.cost_usd is None or r.baseline_cost_usd is None:
            unpriced += 1
            continue
        row = rows[r.model]
        row[0] += 1
        row[1] += r.tokens_in or 0
        row[2] += r.tokens_out or 0
        row[3] += r.cost_usd
        row[4] += r.baseline_cost_usd

    header = f"{'model':<24}{'requests':>9}{'tokens in':>11}{'tokens out':>11}"
    print(f"{header}{'cost $':>12}{'baseline $':>12}")
    for model, (n, tin, tout, cost, base) in sorted(rows.items()):
        print(f"{model:<24}{int(n):>9}{int(tin):>11}{int(tout):>11}{cost:>12.6f}{base:>12.6f}")
    total = sum(r[3] for r in rows.values())
    baseline = sum(r[4] for r in rows.values())
    saved = baseline - total
    pct = f" ({saved / baseline:.0%})" if baseline else ""
    print(f"\nTotal ${total:.6f} vs baseline ${baseline:.6f}: saved ${saved:.6f}{pct}")
    if unpriced:
        print(f"{unpriced} request(s) without usage (errors or streams without usage) not counted")


if __name__ == "__main__":
    asyncio.run(report(Settings().db_url))
