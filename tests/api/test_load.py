"""Spec §7 concurrency check: a burst of 200 requests against fake providers."""

import asyncio
import logging
import statistics
from pathlib import Path

import pytest

from tests.api.helpers import body, gateway, make_key


async def test_200_concurrent_requests(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    async with gateway(db_dir=tmp_path, require_api_key=True) as (app, client):
        keys = [(await make_key(app, budget=1000.0, rpm=1000))[0] for _ in range(3)]
        responses = await asyncio.gather(*(
            client.post(
                "/v1/chat/completions",
                json=body("gpt-4.1-mini"),
                headers={"Authorization": f"Bearer {keys[i % 3]}"},
            )
            for i in range(200)
        ))  # fmt: skip
        log = app.state.log
    assert [r.status_code for r in responses] == [200] * 200

    rows = await log.fetch_all()
    usage = await log.fetch_usage()
    assert len(rows) == 200 and log.dropped == 0
    assert "database is locked" not in caplog.text
    # every debit landed exactly once: the ledger matches the per-request costs
    assert sum(spent for _, _, spent, _ in usage) == pytest.approx(
        sum(r.cost_usd or 0 for r in rows), abs=1e-9
    )
    assert sum(count for _, _, _, count in usage) == 200
    assert statistics.median(r.latency_ms for r in rows) < 30
