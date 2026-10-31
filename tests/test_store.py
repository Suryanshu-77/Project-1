import asyncio
import sqlite3
from pathlib import Path

from tierwise.store import RequestLog, RequestRecord


def _record(i: int = 0, **overrides: object) -> RequestRecord:
    fields: dict[str, object] = {
        "id": f"r{i}",
        "ts": 1000.0 + i,
        "model": "m",
        "provider": "openai",
        "status": 200,
        "stream": False,
        "latency_ms": 12.5,
        "tokens_in": 10,
        "tokens_cached": 4,
        "tokens_out": 2,
        "cost_usd": 0.001,
        "baseline_cost_usd": 0.01,
    }
    fields.update(overrides)
    return RequestRecord(**fields)  # type: ignore[arg-type]


def _url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{(tmp_path / 't.db').as_posix()}"


async def test_round_trip(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit(_record())
    await log.stop()
    assert await log.fetch_all() == [_record()]


async def test_wal_mode_enabled(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    await log.stop()
    with sqlite3.connect(tmp_path / "t.db") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


async def test_concurrent_burst_no_lock_errors(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()

    async def one(i: int) -> None:
        await asyncio.sleep(0)
        log.submit(_record(i))

    await asyncio.gather(*(one(i) for i in range(200)))
    await log.stop()
    assert len(await log.fetch_all()) == 200
    assert log.dropped == 0


async def test_full_queue_drops_and_counts(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path), max_queue=2)
    for i in range(5):
        log.submit(_record(i))
    assert log.dropped == 3


async def test_stop_drains_pending(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path), flush_interval=5.0)
    await log.start()
    for i in range(50):
        log.submit(_record(i))
    await log.stop()
    assert len(await log.fetch_all()) == 50


async def test_null_usage_fields_persist_as_none(tmp_path: Path) -> None:
    record = _record(
        tokens_in=None, tokens_cached=None, tokens_out=None, cost_usd=None, baseline_cost_usd=None
    )
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit(record)
    await log.stop()
    assert await log.fetch_all() == [record]
