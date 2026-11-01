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


async def test_fetch_after_stop_leaves_no_pooled_connections(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit(_record())
    await log.stop()
    await log.fetch_all()
    assert log._engine is not None
    assert log._engine.pool.checkedin() == 0


M3_COLUMNS = (
    "id TEXT PRIMARY KEY, ts FLOAT NOT NULL, model TEXT NOT NULL, provider TEXT NOT NULL, "
    "status INTEGER NOT NULL, stream BOOLEAN NOT NULL, latency_ms FLOAT NOT NULL, "
    "tokens_in INTEGER, tokens_cached INTEGER, tokens_out INTEGER, cost_usd FLOAT, "
    "baseline_cost_usd FLOAT, source TEXT NOT NULL, mode TEXT, key_id TEXT, difficulty FLOAT, "
    "confidence FLOAT, prefix_hash TEXT, cache_hit BOOLEAN NOT NULL, "
    "failover_count INTEGER NOT NULL, error_code TEXT"
)


def _m3_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE requests ({M3_COLUMNS})")
        conn.execute(
            "INSERT INTO requests (id, ts, model, provider, status, stream, latency_ms, source, "
            "cache_hit, failover_count) VALUES ('old', 1.0, 'm', 'openai', 200, 0, 5.0, "
            "'rules', 0, 0)"
        )


async def test_upgrades_m3_database_in_place(tmp_path: Path) -> None:
    _m3_database(tmp_path / "t.db")
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit(_record(5, effort="low", tokens_reasoning=12))
    await log.stop()
    rows = {r.id: r for r in await log.fetch_all()}
    assert set(rows) == {"old", "r5"}
    assert rows["old"].effort is None and rows["old"].tokens_reasoning is None
    assert (rows["r5"].effort, rows["r5"].tokens_reasoning) == ("low", 12)


async def test_upgrade_is_idempotent(tmp_path: Path) -> None:
    _m3_database(tmp_path / "t.db")
    first = RequestLog(_url(tmp_path))
    await first.start()
    await first.stop()
    second = RequestLog(_url(tmp_path))
    await second.start()
    assert second.added_columns == []
    await second.stop()
    assert first.added_columns == ["effort", "tokens_reasoning"]


async def test_new_fields_round_trip(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit(_record(1, effort="high", tokens_reasoning=3))
    await log.stop()
    [row] = await log.fetch_all()
    assert (row.effort, row.tokens_reasoning) == ("high", 3)
