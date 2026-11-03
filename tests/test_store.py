import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from tierwise.store import RequestLog, RequestRecord, ShadowSample, metadata


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


PG_URL = os.environ.get("TIERWISE_TEST_PG_URL")


@pytest.fixture(params=["sqlite", "postgres"])
async def db_url(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[str]:
    """The same test on SQLite and, when TIERWISE_TEST_PG_URL is set (CI), on Postgres."""
    if request.param == "sqlite":
        yield _url(tmp_path)
        return
    if not PG_URL:
        pytest.skip("set TIERWISE_TEST_PG_URL to run against Postgres")
    engine = create_async_engine(PG_URL)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)  # each test starts from empty tables
    await engine.dispose()
    yield PG_URL


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


# --- M7a: keys and never-dropped debits --------------------------------------------------


async def test_create_list_revoke_key(db_url: str) -> None:
    log = RequestLog(db_url)
    await log.start()
    row = await log.create_key("team-a", "h" * 64, "tw-abcdefg", 50.0, 60)
    [listed] = await log.list_keys()
    assert listed == row and listed.revoked_ts is None
    assert await log.revoke_key(row.id) is True
    assert await log.revoke_key(row.id) is False
    assert await log.revoke_key("nope") is False
    [revoked] = await log.list_keys()
    assert revoked.revoked_ts is not None
    await log.stop()


async def test_duplicate_key_hash_rejected(db_url: str) -> None:
    import pytest
    from sqlalchemy.exc import IntegrityError

    log = RequestLog(db_url)
    await log.start()
    await log.create_key("a", "same", "tw-1", None, None)
    with pytest.raises(IntegrityError):
        await log.create_key("b", "same", "tw-2", None, None)
    await log.stop()


async def test_debits_aggregate_per_key_and_month(db_url: str) -> None:
    log = RequestLog(db_url)
    await log.start()
    log.submit_debit("a", "2026-11", 0.01)
    log.submit_debit("a", "2026-11", 0.02)
    log.submit_debit("a", "2026-12", 0.05)
    await log.stop()
    usage = sorted(await log.fetch_usage())
    assert [(k, m, round(s, 9), n) for k, m, s, n in usage] == [
        ("a", "2026-11", 0.03, 2),
        ("a", "2026-12", 0.05, 1),
    ]


async def test_debits_accumulate_across_flushes(db_url: str) -> None:
    log = RequestLog(db_url, flush_interval=0.01)
    await log.start()
    log.submit_debit("a", "2026-11", 0.01)
    await asyncio.sleep(0.1)
    assert not log.pending_debits
    log.submit_debit("a", "2026-11", 0.02)
    await log.stop()
    [(_, _, spent, count)] = await log.fetch_usage()
    assert (round(spent, 9), count) == (0.03, 2)


async def test_debits_survive_failed_flush(db_url: str) -> None:
    log = RequestLog(db_url, flush_interval=0.01)
    await log.start()
    real = log._upsert_debits
    calls = {"n": 0}

    async def flaky(conn: object, rows: object) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is down")
        await real(conn, rows)  # type: ignore[arg-type]

    log._upsert_debits = flaky  # type: ignore[method-assign]
    log.submit_debit("a", "2026-11", 0.04)
    await asyncio.sleep(0.1)
    await log.stop()
    [(_, _, spent, count)] = await log.fetch_usage()
    assert (round(spent, 9), count) == (0.04, 1)  # written once, after the retry
    assert calls["n"] >= 2


async def test_debits_never_dropped_when_telemetry_queue_full(db_url: str) -> None:
    log = RequestLog(db_url, max_queue=1)
    log.submit(_record(1))  # not started: the queue stays full
    log.submit(_record(2))
    log.submit_debit("a", "2026-11", 0.5)
    assert log.dropped == 1
    await log.start()
    await log.stop()
    [(_, _, spent, _)] = await log.fetch_usage()
    assert spent == 0.5


async def test_load_month_spend(db_url: str) -> None:
    log = RequestLog(db_url)
    await log.start()
    log.submit_debit("a", "2026-11", 0.25)
    log.submit_debit("b", "2026-10", 9.0)
    await log.stop()
    await log.start()
    assert await log.load_month_spend("2026-11") == {"a": 0.25}
    await log.stop()


async def test_stop_gives_up_on_unwritable_debits_and_logs_them(
    db_url: str, caplog: "pytest.LogCaptureFixture"
) -> None:
    import logging
    import time

    caplog.set_level(logging.ERROR)
    log = RequestLog(db_url, flush_interval=0.01)
    await log.start()

    async def down(conn: object, rows: object) -> None:
        raise RuntimeError("disk full")

    log._upsert_debits = down  # type: ignore[method-assign]
    log.submit_debit("key9", "2026-11", 0.42)
    started = time.monotonic()
    await asyncio.wait_for(log.stop(), timeout=5)
    assert time.monotonic() - started < 3
    assert "key9" in caplog.text and "2026-11" in caplog.text and "0.42" in caplog.text


async def test_concurrent_flushes_sum_not_overwrite(db_url: str) -> None:
    first, second = RequestLog(db_url), RequestLog(db_url)
    await first.start()
    await second.start()
    first.submit_debit("k", "2026-11", 0.10)
    second.submit_debit("k", "2026-11", 0.10)
    await asyncio.gather(first.stop(), second.stop())
    await first.start()
    [(_, _, spent, count)] = await first.fetch_usage()
    await first.stop()
    assert (round(spent, 9), count) == (0.2, 2)


def _sample(i: int = 0, **overrides: object) -> ShadowSample:
    fields: dict[str, object] = {
        "id": f"s{i}", "ts": 1000.0 + i, "key_id": "k1",
        "request_json": '{"model": "auto", "messages": [{"role": "user", "content": "hi"}]}',
        "routed_model": "small", "routed_answer": "hello", "difficulty": 0.2,
        "confidence": 0.9, "source": "rules",
    }  # fmt: skip
    fields.update(overrides)
    return ShadowSample(**fields)  # type: ignore[arg-type]


async def test_sample_roundtrip(db_url: str) -> None:
    log = RequestLog(db_url)
    await log.start()
    assert log.submit_sample(_sample(1)) and log.submit_sample(_sample(0))
    await log.stop()
    await log.start()
    pending = await log.pending_samples(10)
    await log.stop()
    assert [s.id for s in pending] == ["s0", "s1"]  # oldest first
    assert pending[0] == _sample(0)


async def test_save_label_moves_out_of_pending(db_url: str) -> None:
    log = RequestLog(db_url)
    await log.start()
    log.submit_sample(_sample(0))
    log.submit_sample(_sample(1))
    await log.stop()
    await log.start()
    labeled = replace(_sample(0), status="labeled", contrast_model="large",
                      contrast_answer="better", judge_score=2, judge_score_strong=5,
                      needs_strong=True,
                      label_cost_usd=0.01, labeled_ts=2000.0)  # fmt: skip
    await log.save_label(labeled)
    assert [s.id for s in await log.pending_samples(10)] == ["s1"]
    assert await log.labeled_samples() == [labeled]
    await log.stop()


async def test_label_spend_sums_month(db_url: str) -> None:
    log = RequestLog(db_url)
    await log.start()
    for i in range(3):
        log.submit_sample(_sample(i))
    await log.stop()
    await log.start()
    nov, dec = 1793577600.0, 1796256000.0  # 2026-11-02 and 2026-12-03 (UTC)
    await log.save_label(replace(_sample(0), status="labeled", label_cost_usd=0.5, labeled_ts=nov))
    await log.save_label(replace(_sample(1), status="failed", label_cost_usd=0.25, labeled_ts=nov))
    await log.save_label(replace(_sample(2), status="labeled", label_cost_usd=4.0, labeled_ts=dec))
    assert await log.label_spend("2026-11") == pytest.approx(0.75)  # failed labels cost too
    await log.stop()


async def test_sample_queue_full_drops_and_counts(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path), max_queue=2)
    assert log.submit_sample(_sample(0)) and log.submit_sample(_sample(1))
    assert not log.submit_sample(_sample(2))
    assert log.dropped == 1


async def test_existing_db_gains_allow_shadow_column(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    await log.create_key("old", "h1", "tw-old", None, None)
    await log.stop()
    with sqlite3.connect(tmp_path / "t.db") as conn:  # an M8 database: no allow_shadow yet
        conn.execute("ALTER TABLE api_keys DROP COLUMN allow_shadow")
    upgraded = RequestLog(_url(tmp_path))
    await upgraded.start()
    assert upgraded.added_columns == ["allow_shadow"]
    new = await upgraded.create_key("new", "h2", "tw-new", None, None, allow_shadow=True)
    keys = {k.name: k for k in await upgraded.list_keys()}
    await upgraded.stop()
    assert keys["old"].allow_shadow is False and keys["new"].allow_shadow is True
    assert new.allow_shadow


async def test_existing_db_gains_strong_judge_score_column(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit_sample(_sample(0))
    await log.stop()
    with sqlite3.connect(tmp_path / "t.db") as conn:  # a 0.9 database: one judge score per label
        conn.execute("ALTER TABLE shadow_samples DROP COLUMN judge_score_strong")
    upgraded = RequestLog(_url(tmp_path))
    await upgraded.start()
    assert upgraded.added_columns == ["judge_score_strong"]
    [old] = await upgraded.pending_samples(10)
    await upgraded.save_label(replace(old, status="labeled", judge_score=2, judge_score_strong=5,
                                      needs_strong=True, labeled_ts=2000.0))  # fmt: skip
    [labeled] = await upgraded.labeled_samples()
    await upgraded.stop()
    assert (old.judge_score_strong, labeled.judge_score_strong) == (None, 5)


async def test_existing_db_gains_judge_model_column(tmp_path: Path) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit_sample(_sample(0))
    await log.stop()
    with sqlite3.connect(tmp_path / "t.db") as conn:  # a 0.10 database: no judge recorded
        conn.execute("ALTER TABLE shadow_samples DROP COLUMN judge_model")
    upgraded = RequestLog(_url(tmp_path))
    await upgraded.start()
    assert upgraded.added_columns == ["judge_model"]
    [old] = await upgraded.pending_samples(10)
    await upgraded.save_label(replace(old, status="labeled", judge_score=2, judge_score_strong=5,
                                      needs_strong=True, labeled_ts=2000.0,
                                      judge_model="j"))  # fmt: skip
    [labeled] = await upgraded.labeled_samples()
    await upgraded.stop()
    assert (old.judge_model, labeled.judge_model) == (None, "j")


async def test_failed_sample_write_logs_no_payload_and_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log = RequestLog(_url(tmp_path))
    await log.start()
    log.submit_sample(_sample(0, request_json="SECRET-PROMPT", routed_answer="SECRET-ANSWER"))
    await log.stop()
    await log.start()
    log.submit_sample(_sample(0, request_json="SECRET-PROMPT", routed_answer="SECRET-ANSWER"))
    await log.stop()  # same id again: the insert fails
    assert "failed to write 1 shadow samples" in caplog.text
    assert "SECRET" not in caplog.text  # prompts and answers never reach the logs
    assert log.dropped == 1  # visible as tierwise_telemetry_dropped_total
