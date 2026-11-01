"""Request log: handlers submit records to a bounded queue; one background task writes them.

Keeping disk I/O off the request path means a slow or locked database never delays a
response, and a single writer means SQLite never sees competing writers.

Budget debits take a separate path: they are summed in memory per (key, month) and upserted
by the same writer. Unlike telemetry rows they are never dropped; a failed write keeps them
for the next flush, and shutdown waits until they are written.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    event,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

SHUTDOWN_FLUSH_ATTEMPTS = 3  # then unwritten debits are logged for a manual replay


@dataclass(frozen=True)
class RequestRecord:
    id: str
    ts: float
    model: str
    provider: str
    status: int
    stream: bool
    latency_ms: float
    tokens_in: int | None
    tokens_cached: int | None
    tokens_out: int | None
    cost_usd: float | None
    baseline_cost_usd: float | None
    source: str = "override"
    mode: str | None = None
    key_id: str | None = None
    difficulty: float | None = None
    confidence: float | None = None
    prefix_hash: str | None = None
    cache_hit: bool = False
    failover_count: int = 0
    error_code: str | None = None
    effort: str | None = None
    tokens_reasoning: int | None = None


metadata = MetaData()
requests_table = Table(
    "requests",
    metadata,
    Column("id", String, primary_key=True),
    Column("ts", Float, nullable=False, index=True),
    Column("model", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("status", Integer, nullable=False),
    Column("stream", Boolean, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("tokens_in", Integer),
    Column("tokens_cached", Integer),
    Column("tokens_out", Integer),
    Column("cost_usd", Float),
    Column("baseline_cost_usd", Float),
    Column("source", String, nullable=False),
    Column("mode", String),
    Column("key_id", String),
    Column("difficulty", Float),
    Column("confidence", Float),
    Column("prefix_hash", String),
    Column("cache_hit", Boolean, nullable=False),
    Column("failover_count", Integer, nullable=False),
    Column("error_code", String),
    Column("effort", String),
    Column("tokens_reasoning", Integer),
)


api_keys_table = Table(
    "api_keys",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("key_hash", String, nullable=False, unique=True),
    Column("prefix", String, nullable=False),
    Column("monthly_budget_usd", Float),
    Column("rpm", Integer),
    Column("created_ts", Float, nullable=False),
    Column("revoked_ts", Float),
)
usage_monthly_table = Table(
    "usage_monthly",
    metadata,
    Column("key_id", String, primary_key=True),
    Column("month", String, primary_key=True),  # "YYYY-MM", UTC
    Column("spent_usd", Float, nullable=False),
    Column("requests", Integer, nullable=False),
)


@dataclass(frozen=True)
class KeyRow:
    id: str
    name: str
    key_hash: str
    prefix: str
    monthly_budget_usd: float | None
    rpm: int | None
    created_ts: float
    revoked_ts: float | None


def _sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")  # readers never block the writer
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def _add_missing_columns(conn: Connection) -> list[str]:
    """Bring an older ``requests`` table up to date by adding new nullable columns.

    ``create_all`` never alters an existing table, so a database written by an earlier
    version would otherwise reject every insert. Existing rows keep NULL in the new columns.
    """
    present = {column["name"] for column in inspect(conn).get_columns("requests")}
    added = []
    for column in requests_table.columns:
        if column.name in present:
            continue
        if not column.nullable:
            raise RuntimeError(f"cannot add required column '{column.name}' to existing table")
        column_type = column.type.compile(dialect=conn.dialect)
        conn.execute(text(f"ALTER TABLE requests ADD COLUMN {column.name} {column_type}"))
        added.append(column.name)
    return added


class RequestLog:
    def __init__(
        self,
        db_url: str,
        max_queue: int = 10_000,
        batch_size: int = 200,
        flush_interval: float = 0.25,
    ) -> None:
        self._db_url = db_url
        self._queue: asyncio.Queue[RequestRecord] = asyncio.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._engine: AsyncEngine | None = None
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self.dropped = 0
        self._debits: dict[tuple[str, str], tuple[float, int]] = {}
        self._shutdown_failures = 0
        self.added_columns: list[str] = []  # columns the last start() had to add

    async def start(self) -> None:
        self._engine = create_async_engine(self._db_url)
        if self._engine.dialect.name == "sqlite":
            event.listen(self._engine.sync_engine, "connect", _sqlite_pragmas)
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            self.added_columns = await conn.run_sync(_add_missing_columns)
        self._closing = False
        self._task = asyncio.create_task(self._run(), name="tierwise-request-log")

    def submit(self, record: RequestRecord) -> None:
        """Queue a record without blocking; under overload the record is dropped and counted."""
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1

    def submit_debit(self, key_id: str, month: str, cost_usd: float) -> None:
        """Charge a key; summed in memory until the writer upserts it. Never dropped."""
        spent, count = self._debits.get((key_id, month), (0.0, 0))
        self._debits[(key_id, month)] = (spent + cost_usd, count + 1)

    @property
    def pending_debits(self) -> dict[tuple[str, str], tuple[float, int]]:
        return dict(self._debits)

    async def create_key(
        self, name: str, key_hash: str, prefix: str, budget: float | None, rpm: int | None
    ) -> KeyRow:
        row = KeyRow(uuid.uuid4().hex[:12], name, key_hash, prefix, budget, rpm, time.time(), None)
        async with self._require_engine().begin() as conn:
            await conn.execute(insert(api_keys_table), [asdict(row)])
        return row

    async def list_keys(self) -> list[KeyRow]:
        async with self._require_engine().connect() as conn:
            rows = await conn.execute(select(api_keys_table).order_by(api_keys_table.c.created_ts))
            return [KeyRow(**row._mapping) for row in rows]

    async def revoke_key(self, key_id: str) -> bool:
        """False when the key doesn't exist or is already revoked."""
        table = api_keys_table
        statement = (
            update(table)
            .where(table.c.id == key_id, table.c.revoked_ts.is_(None))
            .values(revoked_ts=time.time())
        )
        async with self._require_engine().begin() as conn:
            result = await conn.execute(statement)
        return bool(result.rowcount)

    async def load_month_spend(self, month: str) -> dict[str, float]:
        table = usage_monthly_table
        async with self._require_engine().connect() as conn:
            rows = await conn.execute(
                select(table.c.key_id, table.c.spent_usd).where(table.c.month == month)
            )
            return {key_id: spent for key_id, spent in rows}

    async def fetch_usage(self) -> list[tuple[str, str, float, int]]:
        table = usage_monthly_table
        engine = self._require_engine()
        async with engine.connect() as conn:
            rows = await conn.execute(
                select(table.c.key_id, table.c.month, table.c.spent_usd, table.c.requests)
            )
            usage = [(k, m, s, n) for k, m, s, n in rows]
        if self._task is None:
            await engine.dispose()
        return usage

    def _require_engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("RequestLog.start() has not been called")
        return self._engine

    async def stop(self) -> None:
        """Write everything still queued, then release database connections."""
        self._closing = True
        if self._task is not None:
            await self._task
            self._task = None
        if self._engine is not None:
            await self._engine.dispose()

    async def fetch_all(self) -> list[RequestRecord]:
        if self._engine is None:
            return []
        async with self._engine.connect() as conn:
            rows = await conn.execute(select(requests_table).order_by(requests_table.c.ts))
            records = [RequestRecord(**row._mapping) for row in rows]
        if self._task is None:  # stopped: don't leave a pooled connection behind
            await self._engine.dispose()
        return records

    async def _run(self) -> None:
        while not (self._closing and self._queue.empty() and not self._debits):
            batch = await self._next_batch()
            if batch:
                await self._write(batch)
            if self._debits:
                await self._flush_debits()

    async def _flush_debits(self) -> None:
        snapshot = dict(self._debits)
        rows = [
            {"key_id": key_id, "month": month, "spent_usd": spent, "requests": count}
            for (key_id, month), (spent, count) in snapshot.items()
        ]
        try:
            async with self._require_engine().begin() as conn:
                await self._upsert_debits(conn, rows)
        except Exception:
            logger.exception("failed to write %d budget debits; retrying", len(rows))
            if self._closing:
                self._shutdown_failures += 1
                if self._shutdown_failures >= SHUTDOWN_FLUSH_ATTEMPTS:
                    self._abandon_debits()
                    return
                await asyncio.sleep(self._flush_interval)  # don't spin while shutting down
            return
        for key, (spent, count) in snapshot.items():  # remove only what was written
            left_spent, left_count = self._debits[key]
            if left_count == count:
                del self._debits[key]
            else:
                self._debits[key] = (left_spent - spent, left_count - count)

    def _abandon_debits(self) -> None:
        """Shutting down and the database won't take them: log every debit so it can be
        replayed by hand, rather than hang the process until it is killed."""
        lines = [
            f"key_id={key_id} month={month} usd={spent:.6f} requests={count}"
            for (key_id, month), (spent, count) in self._debits.items()
        ]
        logger.error("unwritten budget debits (replay manually):\n%s", "\n".join(lines))
        self._debits.clear()

    async def _upsert_debits(self, conn: AsyncConnection, rows: list[dict[str, Any]]) -> None:
        dialect = postgresql if conn.dialect.name == "postgresql" else sqlite
        statement = dialect.insert(usage_monthly_table)
        statement = statement.on_conflict_do_update(
            index_elements=["key_id", "month"],
            set_={
                "spent_usd": usage_monthly_table.c.spent_usd + statement.excluded.spent_usd,
                "requests": usage_monthly_table.c.requests + statement.excluded.requests,
            },
        )
        await conn.execute(statement, rows)

    async def _next_batch(self) -> list[RequestRecord]:
        """Collect up to batch_size records, or whatever arrived within flush_interval."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._flush_interval
        batch: list[RequestRecord] = []
        while len(batch) < self._batch_size:
            if self._closing and self._queue.empty():
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def _write(self, batch: list[RequestRecord]) -> None:
        assert self._engine is not None
        try:
            async with self._engine.begin() as conn:
                await conn.execute(insert(requests_table), [asdict(r) for r in batch])
        except Exception:
            logger.exception("failed to write %d request records", len(batch))
