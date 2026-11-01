"""Request log: handlers submit records to a bounded queue; one background task writes them.

Keeping disk I/O off the request path means a slow or locked database never delays a
response, and a single writer means SQLite never sees competing writers.
"""

import asyncio
import logging
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
)
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)


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
        while not (self._closing and self._queue.empty()):
            batch = await self._next_batch()
            if batch:
                await self._write(batch)

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
