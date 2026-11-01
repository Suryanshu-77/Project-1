"""Gateway API keys: ``tw-`` + random secret, stored only as a SHA-256 hash.

The request path checks keys against an in-memory set, refreshed from the database in the
background, so an invalid key can never make a request touch the database.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets

from tierwise.store import KeyRow, RequestLog

logger = logging.getLogger(__name__)

KEY_PREFIX = "tw-"


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore:
    """Active keys by hash."""

    def __init__(self) -> None:
        self._by_hash: dict[str, KeyRow] = {}

    def replace(self, rows: list[KeyRow]) -> None:
        self._by_hash = {row.key_hash: row for row in rows if row.revoked_ts is None}

    def find(self, key: str) -> KeyRow | None:
        digest = hash_key(key)
        row = self._by_hash.get(digest)
        if row is not None and hmac.compare_digest(row.key_hash, digest):
            return row
        return None

    async def refresh_forever(self, log: RequestLog, interval_s: float) -> None:
        """Pick up keys created or revoked by the CLI; on failure keep the last good set."""
        while True:
            await asyncio.sleep(interval_s)
            try:
                self.replace(await log.list_keys())
            except Exception:
                logger.exception("could not refresh API keys; keeping the previous set")
