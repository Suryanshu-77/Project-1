"""Per-key limits: a token bucket for requests per minute, and monthly USD budgets.

Both live in memory, so the request path never reads the database.
"""

import math
import time
from collections.abc import Callable

from tierwise.api.errors import RateLimited
from tierwise.store import KeyRow


class TokenBucket:
    """Holds up to ``rpm`` tokens and refills at ``rpm / 60`` per second; a request costs one."""

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rpm = rpm
        self._clock = clock
        self._tokens = float(rpm)  # starts full: a fresh key can burst up to its limit
        self._updated = clock()

    def take(self) -> float:
        """Take one token if there is one and return 0.0; otherwise take nothing and return
        the seconds until a token will be available."""
        now = self._clock()
        elapsed = now - self._updated
        rate = self.rpm / 60.0  # tokens per second

        # Refill first, capped at maximum capacity (rpm)
        self._tokens = min(float(self.rpm), self._tokens + elapsed * rate)
        self._updated = now

        # Take or Wait
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0
        else:
            # We need 1 token. We have self._tokens.
            # Time required to generate the missing amount = missing_tokens / rate
            return (1.0 - self._tokens) / rate


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, key: KeyRow) -> None:
        if key.rpm is None:
            return
        bucket = self._buckets.get(key.id)
        if bucket is None or bucket.rpm != key.rpm:  # new key, or its limit was changed
            bucket = self._buckets[key.id] = TokenBucket(key.rpm, self._clock)
        wait = bucket.take()
        if wait > 0:
            raise RateLimited(retry_after=max(1, math.ceil(wait)))
