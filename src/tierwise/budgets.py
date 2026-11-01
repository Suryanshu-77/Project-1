"""Per-key limits: a token bucket for requests per minute, and monthly USD budgets.

Both live in memory, so the request path never reads the database.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from tierwise.api.errors import BudgetExceeded, RateLimited
from tierwise.registry import ModelSpec
from tierwise.schema import ChatRequest
from tierwise.selector import estimated_cost
from tierwise.store import KeyRow, RequestLog
from tierwise.tokens import estimate_output, estimate_tokens


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


def month_of(ts: float) -> str:
    """The budget period: the UTC calendar month, "YYYY-MM"."""
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m")


def estimate_for(spec: ModelSpec, req: ChatRequest, thinking_tokens: int = 0) -> float:
    """What the selector expects this request to cost on ``spec`` (no cache discount)."""
    base = estimated_cost(spec, estimate_tokens(req), estimate_output(req), 0)
    return base + thinking_tokens * spec.price_out / 1e6


@dataclass
class Reservation:
    key_id: str
    month: str
    amount: float


class Budgets:
    """Monthly spend per key, in memory. A request reserves its estimate before calling and
    settles the actual cost afterwards, so concurrent requests can't jointly overshoot."""

    def __init__(self, log: RequestLog, clock: Callable[[], float] = time.time) -> None:
        self._log = log
        self._clock = clock
        self._spent: dict[tuple[str, str], float] = {}
        self._reserved: dict[tuple[str, str], float] = {}

    def load(self, month: str, spend: dict[str, float]) -> None:
        for key_id, spent in spend.items():
            self._spent[(key_id, month)] = spent

    def spent(self, key_id: str, month: str | None = None) -> float:
        return self._spent.get((key_id, month or month_of(self._clock())), 0.0)

    def reserved(self, key_id: str, month: str | None = None) -> float:
        return self._reserved.get((key_id, month or month_of(self._clock())), 0.0)

    def reserve(self, key: KeyRow, estimate: float) -> Reservation | None:
        """None for a key without a budget; raises BudgetExceeded when it doesn't fit."""
        budget = key.monthly_budget_usd
        if budget is None:
            return None
        month = month_of(self._clock())
        spent, reserved = self.spent(key.id, month), self.reserved(key.id, month)
        if spent + reserved + estimate > budget:
            raise BudgetExceeded(
                f"Monthly budget exceeded for this API key: spent ${spent:.2f}, "
                f"${reserved:.2f} in flight, ${estimate:.2f} estimated, budget ${budget:.2f}."
            )
        self._reserved[(key.id, month)] = reserved + estimate
        return Reservation(key.id, month, estimate)

    def settle(
        self, reservation: Reservation | None, key: KeyRow | None, actual: float | None
    ) -> None:
        """Release the reservation and charge what the request really cost (None: nothing)."""
        if reservation is not None:
            slot = (reservation.key_id, reservation.month)
            self._reserved[slot] = max(0.0, self._reserved.get(slot, 0.0) - reservation.amount)
        if key is None or actual is None:
            return
        month = month_of(self._clock())
        self._spent[(key.id, month)] = self.spent(key.id, month) + actual
        self._log.submit_debit(key.id, month, actual)
