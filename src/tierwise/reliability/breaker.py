"""Circuit breakers: stop calling a model (or a whole provider) that keeps failing.

closed --N consecutive failures within the window--> open --cooldown--> half_open
half_open --the single probe succeeds--> closed;  --the probe fails--> open (cooldown restarts)
"""

import time
from collections.abc import Callable
from typing import Literal

from tierwise.providers.base import (
    AuthError,
    ConnectionFailure,
    ProviderError,
    RetryableError,
)

State = Literal["closed", "open", "half_open"]
OPEN_MODELS_FOR_PROVIDER = 2  # this many open model breakers mark the whole provider unavailable


class CircuitBreaker:
    def __init__(
        self,
        failures: int = 5,
        window_s: float = 60.0,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failures = failures
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._streak = 0
        self._streak_start = 0.0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> State:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at < self.cooldown_s:
            return "open"
        return "half_open"

    def try_acquire(self) -> bool:
        """May a call go through? In half-open, only one caller gets the probe."""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probing:
            self._probing = True
            return True
        return False

    def release(self) -> None:
        """Give back a probe that was claimed but never used."""
        self._probing = False

    def record_success(self) -> None:
        self._streak = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        now = self._clock()
        state = self.state
        if state == "half_open":  # the probe failed: back to open for a fresh cooldown
            self._opened_at, self._probing, self._streak = now, False, 0
            return
        if state == "open":
            return  # stragglers that started before the trip don't extend it
        if self._streak == 0 or now - self._streak_start > self.window_s:
            self._streak, self._streak_start = 1, now
        else:
            self._streak += 1
        if self._streak >= self.failures:
            self._opened_at, self._streak = now, 0


class Breakers:
    """Breakers per (provider, model) and per provider, created on first use."""

    def __init__(
        self,
        failures: int = 5,
        window_s: float = 60.0,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._new = lambda: CircuitBreaker(failures, window_s, cooldown_s, clock)
        self._models: dict[tuple[str, str], CircuitBreaker] = {}
        self._providers: dict[str, CircuitBreaker] = {}

    def model(self, provider: str, model_id: str) -> CircuitBreaker:
        key = (provider, model_id)
        if key not in self._models:
            self._models[key] = self._new()
        return self._models[key]

    def provider(self, name: str) -> CircuitBreaker:
        if name not in self._providers:
            self._providers[name] = self._new()
        return self._providers[name]

    def _open_models(self, provider: str) -> int:
        return sum(
            1 for (p, _), b in self._models.items() if p == provider and b.state == "open"
        )

    def available(self, provider: str, model_id: str) -> bool:
        """Read-only: could this model be tried now? Claims nothing."""
        return (
            self.provider(provider).state != "open"
            and self._open_models(provider) < OPEN_MODELS_FOR_PROVIDER
            and self.model(provider, model_id).state != "open"
        )

    def acquire(self, provider: str, model_id: str) -> bool:
        """Claim the right to call this model now; both levels must agree."""
        if not self.available(provider, model_id):
            return False
        provider_breaker = self.provider(provider)
        if not provider_breaker.try_acquire():
            return False
        if not self.model(provider, model_id).try_acquire():
            provider_breaker.release()
            return False
        return True

    def record(self, provider: str, model_id: str, error: ProviderError | None) -> None:
        """Feed one call's outcome to the right level. Bad requests (FatalError) mean the
        upstream is healthy, so they count as successes."""
        provider_breaker, model_breaker = self.provider(provider), self.model(provider, model_id)
        if isinstance(error, ConnectionFailure | AuthError):
            provider_breaker.record_failure()
            model_breaker.release()
        elif isinstance(error, RetryableError):
            provider_breaker.record_success()  # it answered, so the provider is reachable
            model_breaker.record_failure()
        else:
            provider_breaker.record_success()
            model_breaker.record_success()
