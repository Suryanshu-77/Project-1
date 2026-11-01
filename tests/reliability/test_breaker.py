from tierwise.providers.base import (
    AuthError,
    ConnectionFailure,
    FatalError,
    RetryableError,
)
from tierwise.reliability.breaker import Breakers, CircuitBreaker


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _breaker(clock: Clock) -> CircuitBreaker:
    return CircuitBreaker(failures=5, window_s=60.0, cooldown_s=30.0, clock=clock)


def _trip(breaker: CircuitBreaker) -> None:
    for _ in range(5):
        breaker.record_failure()


def test_trips_after_five_consecutive_failures() -> None:
    breaker = _breaker(Clock())
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.try_acquire() is False


def test_success_resets_the_streak() -> None:
    breaker = _breaker(Clock())
    for _ in range(4):
        breaker.record_failure()
    breaker.record_success()
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == "closed"


def test_failures_spread_beyond_window_do_not_trip() -> None:
    clock = Clock()
    breaker = _breaker(clock)
    for _ in range(5):
        breaker.record_failure()
        clock.now += 20
        if breaker.state == "open":
            break
    assert breaker.state == "closed"


def test_half_open_after_cooldown_allows_one_probe() -> None:
    clock = Clock()
    breaker = _breaker(clock)
    _trip(breaker)
    clock.now = 29
    assert breaker.try_acquire() is False
    clock.now = 30
    assert breaker.state == "half_open"
    assert breaker.try_acquire() is True
    assert breaker.try_acquire() is False  # only one probe in flight


def test_probe_success_closes_and_failure_reopens() -> None:
    clock = Clock()
    breaker = _breaker(clock)
    _trip(breaker)
    clock.now = 30
    assert breaker.try_acquire()
    breaker.record_success()
    assert breaker.state == "closed"
    _trip(breaker)
    clock.now = 70
    assert breaker.try_acquire()
    breaker.record_failure()
    assert breaker.state == "open"
    clock.now = 99
    assert breaker.try_acquire() is False  # a fresh 30 s cooldown from the failed probe


def _breakers(clock: Clock) -> Breakers:
    return Breakers(failures=5, window_s=60.0, cooldown_s=30.0, clock=clock)


def _fail(breakers: Breakers, provider: str, model: str, error: Exception, times: int = 5) -> None:
    for _ in range(times):
        breakers.record(provider, model, error)  # type: ignore[arg-type]


def test_model_breaker_does_not_block_sibling() -> None:
    breakers = _breakers(Clock())
    _fail(breakers, "anthropic", "opus", RetryableError("anthropic", 429, "slow"))
    assert breakers.available("anthropic", "opus") is False
    assert breakers.available("anthropic", "haiku") is True


def test_two_open_models_make_provider_unavailable() -> None:
    breakers = _breakers(Clock())
    _fail(breakers, "anthropic", "opus", RetryableError("anthropic", 503, "down"))
    _fail(breakers, "anthropic", "sonnet", RetryableError("anthropic", 503, "down"))
    assert breakers.available("anthropic", "haiku") is False
    assert breakers.available("openai", "mini") is True


def test_connection_failure_trips_provider_level() -> None:
    breakers = _breakers(Clock())
    _fail(breakers, "anthropic", "opus", ConnectionFailure("anthropic", None, "ConnectError"))
    assert breakers.available("anthropic", "haiku") is False


def test_auth_error_trips_provider_level() -> None:
    breakers = _breakers(Clock())
    _fail(breakers, "anthropic", "opus", AuthError("anthropic", 401, "key"))
    assert breakers.available("anthropic", "sonnet") is False


def test_fatal_counts_as_success() -> None:
    breakers = _breakers(Clock())
    _fail(breakers, "anthropic", "opus", RetryableError("anthropic", 503, "down"), times=4)
    breakers.record("anthropic", "opus", FatalError("anthropic", 400, "bad request"))
    _fail(breakers, "anthropic", "opus", RetryableError("anthropic", 503, "down"), times=4)
    assert breakers.available("anthropic", "opus") is True


def test_acquire_needs_both_levels() -> None:
    clock = Clock()
    breakers = _breakers(clock)
    _fail(breakers, "anthropic", "opus", ConnectionFailure("anthropic", None, "ConnectError"))
    assert breakers.acquire("anthropic", "haiku") is False
    clock.now = 30
    assert breakers.acquire("anthropic", "haiku") is True  # the provider's single probe
    assert breakers.acquire("anthropic", "sonnet") is False
