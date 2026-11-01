import pytest

from tierwise.api.errors import RateLimited
from tierwise.budgets import RateLimiter, TokenBucket
from tierwise.store import KeyRow


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _key(rpm: int | None = None, budget: float | None = None, key_id: str = "k1") -> KeyRow:
    return KeyRow(key_id, "team", "h", "tw-x", budget, rpm, 0.0, None)


def test_full_bucket_allows_rpm_requests_then_blocks() -> None:
    bucket = TokenBucket(3, clock=Clock())
    assert [bucket.take() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert bucket.take() == pytest.approx(20.0)  # one token per 20 s at 3 rpm


def test_refills_at_rpm_per_minute() -> None:
    clock = Clock()
    bucket = TokenBucket(60, clock=clock)
    for _ in range(60):
        assert bucket.take() == 0.0
    assert bucket.take() > 0
    clock.now += 1.0
    assert bucket.take() == 0.0
    assert bucket.take() > 0


def test_never_exceeds_capacity() -> None:
    clock = Clock()
    bucket = TokenBucket(2, clock=clock)
    clock.now += 600
    assert [bucket.take() == 0.0 for _ in range(3)] == [True, True, False]


def test_blocked_take_does_not_consume() -> None:
    clock = Clock()
    bucket = TokenBucket(1, clock=clock)
    assert bucket.take() == 0.0
    assert bucket.take() == pytest.approx(60.0)
    clock.now += 30
    assert bucket.take() == pytest.approx(30.0)  # waiting doesn't push the refill further out


def test_unlimited_key_never_limited() -> None:
    limiter = RateLimiter(clock=Clock())
    for _ in range(1000):
        limiter.check(_key(rpm=None))


def test_limiter_raises_with_whole_second_retry_after() -> None:
    limiter = RateLimiter(clock=Clock())
    limiter.check(_key(rpm=1))
    with pytest.raises(RateLimited) as info:
        limiter.check(_key(rpm=1))
    assert info.value.retry_after == 60 and info.value.status == 429
