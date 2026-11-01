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


# --- budgets -------------------------------------------------------------------------------
from datetime import UTC, datetime  # noqa: E402

from tierwise.api.errors import BudgetExceeded  # noqa: E402
from tierwise.budgets import Budgets, month_of  # noqa: E402


class FakeLog:
    def __init__(self) -> None:
        self.debits: list[tuple[str, str, float]] = []

    def submit_debit(self, key_id: str, month: str, cost_usd: float) -> None:
        self.debits.append((key_id, month, cost_usd))


def _at(iso: str) -> float:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp()


def _budgets(clock: Clock | None = None) -> tuple[Budgets, FakeLog]:
    log = FakeLog()
    clock = clock or Clock()
    clock.now = clock.now if clock.now != 1000.0 else _at("2026-11-15T12:00:00")
    return Budgets(log, clock=clock), log  # type: ignore[arg-type]


def test_month_of_is_utc() -> None:
    assert month_of(_at("2026-11-30T23:59:59")) == "2026-11"
    assert month_of(_at("2026-12-01T00:00:00")) == "2026-12"


def test_reserve_within_budget_then_settle_debits_actual() -> None:
    budgets, log = _budgets()
    key = _key(budget=1.0)
    reservation = budgets.reserve(key, 0.40)
    budgets.settle(reservation, key, 0.25)
    assert budgets.spent("k1") == pytest.approx(0.25)
    assert log.debits == [("k1", "2026-11", 0.25)]
    assert budgets.reserve(key, 0.75) is not None  # the reservation was released


def test_reserve_over_budget_raises() -> None:
    budgets, _ = _budgets()
    budgets.load("2026-11", {"k1": 0.90})
    with pytest.raises(BudgetExceeded, match=r"0\.90"):
        budgets.reserve(_key(budget=1.0), 0.20)


def test_reservations_count_against_budget() -> None:
    budgets, _ = _budgets()
    key = _key(budget=1.0)
    budgets.reserve(key, 0.40)
    budgets.reserve(key, 0.40)
    with pytest.raises(BudgetExceeded):
        budgets.reserve(key, 0.40)


def test_settle_none_releases_without_debit() -> None:
    budgets, log = _budgets()
    key = _key(budget=1.0)
    budgets.settle(budgets.reserve(key, 0.90), key, None)
    assert log.debits == [] and budgets.spent("k1") == 0.0
    assert budgets.reserve(key, 0.90) is not None


def test_month_rollover_starts_fresh() -> None:
    clock = Clock()
    clock.now = _at("2026-11-30T23:59:00")
    budgets, log = _budgets(clock)
    key = _key(budget=1.0)
    budgets.settle(budgets.reserve(key, 0.9), key, 0.9)
    clock.now = _at("2026-12-01T00:01:00")
    assert budgets.spent("k1") == 0.0
    assert budgets.reserve(key, 0.9) is not None
    assert log.debits == [("k1", "2026-11", 0.9)]


def test_unlimited_key_is_debited_but_never_blocked() -> None:
    budgets, log = _budgets()
    key = _key(budget=None)
    assert budgets.reserve(key, 1e9) is None
    budgets.settle(None, key, 0.5)
    assert log.debits == [("k1", "2026-11", 0.5)]
