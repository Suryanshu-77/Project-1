from collections.abc import Awaitable, Callable

import pytest

from tierwise.providers.base import AuthError, FatalError, ProviderError, RetryableError
from tierwise.reliability.retry import RetryPolicy, with_retries


def _flaky(errors: list[ProviderError]) -> tuple[list[int], Callable[[], Awaitable[str]]]:
    calls: list[int] = []

    async def call() -> str:
        calls.append(1)
        if errors:
            raise errors.pop(0)
        return "ok"

    return calls, call


def _sleeps() -> tuple[list[float], Callable[[float], Awaitable[None]]]:
    recorded: list[float] = []

    async def sleep(seconds: float) -> None:
        recorded.append(seconds)

    return recorded, sleep


async def test_succeeds_after_two_retryable_failures() -> None:
    calls, call = _flaky([RetryableError("p", 503, "x"), RetryableError("p", 429, "x")])
    slept, sleep = _sleeps()
    assert await with_retries(call, RetryPolicy(), sleep=sleep) == "ok"
    assert (len(calls), len(slept)) == (3, 2)


async def test_gives_up_after_policy_retries() -> None:
    calls, call = _flaky([RetryableError("p", 503, str(i)) for i in range(10)])
    _, sleep = _sleeps()
    with pytest.raises(RetryableError, match="2"):
        await with_retries(call, RetryPolicy(retries=2), sleep=sleep)
    assert len(calls) == 3


@pytest.mark.parametrize("error", [FatalError("p", 400, "bad"), AuthError("p", 401, "key")])
async def test_fatal_and_auth_are_not_retried(error: ProviderError) -> None:
    calls, call = _flaky([error])
    _, sleep = _sleeps()
    with pytest.raises(type(error)):
        await with_retries(call, RetryPolicy(), sleep=sleep)
    assert len(calls) == 1


async def test_backoff_is_full_jitter_capped() -> None:
    _, call = _flaky([RetryableError("p", 503, "x") for _ in range(10)])
    slept, sleep = _sleeps()
    with pytest.raises(RetryableError):
        await with_retries(call, RetryPolicy(retries=5), sleep=sleep, rand=lambda: 1.0)
    assert slept == [0.25, 0.5, 1.0, 2.0, 2.0]
