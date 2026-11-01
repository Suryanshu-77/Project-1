"""Retry a provider call on transient errors, with exponential backoff and full jitter."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from tierwise.providers.base import RetryableError


@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 2
    base_s: float = 0.25
    max_s: float = 2.0


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Run ``call``, retrying only RetryableError; fatal and auth errors surface at once."""
    for attempt in range(policy.retries + 1):
        try:
            return await call()
        except RetryableError:
            if attempt == policy.retries:
                raise
            # full jitter spreads retries out so a recovering upstream isn't hit in lockstep
            await sleep(rand() * min(policy.max_s, policy.base_s * 2**attempt))
    raise AssertionError("unreachable")  # the loop always returns or raises
