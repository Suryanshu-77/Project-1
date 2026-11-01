"""Walk the selector's candidates: retry each, skip open circuits, fail over on transient errors."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial

from tierwise.api.errors import AllCircuitsOpen
from tierwise.options import Option
from tierwise.providers.base import FatalError, ProviderError
from tierwise.reliability.breaker import Breakers
from tierwise.reliability.retry import RetryPolicy, with_retries

MAX_MODELS = 3  # distinct candidates attempted per request, at most


@dataclass(frozen=True)
class Served[T]:
    value: T
    option: Option
    failovers: int  # candidates abandoned before this one served


def next_candidates(remaining: list[Option], failed: Option, error: ProviderError) -> list[Option]:
    """Which of the remaining candidates are still worth trying after `failed` raised `error`.

    `remaining` is in the selector's order: cheapest clearing the quality bar first, then the
    rest. The result is tried in order, so it may also reorder.
    """
    error_name = type(error).__name__
    is_provider_wide = error_name in ("ConnectionFailure", "AuthError")

    # If the provider is completely down or our auth is rejected, drop all its models
    if is_provider_wide:
        return [opt for opt in remaining if opt.spec.provider != failed.spec.provider]

    # Otherwise, it's a model-level failure (429, 5xx, timeout).
    # Drop the exact same model (e.g. at different effort levels).
    # Reorder to push same-provider siblings to the back—trading a cheaper sibling for a safer bet.
    other_providers = []
    same_provider_siblings = []

    for opt in remaining:
        if opt.spec.id == failed.spec.id:
            continue  # Drop same model ID entirely

        if opt.spec.provider == failed.spec.provider:
            same_provider_siblings.append(opt)
        else:
            other_providers.append(opt)

    return other_providers + same_provider_siblings


async def serve[T](
    candidates: list[Option],
    attempt: Callable[[Option], Awaitable[T]],
    breakers: Breakers,
    policy: RetryPolicy,
    *,
    clear_count: int | None = None,
    record_success: bool = True,
) -> Served[T]:
    """Try candidates in order. With `clear_count`, the first that many clear the quality bar,
    and failover reorders within each group, so a below-bar model never jumps the queue.

    `record_success=False` leaves the success verdict to the caller (a stream is only a
    success once it has finished, not when it starts).
    """
    split = len(candidates) if clear_count is None else clear_count
    groups = [list(candidates[:split]), list(candidates[split:])]
    skipped: list[str] = []
    attempted = 0
    last_error: ProviderError | None = None
    while any(groups) and attempted < MAX_MODELS:
        option = next(group for group in groups if group).pop(0)
        provider, model_id = option.spec.provider, option.spec.id
        if not breakers.acquire(provider, model_id):
            skipped.append(f"{provider}/{model_id}")
            continue
        attempted += 1
        try:
            value = await with_retries(partial(attempt, option), policy)
        except ProviderError as exc:
            breakers.record(provider, model_id, exc)
            if isinstance(exc, FatalError):
                raise  # the request is bad; another model won't fix it
            last_error = exc
            groups = [next_candidates(group, option, exc) for group in groups]
            continue
        except BaseException:
            # cancelled or a bug: no verdict on the upstream, so give back any probe we hold
            breakers.provider(provider).release()
            breakers.model(provider, model_id).release()
            raise
        if record_success:
            breakers.record(provider, model_id, None)
        return Served(value, option, attempted - 1)
    if last_error is not None:
        raise last_error
    raise AllCircuitsOpen(skipped)
