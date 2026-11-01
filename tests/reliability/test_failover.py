from typing import Any

import pytest

from tierwise.api.errors import AllCircuitsOpen
from tierwise.options import Option
from tierwise.providers.base import (
    AuthError,
    ConnectionFailure,
    FatalError,
    ProviderError,
    RetryableError,
)
from tierwise.registry import ModelSpec
from tierwise.reliability.breaker import Breakers
from tierwise.reliability.failover import MAX_MODELS, next_candidates, serve
from tierwise.reliability.retry import RetryPolicy

NO_RETRY = RetryPolicy(retries=0)


def _opt(provider: str, model: str, effort: Any = None) -> Option:
    spec = ModelSpec.model_validate(
        {"id": model, "provider": provider, "price_in": 1.0, "price_out": 1.0,
         "context": 1000, "caps": [], "quality": 0.5, "tier": "basic"}  # fmt: skip
    )
    return Option(spec=spec, effort=effort, quality=0.5, thinking_tokens=0, send=False)


def _attempt(outcomes: dict[str, ProviderError | None]) -> tuple[list[str], Any]:
    calls: list[str] = []

    async def attempt(option: Option) -> str:
        calls.append(option.spec.id)
        error = outcomes.get(option.spec.id)
        if error is not None:
            raise error
        return option.spec.id

    return calls, attempt


A, B, C, D = (_opt("openai", "a"), _opt("anthropic", "b"), _opt("gemini", "c"),
              _opt("ollama", "d"))  # fmt: skip


async def test_first_candidate_serves() -> None:
    calls, attempt = _attempt({})
    served = await serve([A, B], attempt, Breakers(), NO_RETRY)
    assert (served.value, served.failovers, calls) == ("a", 0, ["a"])


async def test_retryable_fails_over_to_next() -> None:
    calls, attempt = _attempt({"a": RetryableError("openai", 429, "slow")})
    served = await serve([A, B], attempt, Breakers(), NO_RETRY)
    assert (served.value, served.option, served.failovers) == ("b", B, 1)


async def test_fatal_does_not_fail_over() -> None:
    calls, attempt = _attempt({"a": FatalError("openai", 400, "bad")})
    with pytest.raises(FatalError):
        await serve([A, B], attempt, Breakers(), NO_RETRY)
    assert calls == ["a"]


async def test_open_circuit_is_skipped_without_calling() -> None:
    breakers = Breakers()
    for _ in range(5):
        breakers.record("openai", "a", RetryableError("openai", 503, "down"))
    calls, attempt = _attempt({})
    served = await serve([A, B], attempt, breakers, NO_RETRY)
    assert calls == ["b"] and served.failovers == 0


async def test_all_open_raises_all_circuits_open() -> None:
    breakers = Breakers()
    for option in (A, B):
        for _ in range(5):
            breakers.record(option.spec.provider, option.spec.id, RetryableError("x", 503, ""))
    _, attempt = _attempt({})
    with pytest.raises(AllCircuitsOpen, match="openai/a"):
        await serve([A, B], attempt, breakers, NO_RETRY)


async def test_stops_after_three_models() -> None:
    down = RetryableError("x", 503, "down")
    calls, attempt = _attempt({"a": down, "b": down, "c": down})
    with pytest.raises(RetryableError):
        await serve([A, B, C, D], attempt, Breakers(), NO_RETRY)
    assert calls == ["a", "b", "c"] and MAX_MODELS == 3


# --- the hand-off's contract ---------------------------------------------------------------

OPUS_LOW, OPUS_HIGH = _opt("anthropic", "opus", "low"), _opt("anthropic", "opus", "high")
HAIKU, SONNET = _opt("anthropic", "haiku"), _opt("anthropic", "sonnet")
MINI = _opt("openai", "mini")


def test_connection_failure_drops_same_provider() -> None:
    failed = _opt("anthropic", "opus")
    error = ConnectionFailure("anthropic", None, "ConnectError")
    assert next_candidates([HAIKU, MINI, SONNET], failed, error) == [MINI]


def test_auth_error_drops_same_provider() -> None:
    failed = _opt("anthropic", "opus")
    assert next_candidates([HAIKU, MINI], failed, AuthError("anthropic", 401, "key")) == [MINI]


def test_rate_limit_drops_same_model_other_efforts() -> None:
    error = RetryableError("anthropic", 429, "slow")
    remaining = next_candidates([OPUS_HIGH, SONNET, MINI], OPUS_LOW, error)
    assert OPUS_HIGH not in remaining
    assert SONNET in remaining and MINI in remaining


def test_model_failure_tries_other_providers_before_siblings() -> None:
    # a 5xx on one model is often the start of a provider-wide incident: prefer other providers,
    # then the failed model's siblings, each group in the selector's order
    error = RetryableError("anthropic", 503, "down")
    gem = _opt("gemini", "flash")
    remaining = next_candidates([SONNET, MINI, HAIKU, gem], OPUS_LOW, error)
    assert remaining == [MINI, gem, SONNET, HAIKU]


async def test_failover_stays_above_the_bar_first() -> None:
    sonnet, opus = _opt("anthropic", "sonnet"), _opt("anthropic", "opus")
    mini, llama = _opt("openai", "mini"), _opt("ollama", "llama")
    calls, attempt = _attempt({"sonnet": RetryableError("anthropic", 429, "slow")})
    served = await serve(
        [sonnet, opus, mini, llama], attempt, Breakers(), NO_RETRY, clear_count=2
    )
    # opus clears the quality bar; mini and llama don't, even though they're other providers
    assert calls == ["sonnet", "opus"] and served.value == "opus"
