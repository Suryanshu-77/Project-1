import pytest

from tierwise.costs import cost_usd
from tierwise.registry import ModelSpec
from tierwise.schema import Usage


def _spec(cache_read: float | None = 0.3, cache_write: float | None = 3.75) -> ModelSpec:
    return ModelSpec(
        id="m",
        provider="anthropic",
        price_in=3.0,
        price_out=15.0,
        price_cache_read=cache_read,
        price_cache_write=cache_write,
        context=1000,
        caps=set(),
        quality=0.5,
        tier="moderate",
    )


def test_plain() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(usage, _spec()) == pytest.approx(18.0)


def test_cached_read_discounted() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=800_000)
    assert cost_usd(usage, _spec()) == pytest.approx(0.2 * 3.0 + 0.8 * 0.3)


def test_cache_write_priced() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=0, cache_write_tokens=1_000_000)
    assert cost_usd(usage, _spec()) == pytest.approx(3.75)


def test_null_cache_prices_fall_back_to_price_in() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=500_000)
    assert cost_usd(usage, _spec(cache_read=None, cache_write=None)) == pytest.approx(3.0)


def test_zero_usage_is_zero() -> None:
    assert cost_usd(Usage(input_tokens=0, output_tokens=0), _spec()) == 0.0
