"""Turn token usage into dollars using a model's registry prices."""

from tierwise.registry import ModelSpec
from tierwise.schema import Usage

PER_TOKENS = 1_000_000  # registry prices are USD per 1M tokens


def cost_usd(usage: Usage, spec: ModelSpec) -> float:
    """Price one request.

    ``usage.input_tokens`` is the whole prompt; ``cached_input_tokens`` and
    ``cache_write_tokens`` are the parts of it that were read from, or written to,
    the provider's prompt cache. A ``None`` cache price means "same as price_in".
    """
    p_read = spec.price_cache_read if spec.price_cache_read is not None else spec.price_in
    p_write = spec.price_cache_write if spec.price_cache_write is not None else spec.price_in
    fresh = max(
        0, usage.input_tokens - (usage.cached_input_tokens or 0) - (usage.cache_write_tokens or 0)
    )

    total = (
        fresh * spec.price_in
        + (usage.cached_input_tokens or 0) * p_read
        + (usage.cache_write_tokens or 0) * p_write
        + usage.output_tokens * spec.price_out
    )
    return total / PER_TOKENS
