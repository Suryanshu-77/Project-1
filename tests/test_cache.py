from typing import Any

import numpy as np
import pytest

from tierwise.cache import CachedAnswer, SemanticCache, cacheable, scope_key
from tierwise.router import RoutingDecision
from tierwise.router.modes import parse_target
from tierwise.router.router import RoutingResult, required_caps
from tierwise.schema import ChatRequest, Usage

VECTORS = {
    "what is the capital of france": np.array([1.0, 0.0, 0.0]),
    "what's the capital of france?": np.array([0.99, 0.141, 0.0]),  # cosine ~0.99
    "how do i bake bread": np.array([0.0, 1.0, 0.0]),
    "who wrote hamlet": np.array([0.0, 0.0, 1.0]),
}


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


async def fake_embed(text: str) -> np.ndarray | None:
    if text.startswith("LONG"):
        return None  # over the encoder's budget: exact matching only
    vector = VECTORS[text]
    return vector / np.linalg.norm(vector)


ANSWER = CachedAnswer(
    content="Paris", model_id="m", provider="p",
    usage=Usage(input_tokens=5, output_tokens=1), finish_reason="stop",
)  # fmt: skip


def _cache(clock: Clock | None = None, max_entries: int = 100) -> SemanticCache:
    return SemanticCache(fake_embed, 0.95, 3600, max_entries, clock=clock or Clock())


async def _put(cache: SemanticCache, scope: str, text: str) -> None:
    cache.store(await cache.key(scope, text), ANSWER)


async def _get(cache: SemanticCache, scope: str, text: str) -> CachedAnswer | None:
    return cache.lookup(await cache.key(scope, text))


async def test_similar_text_in_same_scope_hits() -> None:
    cache = _cache()
    await _put(cache, "s", "what is the capital of france")
    assert await _get(cache, "s", "what's the capital of france?") == ANSWER


async def test_below_threshold_misses() -> None:
    cache = _cache()
    await _put(cache, "s", "what is the capital of france")
    assert await _get(cache, "s", "how do i bake bread") is None


async def test_other_scope_never_hits() -> None:
    cache = _cache()
    await _put(cache, "s", "what is the capital of france")
    assert await _get(cache, "other", "what is the capital of france") is None


async def test_expired_entry_misses() -> None:
    clock = Clock()
    cache = _cache(clock)
    await _put(cache, "s", "what is the capital of france")
    clock.now = 3601
    assert await _get(cache, "s", "what is the capital of france") is None


async def test_oldest_evicted_at_capacity() -> None:
    cache = _cache(max_entries=2)
    for text in ("what is the capital of france", "how do i bake bread", "who wrote hamlet"):
        await _put(cache, "s", text)
    assert await _get(cache, "s", "how do i bake bread") == ANSWER
    assert await _get(cache, "s", "what is the capital of france") is None  # evicted first
    assert len(cache) == 2


async def test_long_text_matches_exactly_only() -> None:
    cache = _cache()
    await _put(cache, "s", "LONG " + "x" * 5000)
    assert await _get(cache, "s", "LONG " + "x" * 5000) == ANSWER
    assert await _get(cache, "s", "LONG " + "x" * 4999 + "y") is None


def _result(messages: list[dict[str, Any]], model: str = "auto", **extra: Any) -> RoutingResult:
    req = ChatRequest.model_validate({"model": model, "messages": messages, **extra})
    return RoutingResult(parse_target(model), None, req, required_caps(req))


USER = {"role": "user", "content": "what is the capital of france"}


def test_scope_changes_with_system_prompt_mode_and_caps() -> None:
    base = scope_key(_result([USER], temperature=0))
    assert base == scope_key(_result([{**USER, "content": "something else"}], temperature=0))
    system = [{"role": "system", "content": "be terse"}, USER]
    assert scope_key(_result(system, temperature=0)) != base
    assert scope_key(_result([USER], model="auto:quality", temperature=0)) != base
    json_mode = _result([USER], temperature=0, response_format={"type": "json_object"})
    assert scope_key(json_mode) != base


def test_scope_includes_routing_tag() -> None:
    plain = _result([USER], temperature=0)
    tagged = RoutingResult(
        plain.target,
        RoutingDecision(difficulty=1.0, confidence=1.0, source="override"),
        plain.request,
        plain.required_caps,
    )
    assert scope_key(tagged) != scope_key(plain)


IMAGE = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
TOOLS = [{"type": "function", "function": {"name": "f"}}]


@pytest.mark.parametrize(
    ("messages", "model", "extra", "headers", "expected"),
    [
        ([USER], "auto", {"temperature": 0.2}, {}, True),
        ([USER], "auto", {"temperature": 0.5}, {}, False),
        ([USER], "auto", {}, {}, False),  # unset temperature: the provider default, 1.0
        ([USER], "auto", {"temperature": 0, "tools": TOOLS}, {}, False),
        ([USER], "auto", {"temperature": 0}, {"cache-control": "no-cache"}, False),
        ([USER], "claude-haiku-5-5", {"temperature": 0}, {}, False),  # passthrough
        ([{"role": "user", "content": IMAGE}], "auto", {"temperature": 0}, {}, False),
        ([USER, {"role": "assistant", "content": "Paris"}], "auto", {"temperature": 0}, {}, False),
    ],
)
def test_cacheable_rules(
    messages: list[dict[str, Any]], model: str, extra: dict[str, Any],
    headers: dict[str, str], expected: bool,
) -> None:  # fmt: skip
    assert cacheable(_result(messages, model, **extra), headers, 0.3) is expected
