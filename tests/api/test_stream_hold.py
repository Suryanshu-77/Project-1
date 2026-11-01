import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.api.helpers import gateway, openai_chunk, sse
from tierwise.api.streaming import prime_stream, sse_response
from tierwise.providers.base import RetryableError
from tierwise.registry import ModelSpec, Registry
from tierwise.schema import StreamChunk, Usage


def _spec(model_id: str, provider: str, price: float) -> ModelSpec:
    return ModelSpec.model_validate(
        {"id": model_id, "provider": provider, "price_in": price, "price_out": price,
         "context": 100_000, "caps": ["tools", "json"], "quality": 0.9, "tier": "basic",
         "upstream_id": "llama3.2" if provider == "ollama" else None}  # fmt: skip
    )


# Two OpenAI-format upstreams: the cheap one is tried first, ollama is the failover.
TWO = Registry(
    baseline_model="local", models=[_spec("cheap", "openai", 0.1), _spec("local", "ollama", 1.0)]
)
HI = {"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


def _contents(text: str) -> list[str]:
    events = [line.removeprefix("data: ") for line in text.split("\n") if line.startswith("data:")]
    out = []
    for e in events:
        if e != "[DONE]":
            for choice in json.loads(e).get("choices", []):
                if choice["delta"].get("content"):
                    out.append(choice["delta"]["content"])
    return out


async def test_error_before_first_content_fails_over(tmp_path: Path) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":  # role-only preamble, then a broken event
            return sse(openai_chunk({"role": "assistant"}), done=False, raw_tail=b"data: {x\n\n")
        return sse(openai_chunk({"content": "pong"}), openai_chunk({}, "stop"))

    async with gateway(upstream, db_dir=tmp_path, registry=TWO) as (app, client):
        resp = await client.post("/v1/chat/completions", json=HI)
    assert resp.status_code == 200
    assert resp.headers["X-Tierwise-Provider"] == "ollama"
    assert _contents(resp.text) == ["pong"]
    assert '"error"' not in resp.text
    [row] = await app.state.log.fetch_all()
    assert (row.status, row.failover_count) == (200, 1)


async def test_error_after_first_content_is_in_band(tmp_path: Path) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return sse(openai_chunk({"content": "po"}), done=False, raw_tail=b"data: {x\n\n")

    async with gateway(upstream, db_dir=tmp_path, registry=TWO) as (app, client):
        resp = await client.post("/v1/chat/completions", json=HI)
    assert resp.status_code == 200
    assert _contents(resp.text) == ["po"]
    assert '"error"' in resp.text
    [row] = await app.state.log.fetch_all()
    assert row.status == 502


async def _gen(*steps: Any, closed: list[bool] | None = None) -> AsyncGenerator[StreamChunk, None]:
    try:
        for step in steps:
            if isinstance(step, float):
                await asyncio.sleep(step)
            elif isinstance(step, Exception):
                raise step
            else:
                yield step
    finally:
        if closed is not None:
            closed.append(True)


async def _drain(primed: Any) -> tuple[str, list[tuple[Usage | None, int, str | None]]]:
    outcome: list[tuple[Usage | None, int, str | None]] = []
    response = sse_response(
        primed, request_id="r", model="m", headers={}, include_usage=False,
        on_done=lambda u, s, c: outcome.append((u, s, c)),
    )  # fmt: skip
    text = "".join([part async for part in response.body_iterator])  # type: ignore[misc]
    return text, outcome


async def test_slow_preamble_commits_after_hold() -> None:
    chunks = _gen(StreamChunk(), 0.2, StreamChunk(content="x"))
    primed = await prime_stream(chunks, hold_s=0.05)
    assert primed.pending is not None and len(primed.buffered) == 1
    text, outcome = await _drain(primed)
    assert _contents(text) == ["x"]
    assert text.count("data: [DONE]") == 1
    assert outcome == [(None, 200, None)]


async def test_hold_byte_cap() -> None:
    big = Usage(input_tokens=1, output_tokens=1)
    chunks = _gen(*[StreamChunk(usage=big) for _ in range(100)], StreamChunk(content="late"))
    primed = await prime_stream(chunks, hold_bytes=200)
    assert primed.pending is None
    assert 1 < len(primed.buffered) < 100
    assert not any(c.content for c in primed.buffered)
    await primed.chunks.aclose()


async def test_error_inside_window_raises_and_closes() -> None:
    closed: list[bool] = []
    chunks = _gen(StreamChunk(), RetryableError("p", 500, "boom"), closed=closed)
    with pytest.raises(RetryableError):
        await prime_stream(chunks)
    assert closed == [True]


async def test_client_cancel_during_hold_closes_upstream() -> None:
    closed: list[bool] = []
    chunks = _gen(StreamChunk(), 5.0, StreamChunk(content="x"), closed=closed)
    task = asyncio.ensure_future(prime_stream(chunks, hold_s=10))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]


async def test_cancel_with_pending_read_closes_upstream_and_logs_499() -> None:
    closed: list[bool] = []
    chunks = _gen(StreamChunk(), 5.0, StreamChunk(content="x"), closed=closed)
    primed = await prime_stream(chunks, hold_s=0.01)
    assert primed.pending is not None
    outcome: list[tuple[Usage | None, int, str | None]] = []
    response = sse_response(
        primed, request_id="r", model="m", headers={}, include_usage=False,
        on_done=lambda u, s, c: outcome.append((u, s, c)),
    )  # fmt: skip
    iterator = response.body_iterator
    await anext(iterator)  # the buffered (empty) preamble chunk
    reader = asyncio.ensure_future(anext(iterator))  # blocks on the read still in flight
    await asyncio.sleep(0.05)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    await iterator.aclose()  # type: ignore[attr-defined]
    assert closed == [True]
    assert outcome == [(None, 499, "client_closed")]
