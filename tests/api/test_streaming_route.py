import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.api.helpers import body, gateway, openai_chunk, sse
from tierwise.api.streaming import sse_response
from tierwise.schema import StreamChunk, Usage


def _events(text: str) -> list[Any]:
    payloads = [
        line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")
    ]
    return [p if p == "[DONE]" else json.loads(p) for p in payloads]


def _usage_event() -> dict[str, Any]:
    return {"id": "c", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}


async def test_stream_happy_path_openai_format(tmp_path: Path) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return sse(
            openai_chunk({"content": "po"}), openai_chunk({"content": "ng"}, "stop"), _usage_event()
        )

    async with gateway(upstream, db_dir=tmp_path) as (app, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini", stream=True))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["X-Tierwise-Provider"] == "openai"
    events = _events(resp.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events[:-1] if e["choices"]]
    assert all(e["object"] == "chat.completion.chunk" for e in chunks)
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in chunks) == "pong"
    [row] = await app.state.log.fetch_all()
    assert row.stream is True
    assert (row.status, row.tokens_in, row.tokens_out) == (200, 4, 2)
    assert row.cost_usd == pytest.approx((4 * 1.0 + 2 * 2.0) / 1e6)


async def test_stream_upstream_429_before_first_chunk_is_http_503() -> None:
    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"})

    async with gateway(limited) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini", stream=True))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "upstream_unavailable"


async def test_stream_error_after_first_chunk_emits_error_event(tmp_path: Path) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        return sse(openai_chunk({"content": "po"}), done=False, raw_tail=b"data: {broken\n\n")

    async with gateway(broken, db_dir=tmp_path) as (app, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini", stream=True))
    events = _events(resp.text)
    assert events[0]["choices"][0]["delta"] == {"content": "po"}
    assert events[-2]["error"]["code"] == "upstream_unavailable"
    assert events[-1] == "[DONE]"
    [row] = await app.state.log.fetch_all()
    assert row.status == 502


async def test_stream_without_usage_logs_null_cost(tmp_path: Path) -> None:
    def no_usage(request: httpx.Request) -> httpx.Response:
        return sse(openai_chunk({"content": "pong"}, "stop"))

    async with gateway(no_usage, db_dir=tmp_path) as (app, client):
        resp = await client.post("/v1/chat/completions", json=body("ollama/llama3.2", stream=True))
    assert _events(resp.text)[-1] == "[DONE]"
    [row] = await app.state.log.fetch_all()
    assert row.status == 200
    assert (row.tokens_in, row.cost_usd) == (None, None)


async def test_stream_client_disconnect_closes_upstream_and_logs_499() -> None:
    # httpx's ASGITransport buffers whole responses, so drive the SSE body directly the way
    # Starlette does when a client goes away: take one event, then close the iterator.
    closed: list[bool] = []
    outcome: list[tuple[Usage | None, int, str | None]] = []

    async def upstream() -> AsyncGenerator[StreamChunk, None]:
        try:
            yield StreamChunk(content="b")
            yield StreamChunk(content="c")
        finally:
            closed.append(True)

    chunks = upstream()
    first = await anext(chunks)
    response = sse_response(
        chunks,
        first,
        request_id="r",
        model="m",
        headers={},
        on_done=lambda usage, status, code: outcome.append((usage, status, code)),
    )
    iterator = response.body_iterator
    await anext(iterator)
    await iterator.aclose()  # type: ignore[attr-defined]
    assert closed == [True]
    assert outcome == [(None, 499, "client_closed")]
