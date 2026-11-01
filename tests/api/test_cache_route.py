import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from tests.api.helpers import gateway, ok_upstream, openai_chunk, sse
from tests.api.test_routing_route import routed
from tests.api.test_stream_hold import TWO
from tierwise.cache import SemanticCache


async def _embed(text: str) -> np.ndarray | None:
    return np.array([1.0, 0.0])  # every short question looks alike: the scope decides


def _counting(stream: bool = False) -> tuple[list[str], Any]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if stream:
            return sse(
                openai_chunk({"content": "Par"}),
                openai_chunk({"content": "is"}, "stop"),
                {"id": "c", "model": "m", "created": 1, "choices": [],
                 "usage": {"prompt_tokens": 5, "completion_tokens": 2}},  # fmt: skip
            )
        return ok_upstream(request)

    return calls, handler


def _ask(stream: bool = False, **extra: Any) -> dict[str, Any]:
    return {"model": "auto", "temperature": 0, "stream": stream,
            "messages": [{"role": "user", "content": "capital of france?"}], **extra}  # fmt: skip


async def test_second_identical_request_is_served_from_cache(tmp_path: Path) -> None:
    calls, handler = _counting()
    async with routed(handler=handler, db_dir=tmp_path) as (app, client):
        app.state.cache = SemanticCache(_embed)
        first = await client.post("/v1/chat/completions", json=_ask())
        second = await client.post("/v1/chat/completions", json=_ask())
    assert len(calls) == 1
    assert second.status_code == 200
    assert second.headers["X-Tierwise-Reason"] == "cache"
    assert second.headers["X-Tierwise-Model"] == first.headers["X-Tierwise-Model"]
    assert second.json()["choices"][0]["message"]["content"] == "pong"
    rows = sorted(await app.state.log.fetch_all(), key=lambda r: r.ts)
    assert [r.cache_hit for r in rows] == [False, True]
    assert rows[1].cost_usd == 0 and (rows[1].baseline_cost_usd or 0) > 0


async def test_streaming_hit_is_valid_sse() -> None:
    calls, handler = _counting(stream=True)
    asked = _ask(stream=True, stream_options={"include_usage": True})
    async with gateway(handler, registry=TWO) as (app, client):  # OpenAI-format streams
        app.state.cache = SemanticCache(_embed)
        await client.post("/v1/chat/completions", json=asked)
        hit = await client.post("/v1/chat/completions", json=asked)
    assert len(calls) == 1 and hit.headers["X-Tierwise-Reason"] == "cache"
    lines = hit.text.split("\n")
    events = [line.removeprefix("data: ") for line in lines if line.startswith("data:")]
    assert events[-1] == "[DONE]"
    payloads = [json.loads(e) for e in events[:-1]]
    deltas = [p["choices"][0]["delta"] for p in payloads if p["choices"]]
    content = "".join(d.get("content") or "" for d in deltas)
    assert content == "Paris"
    assert any(p["choices"] and p["choices"][0]["finish_reason"] == "stop" for p in payloads)
    assert payloads[-1]["usage"]["completion_tokens"] == 2


async def test_no_cache_header_bypasses() -> None:
    calls, handler = _counting()
    async with routed(handler=handler) as (app, client):
        app.state.cache = SemanticCache(_embed)
        await client.post("/v1/chat/completions", json=_ask())
        resp = await client.post(
            "/v1/chat/completions", json=_ask(), headers={"Cache-Control": "no-cache"}
        )
    assert len(calls) == 2 and resp.headers["X-Tierwise-Reason"] != "cache"


async def test_failed_request_is_not_cached() -> None:
    state = {"fail": 3}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"] > 0:
            state["fail"] -= 1
            return httpx.Response(503, json={})
        return ok_upstream(request)

    async with routed(handler=handler) as (app, client):
        app.state.cache = SemanticCache(_embed)
        failed = await client.post("/v1/chat/completions", json=_ask())
        retried = await client.post("/v1/chat/completions", json=_ask())
    assert failed.status_code == 503
    assert retried.headers["X-Tierwise-Reason"] != "cache"
