"""Spec §7 exit scenarios, end to end: kill or slow a provider and requests still succeed."""

from pathlib import Path
from typing import Any

import httpx

from tests.api.helpers import gateway, ok_upstream, openai_chunk, sse
from tests.api.test_routing_route import routed
from tests.api.test_stream_hold import _spec
from tierwise.registry import Registry
from tierwise.reliability.breaker import Breakers

HI = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
OPENAI = "api.openai.com"


def _recording(fail: Any) -> tuple[list[str], Any]:
    """Upstream that answers normally unless `fail(request)` returns a response or raises."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return fail(request) or ok_upstream(request)

    return calls, handler


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_invalid_key_on_primary_provider_fails_over(tmp_path: Path) -> None:
    def bad_key(request: httpx.Request) -> httpx.Response | None:
        if request.url.host == OPENAI:
            return httpx.Response(401, json={"error": "bad key"})
        return None

    calls, handler = _recording(bad_key)
    async with routed(handler=handler, db_dir=tmp_path) as (_, client):
        for _ in range(5):
            resp = await client.post("/v1/chat/completions", json=HI)
            assert resp.status_code == 200 and resp.headers["X-Tierwise-Provider"] != "openai"
        assert calls.count(OPENAI) == 5
        sixth = await client.post("/v1/chat/completions", json=HI)
    assert sixth.status_code == 200
    assert calls.count(OPENAI) == 5  # the provider breaker is open: openai is not called
    assert sixth.headers["X-Tierwise-Failovers"] == "0"


async def test_timeout_on_primary_fails_over() -> None:
    def slow(request: httpx.Request) -> None:
        if request.url.host == OPENAI:
            raise httpx.ReadTimeout("timed out", request=request)

    calls, handler = _recording(slow)
    async with routed(handler=handler) as (_, client):
        resp = await client.post("/v1/chat/completions", json=HI)
    assert resp.status_code == 200
    assert calls[0] == OPENAI and resp.headers["X-Tierwise-Provider"] != "openai"


async def test_breaker_half_open_probe_recovers() -> None:
    state = {"down": True}

    def flaky(request: httpx.Request) -> httpx.Response | None:
        down = state["down"] and request.url.host == OPENAI
        return httpx.Response(503, json={"error": "down"}) if down else None

    clock = Clock()
    calls, handler = _recording(flaky)
    async with routed(handler=handler) as (app, client):
        app.state.breakers = Breakers(clock=clock)
        for _ in range(5):
            await client.post("/v1/chat/completions", json=HI)
        calls.clear()
        await client.post("/v1/chat/completions", json=HI)
        assert OPENAI not in calls  # open: skipped
        clock.now = 30
        state["down"] = False
        probe = await client.post("/v1/chat/completions", json=HI)
        assert probe.headers["X-Tierwise-Provider"] == "openai"  # the half-open probe served
        after = await client.post("/v1/chat/completions", json=HI)
    assert after.headers["X-Tierwise-Provider"] == "openai"
    assert app.state.breakers.model("openai", "cheap-gpt").state == "closed"


# cheap and its sibling on openai, the failover on ollama; all speak OpenAI's stream format
THREE = Registry(
    baseline_model="local",
    models=[_spec("cheap", "openai", 0.1), _spec("cheap2", "openai", 0.2),
            _spec("local", "ollama", 1.0)],
)  # fmt: skip


async def test_early_stream_abort_fails_over_end_to_end(tmp_path: Path) -> None:
    calls: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == OPENAI:
            return sse(openai_chunk({"role": "assistant"}), done=False, raw_tail=b"data: {x\n\n")
        return sse(openai_chunk({"content": "pong"}), openai_chunk({}, "stop"))

    stream_hi = {**HI, "stream": True}
    async with gateway(upstream, db_dir=tmp_path, registry=THREE) as (app, client):
        resp = await client.post("/v1/chat/completions", json=stream_hi)
    assert resp.status_code == 200
    assert '"error"' not in resp.text and "pong" in resp.text
    # after a model-level failure, other providers are tried before the failed one's siblings
    assert calls == [OPENAI, "ollama.test"]
    [row] = await app.state.log.fetch_all()
    assert (row.status, row.failover_count, row.model) == (200, 1, "local")


async def test_streams_failing_after_first_content_trip_the_breaker() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == OPENAI:  # content first, then a broken event, every time
            return sse(openai_chunk({"content": "po"}), done=False, raw_tail=b"data: {x\n\n")
        return sse(openai_chunk({"content": "pong"}), openai_chunk({}, "stop"))

    async with gateway(upstream, registry=THREE) as (app, client):
        for _ in range(5):
            await client.post("/v1/chat/completions", json={**HI, "stream": True})
        assert app.state.breakers.model("openai", "cheap").state == "open"
        after = await client.post("/v1/chat/completions", json={**HI, "stream": True})
    assert after.headers["X-Tierwise-Model"] != "cheap"  # skipped; its sibling may serve
