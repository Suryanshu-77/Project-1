from pathlib import Path

import httpx

from tests.api.helpers import OPENAI_REPLY, ok_upstream
from tests.api.test_routing_route import routed

HI = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}


def _hosts(failing: dict[str, httpx.Response]) -> tuple[list[str], object]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return failing.get(request.url.host) or ok_upstream(request)

    return calls, handler


async def test_429_on_first_model_fails_over_and_logs(tmp_path: Path) -> None:
    calls, handler = _hosts({"api.openai.com": httpx.Response(429, json={"error": "slow"})})
    async with routed(handler=handler, db_dir=tmp_path) as (app, client):
        resp = await client.post("/v1/chat/completions", json=HI)
    assert resp.status_code == 200
    assert resp.headers["X-Tierwise-Provider"] != "openai"
    assert resp.headers["X-Tierwise-Failovers"] == "1"
    assert calls[0] == "api.openai.com"
    [row] = await app.state.log.fetch_all()
    assert row.failover_count == 1
    assert row.model == resp.headers["X-Tierwise-Model"]


async def test_bad_request_is_not_failed_over() -> None:
    calls, handler = _hosts({"api.openai.com": httpx.Response(400, json={"error": "bad"})})
    async with routed(handler=handler) as (_, client):
        resp = await client.post("/v1/chat/completions", json=HI)
    assert resp.status_code == 400
    assert calls == ["api.openai.com"]


async def test_concrete_model_retries_but_never_fails_over() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "blip"})
        return httpx.Response(200, json=OPENAI_REPLY)

    async with routed(handler=handler, retries=1, retry_base_s=0.0) as (_, client):
        ok = await client.post("/v1/chat/completions", json={**HI, "model": "cheap-gpt"})
    assert ok.status_code == 200 and calls == ["api.openai.com"] * 2

    calls.clear()
    down = lambda request: (calls.append(request.url.host), httpx.Response(503, json={}))[1]  # noqa: E731
    async with routed(handler=down, retries=1, retry_base_s=0.0) as (_, client):
        resp = await client.post("/v1/chat/completions", json={**HI, "model": "cheap-gpt"})
    assert resp.status_code == 503
    assert set(calls) == {"api.openai.com"}
