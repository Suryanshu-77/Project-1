from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from tierwise.api.app import create_app
from tierwise.config import Settings

ANTHROPIC_KEY = "sk-ant-secret-123"
OPENAI_KEY = "sk-oa-secret-456"
Handler = Callable[[httpx.Request], httpx.Response]

ANTHROPIC_REPLY = {
    "id": "msg_1",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "pong"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "model": "gpt-4.1-mini",
    "created": 1,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


def ok_upstream(request: httpx.Request) -> httpx.Response:
    if request.url.host == "api.anthropic.com":
        return httpx.Response(200, json=ANTHROPIC_REPLY)
    return httpx.Response(200, json=OPENAI_REPLY)


@asynccontextmanager
async def gateway(
    handler: Handler = ok_upstream, **settings: Any
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    config = Settings(
        _env_file=None,  # type: ignore[call-arg]
        **{"anthropic_api_key": ANTHROPIC_KEY, "openai_api_key": OPENAI_KEY, **settings},
    )
    app = create_app(config, transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
            yield app, client


def body(model: str, **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "ping"}], **extra}


async def test_routes_claude_to_anthropic() -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("claude-haiku-5-5"))
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "pong"
    assert resp.headers["X-Tierwise-Provider"] == "anthropic"
    assert resp.headers["X-Tierwise-Model"] == "claude-haiku-5-5"
    assert resp.headers["X-Tierwise-Reason"] == "override"
    request_id = resp.headers["X-Tierwise-Request-Id"]
    assert len(request_id) == 32
    int(request_id, 16)


async def test_routes_gpt_to_openai() -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
    assert resp.status_code == 200
    assert resp.headers["X-Tierwise-Provider"] == "openai"


async def test_unknown_model_404() -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("mystery-model"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


async def test_unconfigured_provider_404() -> None:
    async with gateway(anthropic_api_key=None) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("claude-haiku-5-5"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


async def test_stream_rejected_400() -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini", stream=True))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_parameter"


async def test_bad_body_400_openai_shape() -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4.1-mini"})
    assert resp.status_code == 400
    payload = resp.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "detail" not in payload


async def test_upstream_timeout_503() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with gateway(timeout) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "upstream_unavailable"


async def test_upstream_401_maps_502_without_key() -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"invalid x-api-key {ANTHROPIC_KEY}"})

    async with gateway(reject) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("claude-haiku-5-5"))
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_auth_error"
    assert ANTHROPIC_KEY not in resp.text


async def test_upstream_400_passes_status_and_message() -> None:
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "max_tokens too large"})

    async with gateway(bad) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
    assert resp.status_code == 400
    assert "max_tokens too large" in resp.json()["error"]["message"]


async def test_clients_are_pooled_singletons() -> None:
    async with gateway() as (app, client):
        before = app.state.clients["openai"]
        for _ in range(2):
            await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
        assert app.state.clients["openai"] is before
        assert app.state.providers["openai"]._client is before
    assert before.is_closed


async def test_healthz_lists_configured_providers() -> None:
    async with gateway(openai_api_key=None) as (_, client):
        resp = await client.get("/healthz")
    assert resp.json() == {"status": "ok", "providers": ["anthropic"]}
