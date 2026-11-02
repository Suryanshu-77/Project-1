from pathlib import Path

import httpx
import pytest

from tests.api.helpers import ANTHROPIC_KEY, body, gateway


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


@pytest.mark.parametrize(
    ("model", "provider"),
    [("gpt-4.1-mini", "openai"), ("gemini-x", "gemini"), ("ollama/llama3.2", "ollama")],
)
async def test_routes_by_registry(model: str, provider: str) -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body(model))
    assert resp.status_code == 200
    assert resp.headers["X-Tierwise-Provider"] == provider
    assert resp.json()["choices"][0]["message"]["content"] == "pong"


async def test_ollama_receives_upstream_model_name() -> None:
    seen: list[str] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.read().decode())
        from tests.api.helpers import ok_upstream

        return ok_upstream(request)

    async with gateway(capture) as (_, client):
        await client.post("/v1/chat/completions", json=body("ollama/llama3.2"))
    assert '"model":"llama3.2"' in seen[0].replace(" ", "")


async def test_cost_header_and_log_row(tmp_path: Path) -> None:
    async with gateway(db_dir=tmp_path) as (app, client):
        resp = await client.post("/v1/chat/completions", json=body("claude-haiku-5-5"))
    assert resp.headers["X-Tierwise-Cost-USD"] == "0.000005"  # 3 in x $1 + 1 out x $2, per 1M
    [row] = await app.state.log.fetch_all()
    assert row.status == 200
    assert row.stream is False
    assert row.cost_usd == pytest.approx(5e-6)
    assert row.baseline_cost_usd == pytest.approx(5e-6)
    assert (row.tokens_in, row.tokens_out) == (3, 1)


async def test_model_not_in_registry_404() -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("mystery-model"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


async def test_registry_model_provider_unconfigured_404() -> None:
    async with gateway(anthropic_api_key=None) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("claude-haiku-5-5"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


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


async def test_error_request_is_logged(tmp_path: Path) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    async with gateway(unavailable, db_dir=tmp_path) as (app, client):
        await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
    [row] = await app.state.log.fetch_all()
    assert (row.status, row.error_code) == (503, "upstream_unavailable")
    assert row.cost_usd is None


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


async def test_malformed_upstream_2xx_is_503_not_500() -> None:
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    async with gateway(empty) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "upstream_unavailable"


async def test_clients_are_pooled_singletons() -> None:
    async with gateway() as (app, client):
        before = app.state.clients["openai"]
        for _ in range(2):
            await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
        assert app.state.clients["openai"] is before
        assert app.state.providers["openai"]._client is before
    assert before.is_closed


async def test_healthz_lists_configured_providers() -> None:
    async with gateway(openai_api_key=None, ollama_base_url=None) as (_, client):
        resp = await client.get("/healthz")
    assert resp.json() == {"status": "ok", "providers": ["anthropic", "gemini"],
                           "router_version": "rules-only"}  # fmt: skip


async def test_unexpected_provider_exception_is_500_shape_and_logged(tmp_path: Path) -> None:
    async def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bug in adapter")

    async with gateway(db_dir=tmp_path) as (app, client):
        app.state.providers["openai"].chat = explode
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"
    [row] = await app.state.log.fetch_all()
    assert (row.status, row.error_code) == (500, "internal_error")


async def test_healthz_reports_router_version() -> None:
    async with gateway() as (_, client):  # learned_router=False in the test gateway
        assert (await client.get("/healthz")).json()["router_version"] == "rules-only"
