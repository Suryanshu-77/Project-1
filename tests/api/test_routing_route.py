import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.api.helpers import body, gateway, ok_upstream, openai_chunk, sse
from tierwise.registry import ModelSpec, Registry
from tierwise.router import router as router_module
from tierwise.schema import ChatRequest

SYSTEM = {"role": "system", "content": "You are a meticulous assistant for our team. " * 80}


def _spec(
    model_id: str, provider: str, quality: float, price_in: float, price_out: float, **kw: Any
) -> ModelSpec:
    fields: dict[str, Any] = {
        "id": model_id, "provider": provider, "quality": quality, "price_in": price_in,
        "price_out": price_out, "context": 200_000, "caps": ["tools", "vision", "json"],
        "tier": "moderate", **kw,
    }  # fmt: skip
    return ModelSpec.model_validate(fields)


ROUTING_REGISTRY = Registry(
    baseline_model="top-claude",
    models=[
        _spec("cheap-gpt", "openai", 0.55, 0.1, 0.4, caps=["tools", "json"], tier="basic",
              price_cache_read=0.01, cache_ttl_s=300),
        _spec("mid-claude", "anthropic", 0.80, 2.0, 10.0, price_cache_read=0.1, cache_ttl_s=300),
        _spec("gem-pro", "gemini", 0.90, 2.0, 12.0, context=1_000_000, tier="flagship"),
        _spec("top-claude", "anthropic", 0.95, 4.0, 20.0, tier="flagship", price_cache_read=0.2,
              cache_ttl_s=300),
    ],
)  # fmt: skip


def routed(**kw: Any) -> Any:
    return gateway(registry=ROUTING_REGISTRY, **kw)


def _turn1(text: str = "hi") -> dict[str, Any]:
    return {"model": "auto", "messages": [SYSTEM, {"role": "user", "content": text}]}


def _turn2() -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [
            SYSTEM,
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "pong"},
            {"role": "user", "content": "thanks"},
        ],
    }


async def test_auto_easy_prompt_goes_cheap() -> None:
    async with routed() as (_, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.headers["X-Tierwise-Model"] == "cheap-gpt"
    assert resp.headers["X-Tierwise-Reason"] == "rules"


async def test_auto_critical_tag_goes_flagship_and_tag_not_forwarded() -> None:
    sent: list[str] = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(request.read().decode())
        return ok_upstream(request)

    async with routed(handler=capture) as (_, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi [critical]"}]},
        )
    assert resp.headers["X-Tierwise-Model"] == "top-claude"
    assert resp.headers["X-Tierwise-Reason"] == "override"
    assert "[critical]" not in sent[0]


async def test_auto_quality_vs_cheap_shift() -> None:
    prompt = (
        "Explain how a hash map handles collisions, with a short example of separate "
        "chaining in Python."
    )
    picked = {}
    async with routed() as (_, client):
        for model in ("auto:cheap", "auto:quality"):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            )
            picked[model] = ROUTING_REGISTRY.get(resp.headers["X-Tierwise-Model"])
    assert picked["auto:quality"].quality > picked["auto:cheap"].quality  # type: ignore[union-attr]


async def test_tier_flagship() -> None:
    async with routed() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("tier:flagship"))
    assert resp.headers["X-Tierwise-Model"] == "gem-pro"
    assert resp.headers["X-Tierwise-Reason"] == "override"


async def test_no_eligible_model_422_before_upstream() -> None:
    calls: list[httpx.Request] = []

    def count(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return ok_upstream(request)

    image = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
    async with routed(handler=count, anthropic_api_key=None, gemini_api_key=None) as (_, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": image}]},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "no_eligible_model"
    assert calls == []


async def test_tag_only_message_400() -> None:
    async with routed() as (_, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "[critical]"}]},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request_error"


async def test_unknown_auto_variant_404() -> None:
    async with routed() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("auto:turbo"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


async def test_conversation_sticks_to_warm_model() -> None:
    async with routed() as (_, client):
        first = await client.post("/v1/chat/completions", json=_turn1())
        second = await client.post("/v1/chat/completions", json=_turn2())
    assert first.headers["X-Tierwise-Reason"] == "rules"
    assert second.headers["X-Tierwise-Model"] == first.headers["X-Tierwise-Model"]
    assert second.headers["X-Tierwise-Reason"] == "affinity"


async def test_failed_request_records_no_affinity() -> None:
    state = {"fail": True}

    def flaky(request: httpx.Request) -> httpx.Response:
        if state.pop("fail", False):
            return httpx.Response(503, text="down")
        return ok_upstream(request)

    async with routed(handler=flaky) as (_, client):
        first = await client.post("/v1/chat/completions", json=_turn1())
        second = await client.post("/v1/chat/completions", json=_turn2())
    assert first.status_code == 503
    assert second.headers["X-Tierwise-Reason"] == "rules"


async def test_stream_auto_routes_and_records_affinity_on_200() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if json.loads(request.read()).get("stream"):
            return sse(openai_chunk({"content": "pong"}, "stop"))
        return ok_upstream(request)

    async with routed(handler=upstream) as (_, client):
        streamed = await client.post("/v1/chat/completions", json={**_turn1(), "stream": True})
        follow_up = await client.post("/v1/chat/completions", json=_turn2())
    assert streamed.headers["X-Tierwise-Model"] == "cheap-gpt"
    assert follow_up.headers["X-Tierwise-Reason"] == "affinity"


async def test_log_row_routing_fields(tmp_path: Path) -> None:
    async with routed(db_dir=tmp_path) as (app, client):
        await client.post("/v1/chat/completions", json=_turn2())
    [row] = await app.state.log.fetch_all()
    assert (row.model, row.source, row.mode) == ("cheap-gpt", "rules", "balanced")
    assert row.difficulty is not None and row.confidence is not None
    assert row.prefix_hash is not None


async def test_scorer_failure_still_serves_with_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(req: ChatRequest) -> None:
        raise RuntimeError("bad rule")

    monkeypatch.setattr(router_module, "rules_decision", broken)
    async with routed(db_dir=tmp_path) as (app, client):
        resp = await client.post("/v1/chat/completions", json=body("auto"))
    assert resp.status_code == 200
    assert resp.headers["X-Tierwise-Reason"] == "fallback"
    [row] = await app.state.log.fetch_all()
    assert (row.source, row.difficulty) == ("fallback", 0.7)


async def test_models_lists_virtual_ids_first() -> None:
    async with routed() as (_, client):
        data = (await client.get("/v1/models")).json()["data"]
    assert [m["id"] for m in data[:7]] == [
        "auto", "auto:cheap", "auto:balanced", "auto:quality",
        "tier:basic", "tier:moderate", "tier:flagship",
    ]  # fmt: skip
    assert all(m["owned_by"] == "tierwise" for m in data[:7])
    assert {m["id"] for m in data[7:]} == {"cheap-gpt", "mid-claude", "gem-pro", "top-claude"}


async def test_stream_ending_in_error_records_no_affinity() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if json.loads(request.read()).get("stream"):
            return sse(openai_chunk({"content": "po"}), done=False, raw_tail=b"data: {broken\n\n")
        return ok_upstream(request)

    async with routed(handler=upstream) as (_, client):
        broken = await client.post("/v1/chat/completions", json={**_turn1(), "stream": True})
        follow_up = await client.post("/v1/chat/completions", json=_turn2())
    assert '"error"' in broken.text  # stream ended with an in-band error (logged 502)
    assert follow_up.headers["X-Tierwise-Reason"] == "rules"


def _effort_registry(default: str = "medium") -> Registry:
    levels = {
        "low": {"quality": -0.15, "thinking_tokens": 100},
        "medium": {"quality": 0.0, "thinking_tokens": 500},
        "high": {"quality": 0.03, "thinking_tokens": 2000},
    }
    if default == "low":
        levels = {"low": {"quality": 0.0, "thinking_tokens": 100},
                  "medium": {"quality": 0.03, "thinking_tokens": 500}}  # fmt: skip
    model = _spec(
        "eff-gpt", "openai", 0.80, 0.5, 2.0, efforts={"default": default, "levels": levels}
    )
    return Registry(baseline_model="eff-gpt", models=[model])


def _capture() -> tuple[list[dict[str, Any]], Any]:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        reply = {
            "id": "c", "model": "eff-gpt", "created": 1,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 9,
                      "completion_tokens_details": {"reasoning_tokens": 7}},
        }  # fmt: skip
        if sent[-1].get("stream"):
            return sse(openai_chunk({"content": "pong"}, "stop"))
        return httpx.Response(200, json=reply)

    return sent, handler


def _hi(**extra: Any) -> dict[str, Any]:
    return {"model": "auto", "messages": [{"role": "user", "content": "hi"}], **extra}


async def test_easy_auto_request_sends_low_effort_and_header() -> None:
    sent, handler = _capture()
    async with gateway(handler, registry=_effort_registry()) as (_, client):
        resp = await client.post("/v1/chat/completions", json=_hi())
    assert resp.headers["X-Tierwise-Effort"] == "low"
    assert sent[0]["reasoning_effort"] == "low"


async def test_default_level_sends_no_effort_field_and_header_default() -> None:
    sent, handler = _capture()
    async with gateway(handler, registry=_effort_registry(default="low")) as (_, client):
        resp = await client.post("/v1/chat/completions", json=_hi())
    assert resp.headers["X-Tierwise-Effort"] == "default"
    assert "reasoning_effort" not in sent[0]


async def test_client_effort_wins_on_auto() -> None:
    sent, handler = _capture()
    async with gateway(handler, registry=_effort_registry()) as (_, client):
        resp = await client.post("/v1/chat/completions", json=_hi(reasoning_effort="high"))
    assert resp.headers["X-Tierwise-Effort"] == "high"
    assert sent[0]["reasoning_effort"] == "high"


async def test_client_effort_unsupported_on_auto_422_before_upstream() -> None:
    sent, handler = _capture()
    async with gateway(handler, registry=_effort_registry()) as (_, client):
        resp = await client.post("/v1/chat/completions", json=_hi(reasoning_effort="max"))
    assert resp.status_code == 422
    assert "max" in resp.json()["error"]["message"]
    assert sent == []


async def test_concrete_model_forwards_client_effort_unchanged() -> None:
    sent, handler = _capture()
    async with gateway(handler, registry=_effort_registry()) as (_, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={**_hi(reasoning_effort="minimal"), "model": "eff-gpt"},
        )
    assert resp.headers["X-Tierwise-Effort"] == "minimal"
    assert sent[0]["reasoning_effort"] == "minimal"


async def test_log_row_has_effort_and_reasoning_tokens(tmp_path: Path) -> None:
    _, handler = _capture()
    async with gateway(handler, registry=_effort_registry(), db_dir=tmp_path) as (app, client):
        await client.post("/v1/chat/completions", json=_hi())
    [row] = await app.state.log.fetch_all()
    assert (row.effort, row.tokens_reasoning) == ("low", 7)


async def test_stream_sets_effort_header() -> None:
    _, handler = _capture()
    async with gateway(handler, registry=_effort_registry()) as (_, client):
        resp = await client.post("/v1/chat/completions", json=_hi(stream=True))
    assert resp.headers["X-Tierwise-Effort"] == "low"


async def test_concrete_model_forwards_off_scale_effort() -> None:
    sent, handler = _capture()
    async with gateway(handler, registry=_effort_registry()) as (_, client):
        resp = await client.post(
            "/v1/chat/completions", json={**_hi(reasoning_effort="none"), "model": "eff-gpt"}
        )
    assert resp.status_code == 200
    assert sent[0]["reasoning_effort"] == "none"
    assert resp.headers["X-Tierwise-Effort"] == "none"


async def test_off_scale_effort_on_auto_422_naming_it() -> None:
    sent, handler = _capture()
    async with gateway(handler, registry=_effort_registry()) as (_, client):
        resp = await client.post("/v1/chat/completions", json=_hi(reasoning_effort="none"))
    assert resp.status_code == 422
    assert "none" in resp.json()["error"]["message"]
    assert sent == []


async def test_log_stores_level_that_ran_not_default(tmp_path: Path) -> None:
    _, handler = _capture()
    registry = _effort_registry(default="low")
    async with gateway(handler, registry=registry, db_dir=tmp_path) as (app, client):
        resp = await client.post("/v1/chat/completions", json=_hi())
    assert resp.headers["X-Tierwise-Effort"] == "default"  # nothing was sent
    [row] = await app.state.log.fetch_all()
    assert row.effort == "low"  # but the provider ran at its default level, low
