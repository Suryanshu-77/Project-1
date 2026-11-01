from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.api.helpers import gateway, make_key, ok_upstream, openai_chunk, sse
from tests.api.test_stream_hold import TWO
from tierwise.schema import ChatRequest
from tierwise.store import ShadowSample

SHADOW = {"store_payloads": True, "shadow_rate": 1.0, "require_api_key": True}


def _ask(stream: bool = False, model: str = "auto") -> dict[str, Any]:
    return {"model": model, "stream": stream,
            "messages": [{"role": "user", "content": "capital of france? [easy]"}]}  # fmt: skip


def _streaming(request: httpx.Request) -> httpx.Response:
    return sse(
        openai_chunk({"content": "Par"}),
        openai_chunk({"content": "is"}, "stop"),
        {"id": "c", "model": "m", "created": 1, "choices": [],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2}},  # fmt: skip
    )


async def _samples(app: Any) -> list[ShadowSample]:
    return list(await app.state.log.pending_samples(100))


async def test_sampled_non_stream_request_stores_row(tmp_path: Path) -> None:
    async with gateway(ok_upstream, db_dir=tmp_path, **SHADOW) as (app, client):
        key, row = await make_key(app, allow_shadow=True)
        resp = await client.post("/v1/chat/completions", json=_ask(),
                                 headers={"Authorization": f"Bearer {key}"})  # fmt: skip
        assert resp.status_code == 200
        request_id = resp.headers["X-Tierwise-Request-Id"]
    [sample] = await _samples(app)
    assert sample.id == request_id and sample.key_id == row.id
    assert sample.routed_answer == "pong"
    assert sample.routed_model == resp.headers["X-Tierwise-Model"]
    stored = ChatRequest.model_validate_json(sample.request_json)
    assert stored.messages[-1].content == "capital of france?"  # routing tag already stripped
    assert sample.source == "override" and sample.status == "pending"
    assert app.state.metrics.registry.get_sample_value("tierwise_shadow_samples_total") == 1.0


async def test_sampled_stream_stores_full_text(tmp_path: Path) -> None:
    async with gateway(_streaming, db_dir=tmp_path, registry=TWO, **SHADOW) as (app, client):
        key, _ = await make_key(app, allow_shadow=True)
        resp = await client.post("/v1/chat/completions", json=_ask(stream=True),
                                 headers={"Authorization": f"Bearer {key}"})  # fmt: skip
        assert resp.status_code == 200
    [sample] = await _samples(app)
    assert sample.routed_answer == "Paris"


async def test_no_allow_shadow_never_samples(tmp_path: Path) -> None:
    async with gateway(ok_upstream, db_dir=tmp_path, **SHADOW) as (app, client):
        key, _ = await make_key(app)
        for _ in range(5):
            await client.post("/v1/chat/completions", json=_ask(),
                              headers={"Authorization": f"Bearer {key}"})  # fmt: skip
    assert await _samples(app) == []


async def test_concrete_model_request_never_samples(tmp_path: Path) -> None:
    async with gateway(ok_upstream, db_dir=tmp_path, **SHADOW) as (app, client):
        key, _ = await make_key(app, allow_shadow=True)
        resp = await client.post("/v1/chat/completions", json=_ask(model="gpt-4.1-mini"),
                                 headers={"Authorization": f"Bearer {key}"})  # fmt: skip
        assert resp.status_code == 200
    assert await _samples(app) == []


async def test_sampler_bug_never_breaks_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tierwise.api import routes

    def broken(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("flywheel bug")

    monkeypatch.setattr(routes, "should_sample", broken)
    async with gateway(ok_upstream, db_dir=tmp_path, **SHADOW) as (app, client):
        key, _ = await make_key(app, allow_shadow=True)
        resp = await client.post("/v1/chat/completions", json=_ask(),
                                 headers={"Authorization": f"Bearer {key}"})  # fmt: skip
    assert resp.status_code == 200 and resp.json()["choices"][0]["message"]["content"] == "pong"
