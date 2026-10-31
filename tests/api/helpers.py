"""Shared test gateway: real app + lifespan, fake upstreams via httpx.MockTransport."""

import json
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from tierwise.api.app import create_app
from tierwise.config import Settings
from tierwise.registry import ModelSpec, Registry

ANTHROPIC_KEY = "sk-ant-secret-123"
OPENAI_KEY = "sk-oa-secret-456"
GEMINI_KEY = "g-secret-789"
OLLAMA_URL = "http://ollama.test/v1"
Handler = Callable[[httpx.Request], httpx.Response]

MODELS = {
    "claude-haiku-5-5": "anthropic",
    "gpt-4.1-mini": "openai",
    "gemini-x": "gemini",
    "ollama/llama3.2": "ollama",
    "claude-opus-5-5": "anthropic",
}


def make_registry() -> Registry:
    def spec(model_id: str, provider: str) -> ModelSpec:
        return ModelSpec.model_validate(
            {
                "id": model_id,
                "provider": provider,
                "upstream_id": "llama3.2" if provider == "ollama" else None,
                "price_in": 1.0,
                "price_out": 2.0,
                "context": 1000,
                "caps": ["tools"],
                "quality": 0.5,
                "tier": "flagship" if "opus" in model_id else "basic",
            }
        )

    models = [spec(model_id, provider) for model_id, provider in MODELS.items()]
    return Registry(baseline_model="claude-opus-5-5", models=models)


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
GEMINI_REPLY = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "pong"}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
}


def ok_upstream(request: httpx.Request) -> httpx.Response:
    host = request.url.host
    if host == "api.anthropic.com":
        return httpx.Response(200, json=ANTHROPIC_REPLY)
    if host == "generativelanguage.googleapis.com":
        return httpx.Response(200, json=GEMINI_REPLY)
    return httpx.Response(200, json=OPENAI_REPLY)


def sse(*events: dict[str, Any], done: bool = True, raw_tail: bytes = b"") -> httpx.Response:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    if done:
        body += b"data: [DONE]\n\n"
    return httpx.Response(
        200, content=body + raw_tail, headers={"content-type": "text/event-stream"}
    )


def openai_chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "c",
        "model": "gpt-4.1-mini",
        "created": 1,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


@asynccontextmanager
async def gateway(
    handler: Handler = ok_upstream, db_dir: Path | None = None, **settings: Any
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    directory = db_dir or Path(tempfile.mkdtemp())
    values: dict[str, Any] = {
        "anthropic_api_key": ANTHROPIC_KEY,
        "openai_api_key": OPENAI_KEY,
        "gemini_api_key": GEMINI_KEY,
        "ollama_base_url": OLLAMA_URL,
        "db_url": f"sqlite+aiosqlite:///{(directory / 'log.db').as_posix()}",
        **settings,
    }
    config = Settings(_env_file=None, **values)  # type: ignore[call-arg]
    app = create_app(config, transport=httpx.MockTransport(handler), registry=make_registry())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
            yield app, client


def body(model: str, **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "ping"}], **extra}
