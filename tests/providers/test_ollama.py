import json

import httpx
import respx

from tierwise.providers.ollama import OllamaProvider
from tierwise.schema import ChatRequest

BASE = "http://localhost:11434/v1"
URL = f"{BASE}/chat/completions"
REPLY = {
    "id": "c",
    "model": "llama3.2",
    "created": 1,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


def _req() -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "x", "messages": [{"role": "user", "content": "q"}]}
    )


@respx.mock
async def test_ollama_no_auth_header_and_provider_name() -> None:
    route = respx.post(URL).respond(200, json=REPLY)
    async with httpx.AsyncClient() as client:
        resp = await OllamaProvider(client, BASE).chat(_req(), "llama3.2")
    assert "authorization" not in route.calls.last.request.headers
    assert resp.provider == "ollama"


@respx.mock
async def test_ollama_stream_without_usage_completes() -> None:
    chunk = {
        "id": "c",
        "model": "llama3.2",
        "created": 1,
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
    }
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
    route = respx.post(URL).respond(
        200, content=body, headers={"content-type": "text/event-stream"}
    )
    async with httpx.AsyncClient() as client:
        chunks = [c async for c in OllamaProvider(client, BASE).stream(_req(), "llama3.2")]
    assert [c.content for c in chunks if c.content] == ["hi"]
    assert all(c.usage is None for c in chunks)
    assert "stream_options" not in json.loads(route.calls.last.request.content)
