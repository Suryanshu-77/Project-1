from typing import Any

import httpx
import respx

from tierwise.providers.openai import OpenAIProvider, from_openai, to_openai_request
from tierwise.schema import ChatRequest, Usage

MODEL = "gpt-4.1-mini"


def _req(**extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "ignored", "messages": [{"role": "user", "content": "q"}], **extra}
    )


def _reply(message: dict[str, Any], finish: str = "stop", **usage: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": MODEL,
        "created": 5,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 9, **usage},
    }


def test_request_passthrough_drops_stream_and_none() -> None:
    body = to_openai_request(_req(temperature=0.2), MODEL)
    assert body["model"] == MODEL
    assert body["temperature"] == 0.2
    assert "stream" not in body
    assert "top_p" not in body
    assert body["messages"] == [{"role": "user", "content": "q"}]


def test_from_openai_cached_usage() -> None:
    resp = from_openai(
        _reply({"role": "assistant", "content": "a"}, prompt_tokens_details={"cached_tokens": 60})
    )
    assert resp.content == "a"
    assert resp.provider == "openai"
    assert resp.usage == Usage(input_tokens=100, output_tokens=9, cached_input_tokens=60)


def test_from_openai_tool_calls() -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    resp = from_openai(
        _reply({"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls")
    )
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].function.name == "f"


def test_unknown_finish_reason_becomes_stop() -> None:
    resp = from_openai(_reply({"role": "assistant", "content": "a"}, "function_call"))
    assert resp.finish_reason == "stop"


@respx.mock
async def test_custom_base_url_used() -> None:
    route = respx.post("https://compat.test/v1/chat/completions").respond(
        200, json=_reply({"role": "assistant", "content": "a"})
    )
    async with httpx.AsyncClient() as client:
        provider = OpenAIProvider(client, "sk-o", base_url="https://compat.test/v1")
        resp = await provider.chat(_req(), MODEL)
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-o"
    assert resp.content == "a"
