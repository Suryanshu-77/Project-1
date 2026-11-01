import json
from typing import Any

import httpx
import pytest
import respx

from tierwise.providers.base import RetryableError
from tierwise.providers.openai import OpenAIProvider, from_openai, to_openai_request
from tierwise.schema import ChatRequest, StreamChunk, Usage

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


def test_max_completion_tokens_forwarded() -> None:
    body = to_openai_request(_req(max_completion_tokens=200), MODEL)
    assert body["max_completion_tokens"] == 200


def test_o_series_max_tokens_renamed() -> None:
    body = to_openai_request(_req(max_tokens=50), "o3-mini")
    assert body["max_completion_tokens"] == 50
    assert "max_tokens" not in body


def test_gpt_max_tokens_kept() -> None:
    assert to_openai_request(_req(max_tokens=50), MODEL)["max_tokens"] == 50


@respx.mock
async def test_malformed_2xx_body_is_retryable() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").respond(200, json={"choices": []})
    async with httpx.AsyncClient() as client:
        with pytest.raises(RetryableError):
            await OpenAIProvider(client, "sk-o").chat(_req(), MODEL)


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


SSE = {"content-type": "text/event-stream"}
COMPLETIONS = "https://api.openai.com/v1/chat/completions"


def _sse(*events: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "model": MODEL,
        "created": 1,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def _drain(provider: OpenAIProvider) -> list[StreamChunk]:
    return [chunk async for chunk in provider.stream(_req(), MODEL)]


@respx.mock
async def test_stream_text_then_usage() -> None:
    usage = {
        "id": "c",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }
    body = _sse(
        _chunk({"role": "assistant", "content": "Hel"}),
        _chunk({"content": "lo"}),
        _chunk({}, "stop"),
        usage,
    )
    respx.post(COMPLETIONS).respond(200, content=body, headers=SSE)
    async with httpx.AsyncClient() as client:
        chunks = await _drain(OpenAIProvider(client, "sk"))
    assert [c.content for c in chunks if c.content] == ["Hel", "lo"]
    assert [c.finish_reason for c in chunks if c.finish_reason] == ["stop"]
    assert chunks[-1].usage == Usage(input_tokens=10, output_tokens=2, cached_input_tokens=4)


@respx.mock
async def test_stream_request_sets_include_usage() -> None:
    route = respx.post(COMPLETIONS).respond(200, content=_sse(), headers=SSE)
    async with httpx.AsyncClient() as client:
        await _drain(OpenAIProvider(client, "sk"))
    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


@respx.mock
async def test_stream_tool_call_fragments() -> None:
    first = {"index": 0, "id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}
    body = _sse(
        _chunk({"tool_calls": [first]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q": '}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}),
        _chunk({}, "tool_calls"),
    )
    respx.post(COMPLETIONS).respond(200, content=body, headers=SSE)
    async with httpx.AsyncClient() as client:
        chunks = await _drain(OpenAIProvider(client, "sk"))
    deltas = [d for c in chunks if c.tool_calls for d in c.tool_calls]
    assert (deltas[0].id, deltas[0].name) == ("t1", "f")
    assert json.loads("".join(d.arguments for d in deltas)) == {"q": "x"}
    assert chunks[-1].finish_reason == "tool_calls"


def test_stream_options_not_sent_on_plain_requests() -> None:
    body = to_openai_request(_req(stream_options={"include_usage": True}), MODEL)
    assert "stream_options" not in body


def test_openai_effort_forwarded() -> None:
    body = to_openai_request(_req(reasoning_effort="minimal"), MODEL)
    assert body["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in to_openai_request(_req(), MODEL)


def test_openai_usage_reasoning_tokens() -> None:
    resp = from_openai(
        _reply(
            {"role": "assistant", "content": "a"},
            completion_tokens_details={"reasoning_tokens": 7},
        )
    )
    assert resp.usage.reasoning_tokens == 7
    assert from_openai(_reply({"role": "assistant", "content": "a"})).usage.reasoning_tokens is None


@respx.mock
async def test_openai_stream_usage_reasoning_tokens() -> None:
    usage = {
        "id": "c",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 9,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }
    respx.post(COMPLETIONS).respond(
        200, content=_sse(_chunk({"content": "a"}, "stop"), usage), headers=SSE
    )
    async with httpx.AsyncClient() as client:
        chunks = await _drain(OpenAIProvider(client, "sk"))
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.reasoning_tokens == 4
