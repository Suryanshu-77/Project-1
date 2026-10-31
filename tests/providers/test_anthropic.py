import json
from typing import Any

import httpx
import pytest
import respx

from tierwise.providers.anthropic import AnthropicProvider, from_anthropic, to_anthropic
from tierwise.providers.base import FatalError, RetryableError
from tierwise.schema import ChatRequest, StreamChunk, Usage

MODEL = "claude-haiku-5-5"


def _req(messages: list[dict[str, Any]], **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate({"model": MODEL, "messages": messages, **extra})


def _reply(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "msg_1",
        "model": MODEL,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    data.update(overrides)
    return data


def test_system_messages_joined() -> None:
    body = to_anthropic(
        _req(
            [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "q"},
                {"role": "system", "content": "B"},
            ]
        ),
        MODEL,
    )
    assert body["system"] == "A\n\nB"
    assert all(m["role"] != "system" for m in body["messages"])


def test_default_max_tokens() -> None:
    assert to_anthropic(_req([{"role": "user", "content": "q"}]), MODEL)["max_tokens"] == 4096


def test_tool_round_trip_request() -> None:
    body = to_anthropic(
        _req(
            [
                {"role": "user", "content": "search x"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
            tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
        ),
        MODEL,
    )
    assistant, tool_reply = body["messages"][1], body["messages"][2]
    assert assistant["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}}
    ]
    assert tool_reply["role"] == "user"
    assert tool_reply["content"][0]["type"] == "tool_result"
    assert tool_reply["content"][0]["tool_use_id"] == "call_1"
    assert body["tools"] == [
        {"name": "search", "input_schema": {"type": "object", "properties": {}}}
    ]


def _tool_history(arguments: str) -> ChatRequest:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}
    return _req(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
        ]
    )


@pytest.mark.parametrize("arguments", ['{"q": "x"', "[]", "null"])
def test_bad_tool_arguments_are_fatal_400(arguments: str) -> None:
    with pytest.raises(FatalError) as info:
        to_anthropic(_tool_history(arguments), MODEL)
    assert info.value.status == 400
    assert "messages[1]" in info.value.message


def test_no_parameter_tool_gets_object_schema() -> None:
    body = to_anthropic(
        _req(
            [{"role": "user", "content": "time?"}],
            tools=[{"type": "function", "function": {"name": "get_time"}}],
        ),
        MODEL,
    )
    assert body["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        ({"type": "function", "function": {"name": "f"}}, {"type": "tool", "name": "f"}),
    ],
)
def test_tool_choice_mapped(choice: Any, expected: dict[str, Any]) -> None:
    body = to_anthropic(
        _req(
            [{"role": "user", "content": "q"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice=choice,
        ),
        MODEL,
    )
    assert body["tool_choice"] == expected


def test_max_completion_tokens_used() -> None:
    req = _req([{"role": "user", "content": "q"}], max_completion_tokens=200)
    assert to_anthropic(req, MODEL)["max_tokens"] == 200


def test_image_data_url_and_https() -> None:
    body = to_anthropic(
        _req(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                    ],
                }
            ]
        ),
        MODEL,
    )
    blocks = body["messages"][0]["content"]
    assert blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert blocks[1]["source"] == {"type": "url", "url": "https://x.test/a.png"}


def test_from_anthropic_text_and_usage() -> None:
    resp = from_anthropic(
        _reply(
            usage={
                "input_tokens": 10,
                "cache_read_input_tokens": 90,
                "cache_creation_input_tokens": 0,
                "output_tokens": 7,
            }
        )
    )
    assert resp.content == "hi"
    assert resp.finish_reason == "stop"
    assert resp.usage == Usage(
        input_tokens=100, output_tokens=7, cached_input_tokens=90, cache_write_tokens=0
    )


def test_from_anthropic_tool_use() -> None:
    resp = from_anthropic(
        _reply(
            content=[{"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}}],
            stop_reason="tool_use",
        )
    )
    assert resp.finish_reason == "tool_calls"
    assert resp.content is None
    assert resp.tool_calls is not None
    assert json.loads(resp.tool_calls[0].function.arguments) == {"q": "x"}


@respx.mock
async def test_chat_sends_headers() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").respond(200, json=_reply())
    async with httpx.AsyncClient() as client:
        resp = await AnthropicProvider(client, "sk-test").chat(
            _req([{"role": "user", "content": "q"}]), MODEL
        )
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "sk-test"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert resp.provider == "anthropic"


MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def _events(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _start(**usage: int) -> dict[str, Any]:
    return {"type": "message_start", "message": {"id": "msg_1", "usage": usage}}


async def _stream(body: bytes) -> list[StreamChunk]:
    respx.post(MESSAGES_URL).respond(
        200, content=body, headers={"content-type": "text/event-stream"}
    )
    async with httpx.AsyncClient() as client:
        provider = AnthropicProvider(client, "sk")
        return [c async for c in provider.stream(_req([{"role": "user", "content": "q"}]), MODEL)]


@respx.mock
async def test_stream_text_and_usage() -> None:
    chunks = await _stream(
        _events(
            _start(input_tokens=10, cache_read_input_tokens=90, output_tokens=1),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {"type": "ping"},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 7},
            },
            {"type": "message_stop"},
        )
    )
    assert [c.content for c in chunks if c.content] == ["Hel", "lo"]
    assert chunks[-2].finish_reason == "stop"
    assert chunks[-1].usage == Usage(input_tokens=100, output_tokens=7, cached_input_tokens=90)


@respx.mock
async def test_stream_tool_use_blocks() -> None:
    tool = {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}
    chunks = await _stream(
        _events(
            _start(input_tokens=5, output_tokens=1),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
            {"type": "content_block_start", "index": 1, "content_block": tool},
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"q": '},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '"x"}'},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 9},
            },
            {"type": "message_stop"},
        )
    )
    deltas = [d for c in chunks if c.tool_calls for d in c.tool_calls]
    assert all(d.index == 0 for d in deltas)
    assert (deltas[0].id, deltas[0].name) == ("toolu_1", "search")
    assert json.loads("".join(d.arguments for d in deltas)) == {"q": "x"}
    assert any(c.finish_reason == "tool_calls" for c in chunks)


@respx.mock
async def test_stream_overloaded_error_event_is_retryable() -> None:
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    with pytest.raises(RetryableError):
        await _stream(_events(_start(input_tokens=1, output_tokens=1), error))


@respx.mock
async def test_stream_other_error_event_is_fatal() -> None:
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    with pytest.raises(FatalError):
        await _stream(_events(_start(input_tokens=1, output_tokens=1), error))
