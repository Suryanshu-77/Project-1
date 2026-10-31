import json
from typing import Any

import httpx
import respx

from tierwise.providers.anthropic import AnthropicProvider, from_anthropic, to_anthropic
from tierwise.schema import ChatRequest, Usage

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
    assert body["tools"] == [{"name": "search", "input_schema": {}}]


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
