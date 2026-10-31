import json
from typing import Any

import httpx
import pytest
import respx

from tierwise.providers.base import FatalError
from tierwise.providers.gemini import GeminiProvider, from_gemini, to_gemini
from tierwise.schema import ChatRequest, Usage

MODEL = "gemini-x"
BASE = "https://generativelanguage.googleapis.com/v1beta"


def _req(messages: list[dict[str, Any]], **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate({"model": MODEL, "messages": messages, **extra})


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _reply(parts: list[dict[str, Any]], finish: str = "STOP", **usage: int) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": usage or {"promptTokenCount": 3, "candidatesTokenCount": 1},
        "modelVersion": MODEL,
        "responseId": "r1",
    }


def test_system_and_roles() -> None:
    body = to_gemini(
        _req(
            [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "developer", "content": "B"},
            ]
        )
    )
    assert body["systemInstruction"] == {"parts": [{"text": "A\n\nB"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model"]
    assert body["contents"][0]["parts"] == [{"text": "q"}]


def test_data_url_image_inline() -> None:
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    body = to_gemini(_req([{"role": "user", "content": content}]))
    assert body["contents"][0]["parts"] == [
        {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}
    ]


def test_https_image_rejected_400() -> None:
    content = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
    with pytest.raises(FatalError) as info:
        to_gemini(_req([{"role": "user", "content": content}]))
    assert info.value.status == 400


def test_tool_round_trip_uses_name_lookup() -> None:
    body = to_gemini(
        _req(
            [
                {"role": "user", "content": "search x"},
                {"role": "assistant", "tool_calls": [_call("c1", "search", '{"q": "x"}')]},
                {"role": "tool", "tool_call_id": "c1", "content": "found"},
            ]
        )
    )
    assert body["contents"][1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "search", "args": {"q": "x"}}}],
    }
    assert body["contents"][2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "search", "response": {"content": "found"}}}],
    }


def test_unknown_tool_call_id_400() -> None:
    with pytest.raises(FatalError) as info:
        to_gemini(
            _req(
                [
                    {"role": "user", "content": "q"},
                    {"role": "tool", "tool_call_id": "nope", "content": "r"},
                ]
            )
        )
    assert info.value.status == 400


def test_bad_tool_arguments_400() -> None:
    with pytest.raises(FatalError):
        to_gemini(_req([{"role": "assistant", "tool_calls": [_call("c1", "f", "[1")]}]))


def test_generation_config_and_json_mode() -> None:
    body = to_gemini(
        _req(
            [{"role": "user", "content": "q"}],
            temperature=0.3,
            top_p=0.9,
            max_completion_tokens=50,
            stop="END",
            response_format={"type": "json_object"},
        )
    )
    assert body["generationConfig"] == {
        "temperature": 0.3,
        "topP": 0.9,
        "maxOutputTokens": 50,
        "stopSequences": ["END"],
        "responseMimeType": "application/json",
    }


def test_tool_choice_named_function() -> None:
    body = to_gemini(
        _req(
            [{"role": "user", "content": "q"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
    )
    assert body["tools"] == [{"functionDeclarations": [{"name": "f"}]}]
    assert body["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f"]}
    }


def test_from_gemini_text_usage_with_thoughts() -> None:
    resp = from_gemini(
        _reply(
            [{"text": "Hel"}, {"text": "lo"}],
            promptTokenCount=100,
            cachedContentTokenCount=40,
            candidatesTokenCount=10,
            thoughtsTokenCount=5,
        ),
        MODEL,
    )
    assert resp.content == "Hello"
    assert resp.finish_reason == "stop"
    assert resp.usage == Usage(input_tokens=100, output_tokens=15, cached_input_tokens=40)


def test_from_gemini_function_call_finish_tool_calls() -> None:
    resp = from_gemini(_reply([{"functionCall": {"name": "f", "args": {"q": "x"}}}]), MODEL)
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls is not None
    assert resp.tool_calls[0].id == "call_0"
    assert json.loads(resp.tool_calls[0].function.arguments) == {"q": "x"}


def test_from_gemini_blocked_prompt_content_filter() -> None:
    data = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 4}}
    resp = from_gemini(data, MODEL)
    assert resp.content is None
    assert resp.finish_reason == "content_filter"


@respx.mock
async def test_chat_url_and_key_header() -> None:
    route = respx.post(f"{BASE}/models/{MODEL}:generateContent").respond(
        200, json=_reply([{"text": "hi"}])
    )
    async with httpx.AsyncClient() as client:
        resp = await GeminiProvider(client, "g-key").chat(
            _req([{"role": "user", "content": "q"}]), MODEL
        )
    assert route.calls.last.request.headers["x-goog-api-key"] == "g-key"
    assert resp.provider == "gemini"
    assert resp.content == "hi"


@respx.mock
async def test_stream_text_then_finish_then_usage() -> None:
    events = [
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}],
            "usageMetadata": {"promptTokenCount": 5},
        },
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "lo"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        },
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    respx.post(f"{BASE}/models/{MODEL}:streamGenerateContent").respond(
        200, content=body, headers={"content-type": "text/event-stream"}
    )
    async with httpx.AsyncClient() as client:
        provider = GeminiProvider(client, "g-key")
        chunks = [c async for c in provider.stream(_req([{"role": "user", "content": "q"}]), MODEL)]
    assert [c.content for c in chunks if c.content] == ["Hel", "lo"]
    assert chunks[-2].finish_reason == "stop"
    assert chunks[-1].usage == Usage(input_tokens=5, output_tokens=2)


def test_parallel_tool_results_share_one_turn() -> None:
    calls = [_call("c1", "a", "{}"), _call("c2", "b", "{}")]
    body = to_gemini(
        _req(
            [
                {"role": "user", "content": "both"},
                {"role": "assistant", "tool_calls": calls},
                {"role": "tool", "tool_call_id": "c1", "content": "ra"},
                {"role": "tool", "tool_call_id": "c2", "content": "rb"},
            ]
        )
    )
    assert len(body["contents"]) == 3
    names = [p["functionResponse"]["name"] for p in body["contents"][2]["parts"]]
    assert names == ["a", "b"]
