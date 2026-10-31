import pytest
from pydantic import ValidationError

from tierwise.schema import (
    ChatRequest,
    ChatResponse,
    FunctionCall,
    ImagePart,
    StreamChunk,
    TextPart,
    ToolCall,
    ToolCallDelta,
    Usage,
    chunk_to_openai,
)


def _response(**overrides: object) -> ChatResponse:
    fields: dict[str, object] = {
        "id": "r1",
        "model": "m",
        "provider": "p",
        "content": "ok",
        "tool_calls": None,
        "finish_reason": "stop",
        "usage": Usage(input_tokens=10, output_tokens=5, cached_input_tokens=4),
        "created": 1,
    }
    fields.update(overrides)
    return ChatResponse.model_validate(fields)


def test_extra_fields_ignored() -> None:
    req = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "user": "u1",
            "seed": 3,
            "n": 1,
        }
    )
    assert req.model == "m"


def test_multipart_content_parses() -> None:
    req = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ],
        }
    )
    content = req.messages[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], TextPart)
    assert isinstance(content[1], ImagePart)


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"model": "m", "messages": []})


def test_to_openai_shape() -> None:
    out = _response().to_openai()
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "ok"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 4},
    }


def test_to_openai_tool_calls() -> None:
    call = ToolCall(id="c1", function=FunctionCall(name="f", arguments='{"q": "x"}'))
    out = _response(content=None, tool_calls=[call], finish_reason="tool_calls").to_openai()
    message = out["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"q": "x"}'}}
    ]


def test_chunk_to_openai_content() -> None:
    out = chunk_to_openai(StreamChunk(content="hi"), id="c1", model="m", created=1)
    assert out["object"] == "chat.completion.chunk"
    assert out["choices"][0]["delta"] == {"content": "hi"}
    assert out["choices"][0]["finish_reason"] is None
    assert "usage" not in out


def test_chunk_to_openai_tool_delta() -> None:
    first = StreamChunk(tool_calls=[ToolCallDelta(index=0, id="t1", name="f")])
    later = StreamChunk(tool_calls=[ToolCallDelta(index=0, arguments='{"q"')])
    first_delta = chunk_to_openai(first, id="c", model="m", created=1)["choices"][0]["delta"]
    later_delta = chunk_to_openai(later, id="c", model="m", created=1)["choices"][0]["delta"]
    assert first_delta["tool_calls"] == [
        {"index": 0, "id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}
    ]
    assert later_delta["tool_calls"] == [{"index": 0, "function": {"arguments": '{"q"'}}]


def test_chunk_to_openai_finish() -> None:
    out = chunk_to_openai(StreamChunk(finish_reason="stop"), id="c", model="m", created=1)
    assert out["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}


def test_chunk_to_openai_usage_only() -> None:
    usage = Usage(input_tokens=7, output_tokens=2, cached_input_tokens=3)
    out = chunk_to_openai(StreamChunk(usage=usage), id="c", model="m", created=1)
    assert out["choices"] == []
    assert out["usage"]["prompt_tokens"] == 7
    assert out["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}
