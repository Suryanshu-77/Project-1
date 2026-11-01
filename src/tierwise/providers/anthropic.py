"""Anthropic Messages API adapter."""

import json
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import httpx

from tierwise.providers.base import (
    FatalError,
    RetryableError,
    parse_or_retry,
    parse_tool_arguments,
    send,
    stream_sse,
)
from tierwise.schema import (
    ChatRequest,
    ChatResponse,
    ContentPart,
    FinishReason,
    FunctionCall,
    Message,
    StreamChunk,
    TextPart,
    ToolCall,
    ToolCallDelta,
    Usage,
)

API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _image_block(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header.removeprefix("data:").split(";")[0]
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(content: str | list[ContentPart] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [
        {"type": "text", "text": part.text}
        if isinstance(part, TextPart)
        else _image_block(part.image_url.url)
        for part in content
    ]


def _text_of(content: str | list[ContentPart] | None) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    return "".join(part.text for part in content if isinstance(part, TextPart))


def _input_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    # Anthropic requires an object schema; OpenAI allows tools with no parameters at all.
    return {"type": "object", "properties": {}, **parameters}


def _tool_choice(choice: Any) -> dict[str, Any] | None:
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        return {"type": "tool", "name": choice["function"]["name"]}
    return None


def _convert_message(msg: Message, index: int) -> dict[str, Any]:
    if msg.role == "tool":
        result = {
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id,
            "content": _text_of(msg.content),
        }
        return {"role": "user", "content": [result]}
    if msg.role == "assistant" and msg.tool_calls:
        blocks = _content_blocks(msg.content)
        blocks += [
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": parse_tool_arguments(
                    "anthropic",
                    call.function.arguments,
                    f"messages[{index}].tool_calls[{position}]",
                ),
            }
            for position, call in enumerate(msg.tool_calls)
        ]
        return {"role": "assistant", "content": blocks}
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}
    return {"role": msg.role, "content": _content_blocks(msg.content)}


def _is_tool_results(message: dict[str, Any]) -> bool:
    content = message["content"]
    return (
        message["role"] == "user"
        and isinstance(content, list)
        and all(block.get("type") == "tool_result" for block in content)
    )


def to_anthropic(req: ChatRequest, model_id: str) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for index, msg in enumerate(req.messages):
        if msg.role in ("system", "developer"):
            system_parts.append(_text_of(msg.content))
            continue
        converted = _convert_message(msg, index)
        # Parallel tool results must share one user turn.
        if msg.role == "tool" and messages and _is_tool_results(messages[-1]):
            messages[-1]["content"].extend(converted["content"])
        else:
            messages.append(converted)

    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop is not None:
        body["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.reasoning_effort is not None:
        body["output_config"] = {"effort": req.reasoning_effort}
    if req.tools:
        body["tools"] = [
            {
                "name": tool.function.name,
                **({"description": tool.function.description} if tool.function.description else {}),
                "input_schema": _input_schema(tool.function.parameters),
            }
            for tool in req.tools
        ]
        tool_choice = _tool_choice(req.tool_choice)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    return body


def _usage(usage: dict[str, Any]) -> Usage:
    # Anthropic's input_tokens excludes cached reads and writes; ours is the whole prompt.
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    return Usage(
        input_tokens=(usage.get("input_tokens") or 0) + cache_read + cache_write,
        output_tokens=usage.get("output_tokens") or 0,
        cached_input_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def from_anthropic(data: dict[str, Any]) -> ChatResponse:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            texts.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    id=block["id"],
                    function=FunctionCall(name=block["name"], arguments=json.dumps(block["input"])),
                )
            )
    return ChatResponse(
        id=data["id"],
        model=data["model"],
        provider="anthropic",
        content="".join(texts) if texts else None,
        tool_calls=calls or None,
        finish_reason=_STOP_REASONS.get(data.get("stop_reason") or "", "stop"),
        usage=_usage(data.get("usage") or {}),
        created=int(time.time()),
    )


class _StreamState:
    """Turns Messages API stream events into normalized chunks."""

    def __init__(self) -> None:
        self.usage: dict[str, Any] = {}
        self.finish: FinishReason = "stop"
        self.tool_index: dict[int, int] = {}  # Anthropic block index -> tool ordinal
        self.tool_has_args: set[int] = set()  # block indexes that streamed any arguments

    def handle(self, event: dict[str, Any]) -> list[StreamChunk]:
        kind = event.get("type")
        if kind == "message_start":
            self.usage.update(event["message"].get("usage") or {})
        elif kind == "content_block_start" and event["content_block"]["type"] == "tool_use":
            block = event["content_block"]
            ordinal = self.tool_index[event["index"]] = len(self.tool_index)
            delta = ToolCallDelta(index=ordinal, id=block["id"], name=block["name"])
            return [StreamChunk(tool_calls=[delta])]
        elif kind == "content_block_delta":
            delta = event["delta"]
            if delta["type"] == "text_delta":
                return [StreamChunk(content=delta["text"])]
            if delta["type"] == "input_json_delta":
                ordinal = self.tool_index[event["index"]]
                if delta["partial_json"]:
                    self.tool_has_args.add(event["index"])
                fragment = ToolCallDelta(index=ordinal, arguments=delta["partial_json"])
                return [StreamChunk(tool_calls=[fragment])]
        elif kind == "content_block_stop":
            index = event["index"]
            if index in self.tool_index and index not in self.tool_has_args:
                # A tool called with no arguments streams no JSON; clients expect "{}".
                empty = ToolCallDelta(index=self.tool_index[index], arguments="{}")
                return [StreamChunk(tool_calls=[empty])]
        elif kind == "message_delta":
            self.finish = _STOP_REASONS.get(event["delta"].get("stop_reason") or "", "stop")
            self.usage.update(event.get("usage") or {})
        elif kind == "message_stop":
            return [StreamChunk(finish_reason=self.finish), StreamChunk(usage=_usage(self.usage))]
        elif kind == "error":
            error = event.get("error") or {}
            message = str(error.get("message", ""))
            if error.get("type") == "overloaded_error":
                raise RetryableError("anthropic", None, message)
            raise FatalError("anthropic", None, message)
        return []  # ping, content_block_stop, text block starts


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self, client: httpx.AsyncClient, api_key: str, base_url: str = "https://api.anthropic.com"
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def chat(self, req: ChatRequest, model_id: str) -> ChatResponse:
        data = await send(
            self._client,
            self.name,
            "POST",
            f"{self._base_url}/v1/messages",
            json=to_anthropic(req, model_id),
            headers=self._headers(),
        )
        return parse_or_retry(self.name, data, from_anthropic)

    async def stream(self, req: ChatRequest, model_id: str) -> AsyncGenerator[StreamChunk, None]:
        state = _StreamState()
        events = stream_sse(
            self._client,
            self.name,
            "POST",
            f"{self._base_url}/v1/messages",
            json=to_anthropic(req, model_id) | {"stream": True},
            headers=self._headers(),
        )
        async with aclosing(events):
            async for event in events:
                for chunk in parse_or_retry(self.name, event, state.handle):
                    yield chunk

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": API_VERSION}
