"""Anthropic Messages API adapter."""

import json
import time
from typing import Any

import httpx

from tierwise.providers.base import send
from tierwise.schema import (
    ChatRequest,
    ChatResponse,
    ContentPart,
    FinishReason,
    FunctionCall,
    Message,
    TextPart,
    ToolCall,
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


def _convert_message(msg: Message) -> dict[str, Any]:
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
                "input": json.loads(call.function.arguments or "{}"),
            }
            for call in msg.tool_calls
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
    for msg in req.messages:
        if msg.role in ("system", "developer"):
            system_parts.append(_text_of(msg.content))
            continue
        converted = _convert_message(msg)
        # Parallel tool results must share one user turn.
        if msg.role == "tool" and messages and _is_tool_results(messages[-1]):
            messages[-1]["content"].extend(converted["content"])
        else:
            messages.append(converted)

    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": req.max_tokens or DEFAULT_MAX_TOKENS,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop is not None:
        body["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.tools:
        body["tools"] = [
            {
                "name": tool.function.name,
                **({"description": tool.function.description} if tool.function.description else {}),
                "input_schema": tool.function.parameters,
            }
            for tool in req.tools
        ]
    return body


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
    usage = data.get("usage", {})
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    return ChatResponse(
        id=data["id"],
        model=data["model"],
        provider="anthropic",
        content="".join(texts) if texts else None,
        tool_calls=calls or None,
        finish_reason=_STOP_REASONS.get(data.get("stop_reason") or "", "stop"),
        usage=Usage(
            input_tokens=(usage.get("input_tokens") or 0) + cache_read + cache_write,
            output_tokens=usage.get("output_tokens") or 0,
            cached_input_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
        created=int(time.time()),
    )


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
            headers={"x-api-key": self._api_key, "anthropic-version": API_VERSION},
        )
        return from_anthropic(data)
