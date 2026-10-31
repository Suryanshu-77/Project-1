"""OpenAI Chat Completions adapter; also serves any OpenAI-compatible host via ``base_url``."""

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, cast, get_args

import httpx

from tierwise.providers.base import (
    MAX_ERROR_CHARS,
    RetryableError,
    parse_or_retry,
    send,
    stream_sse,
)
from tierwise.schema import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)

_FINISH_REASONS: frozenset[str] = frozenset(get_args(FinishReason))
_REASONING_PREFIXES = ("o1", "o3", "o4")


def to_openai_request(req: ChatRequest, model_id: str) -> dict[str, Any]:
    body = req.model_dump(exclude_none=True, exclude={"stream", "stream_options"})
    body["model"] = model_id
    # Reasoning (o-series) models reject max_tokens; other OpenAI-compatible hosts may not
    # know max_completion_tokens, so only rename where it is required.
    if model_id.startswith(_REASONING_PREFIXES) and "max_tokens" in body:
        body.setdefault("max_completion_tokens", body.pop("max_tokens"))
    return body


def _finish(value: Any) -> FinishReason:
    return cast(FinishReason, value) if value in _FINISH_REASONS else "stop"


def _usage(usage: dict[str, Any]) -> Usage:
    details = usage.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
        cached_input_tokens=details.get("cached_tokens") or 0,
    )


def from_openai(data: dict[str, Any], provider: str = "openai") -> ChatResponse:
    choice = data["choices"][0]
    message = choice.get("message", {})
    calls = message.get("tool_calls")
    return ChatResponse(
        id=data["id"],
        model=data["model"],
        provider=provider,
        content=message.get("content"),
        tool_calls=[ToolCall.model_validate(call) for call in calls] if calls else None,
        finish_reason=_finish(choice.get("finish_reason") or "stop"),
        usage=_usage(data.get("usage") or {}),
        created=data.get("created") or int(time.time()),
    )


def chunks_from_event(event: dict[str, Any], provider: str = "openai") -> list[StreamChunk]:
    """Normalize one ``chat.completion.chunk`` (some hosts put usage on the last choice chunk)."""
    if event.get("error"):
        error = event["error"]
        message = error.get("message", "") if isinstance(error, dict) else str(error)
        raise RetryableError(provider, None, str(message)[:MAX_ERROR_CHARS])
    chunks: list[StreamChunk] = []
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        tool_calls = [
            ToolCallDelta(
                index=call["index"],
                id=call.get("id"),
                name=(call.get("function") or {}).get("name"),
                arguments=(call.get("function") or {}).get("arguments") or "",
            )
            for call in delta.get("tool_calls") or []
        ]
        if delta.get("content") or tool_calls:
            chunks.append(
                StreamChunk(content=delta.get("content") or None, tool_calls=tool_calls or None)
            )
        if choice.get("finish_reason"):
            chunks.append(StreamChunk(finish_reason=_finish(choice["finish_reason"])))
    if event.get("usage"):
        chunks.append(StreamChunk(usage=_usage(event["usage"])))
    return chunks


class OpenAIProvider:
    name = "openai"
    include_usage = True  # ask for a final usage chunk (stream_options)

    def __init__(
        self, client: httpx.AsyncClient, api_key: str, base_url: str = "https://api.openai.com/v1"
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def chat(self, req: ChatRequest, model_id: str) -> ChatResponse:
        data = await send(
            self._client,
            self.name,
            "POST",
            f"{self._base_url}/chat/completions",
            json=to_openai_request(req, model_id),
            headers=self._headers(),
        )
        return parse_or_retry(self.name, data, lambda d: from_openai(d, provider=self.name))

    async def stream(self, req: ChatRequest, model_id: str) -> AsyncGenerator[StreamChunk, None]:
        body = to_openai_request(req, model_id) | {"stream": True}
        if self.include_usage:
            body["stream_options"] = {"include_usage": True}
        events = stream_sse(
            self._client,
            self.name,
            "POST",
            f"{self._base_url}/chat/completions",
            json=body,
            headers=self._headers(),
        )
        async with aclosing(events):
            async for event in events:
                for chunk in parse_or_retry(
                    self.name, event, lambda e: chunks_from_event(e, self.name)
                ):
                    yield chunk

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
