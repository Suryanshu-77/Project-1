"""OpenAI Chat Completions adapter; also serves any OpenAI-compatible host via ``base_url``."""

import time
from typing import Any, cast, get_args

import httpx

from tierwise.providers.base import parse_or_retry, send
from tierwise.schema import ChatRequest, ChatResponse, FinishReason, ToolCall, Usage

_FINISH_REASONS: frozenset[str] = frozenset(get_args(FinishReason))
_REASONING_PREFIXES = ("o1", "o3", "o4")


def to_openai_request(req: ChatRequest, model_id: str) -> dict[str, Any]:
    body = req.model_dump(exclude_none=True, exclude={"stream"})
    body["model"] = model_id
    # Reasoning (o-series) models reject max_tokens; other OpenAI-compatible hosts may not
    # know max_completion_tokens, so only rename where it is required.
    if model_id.startswith(_REASONING_PREFIXES) and "max_tokens" in body:
        body.setdefault("max_completion_tokens", body.pop("max_tokens"))
    return body


def from_openai(data: dict[str, Any]) -> ChatResponse:
    choice = data["choices"][0]
    message = choice.get("message", {})
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    finish = choice.get("finish_reason") or "stop"
    calls = message.get("tool_calls")
    return ChatResponse(
        id=data["id"],
        model=data["model"],
        provider="openai",
        content=message.get("content"),
        tool_calls=[ToolCall.model_validate(call) for call in calls] if calls else None,
        finish_reason=cast(FinishReason, finish) if finish in _FINISH_REASONS else "stop",
        usage=Usage(
            input_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
            cached_input_tokens=details.get("cached_tokens") or 0,
        ),
        created=data.get("created") or int(time.time()),
    )


class OpenAIProvider:
    name = "openai"

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
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        return parse_or_retry(self.name, data, from_openai)
