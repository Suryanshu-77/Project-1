"""Google Gemini adapter for the native ``generateContent`` API."""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import httpx

from tierwise.providers.base import (
    FatalError,
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
    StreamChunk,
    TextPart,
    ToolCall,
    ToolCallDelta,
    Usage,
)

_FILTERED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


def _parts(content: str | list[ContentPart] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"text": part.text})
            continue
        url = part.image_url.url
        if not url.startswith("data:"):
            raise FatalError("gemini", 400, "Gemini accepts only data: image URLs")
        header, _, data = url.partition(",")
        mime_type = header.removeprefix("data:").split(";")[0]
        parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
    return parts


def _text_of(content: str | list[ContentPart] | None) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    return "".join(part.text for part in content if isinstance(part, TextPart))


def _tool_config(choice: Any) -> dict[str, Any] | None:
    modes = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
    if isinstance(choice, str) and choice in modes:
        return {"functionCallingConfig": {"mode": modes[choice]}}
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        name = choice["function"]["name"]
        return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None


def _generation_config(req: ChatRequest) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if req.temperature is not None:
        config["temperature"] = req.temperature
    if req.top_p is not None:
        config["topP"] = req.top_p
    max_tokens = req.max_completion_tokens or req.max_tokens
    if max_tokens:
        config["maxOutputTokens"] = max_tokens
    if req.stop is not None:
        config["stopSequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.response_format and req.response_format.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    return config


def to_gemini(req: ChatRequest) -> dict[str, Any]:
    system: list[str] = []
    contents: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}  # tool_call_id -> function name, for functionResponse
    for index, msg in enumerate(req.messages):
        if msg.role in ("system", "developer"):
            system.append(_text_of(msg.content))
        elif msg.role == "tool":
            name = tool_names.get(msg.tool_call_id or "")
            if name is None:
                message = f"messages[{index}].tool_call_id does not match an earlier tool call"
                raise FatalError("gemini", 400, message)
            response = {"name": name, "response": {"content": _text_of(msg.content)}}
            contents.append({"role": "user", "parts": [{"functionResponse": response}]})
        elif msg.role == "assistant":
            parts = _parts(msg.content)
            for position, call in enumerate(msg.tool_calls or []):
                tool_names[call.id] = call.function.name
                where = f"messages[{index}].tool_calls[{position}]"
                args = parse_tool_arguments("gemini", call.function.arguments, where)
                parts.append({"functionCall": {"name": call.function.name, "args": args}})
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({"role": "user", "parts": _parts(msg.content)})

    body: dict[str, Any] = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
    config = _generation_config(req)
    if config:
        body["generationConfig"] = config
    if req.tools:
        declarations = []
        for tool in req.tools:
            declaration: dict[str, Any] = {"name": tool.function.name}
            if tool.function.description:
                declaration["description"] = tool.function.description
            if tool.function.parameters:
                declaration["parameters"] = tool.function.parameters
            declarations.append(declaration)
        body["tools"] = [{"functionDeclarations": declarations}]
        tool_config = _tool_config(req.tool_choice)
        if tool_config is not None:
            body["toolConfig"] = tool_config
    return body


def _finish(reason: str | None, has_tool_calls: bool) -> FinishReason:
    if reason == "MAX_TOKENS":
        return "length"
    if reason in _FILTERED:
        return "content_filter"
    return "tool_calls" if has_tool_calls else "stop"


def _usage(meta: dict[str, Any]) -> Usage:
    # Gemini bills "thinking" tokens as output.
    return Usage(
        input_tokens=meta.get("promptTokenCount") or 0,
        output_tokens=(meta.get("candidatesTokenCount") or 0)
        + (meta.get("thoughtsTokenCount") or 0),
        cached_input_tokens=meta.get("cachedContentTokenCount") or 0,
    )


def _candidate_parts(data: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str | None]:
    candidates = data.get("candidates") or []
    if not candidates:
        return [], [], None
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
    calls = [p["functionCall"] for p in parts if "functionCall" in p]
    return texts, calls, candidate.get("finishReason")


def from_gemini(data: dict[str, Any], model: str) -> ChatResponse:
    texts, calls, reason = _candidate_parts(data)
    tool_calls = [
        ToolCall(
            id=call.get("id") or f"call_{i}",
            function=FunctionCall(name=call["name"], arguments=json.dumps(call.get("args") or {})),
        )
        for i, call in enumerate(calls)
    ]
    blocked = not data.get("candidates")
    return ChatResponse(
        id=data.get("responseId") or f"gemini-{uuid.uuid4().hex}",
        model=data.get("modelVersion") or model,
        provider="gemini",
        content="".join(texts) if texts else None,
        tool_calls=tool_calls or None,
        finish_reason="content_filter" if blocked else _finish(reason, bool(tool_calls)),
        usage=_usage(data.get("usageMetadata") or {}),
        created=int(time.time()),
    )


class _StreamState:
    """Each SSE event is a full GenerateContentResponse holding the next slice of output."""

    def __init__(self) -> None:
        self.tool_count = 0
        self.usage: dict[str, Any] | None = None

    def handle(self, data: dict[str, Any]) -> list[StreamChunk]:
        if data.get("usageMetadata"):
            self.usage = data["usageMetadata"]
        texts, calls, reason = _candidate_parts(data)
        chunks: list[StreamChunk] = []
        if texts:
            chunks.append(StreamChunk(content="".join(texts)))
        for call in calls:  # Gemini streams each function call whole
            delta = ToolCallDelta(
                index=self.tool_count,
                id=call.get("id") or f"call_{self.tool_count}",
                name=call["name"],
                arguments=json.dumps(call.get("args") or {}),
            )
            self.tool_count += 1
            chunks.append(StreamChunk(tool_calls=[delta]))
        if reason:
            chunks.append(StreamChunk(finish_reason=_finish(reason, self.tool_count > 0)))
        return chunks


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def chat(self, req: ChatRequest, model_id: str) -> ChatResponse:
        data = await send(
            self._client,
            self.name,
            "POST",
            f"{self._base_url}/models/{model_id}:generateContent",
            json=to_gemini(req),
            headers=self._headers(),
        )
        return parse_or_retry(self.name, data, lambda d: from_gemini(d, model_id))

    async def stream(self, req: ChatRequest, model_id: str) -> AsyncGenerator[StreamChunk, None]:
        state = _StreamState()
        events = stream_sse(
            self._client,
            self.name,
            "POST",
            f"{self._base_url}/models/{model_id}:streamGenerateContent",
            params={"alt": "sse"},
            json=to_gemini(req),
            headers=self._headers(),
        )
        async with aclosing(events):
            async for event in events:
                for chunk in parse_or_retry(self.name, event, state.handle):
                    yield chunk
        if state.usage is not None:
            yield StreamChunk(usage=_usage(state.usage))

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}
