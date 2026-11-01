"""Cheap token estimates (characters / 4) until a real tokenizer arrives with the learned router."""

import json
import math

from tierwise.schema import ChatRequest, ContentPart, TextPart

DEFAULT_OUTPUT_TOKENS = 512


def text_of(content: str | list[ContentPart] | None) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


def estimate_text(text: str) -> int:
    return math.ceil(len(text) / 4)


def estimate_tokens(req: ChatRequest) -> int:
    """Rough prompt size: all message text plus the tool definitions."""
    chars = sum(len(text_of(m.content)) for m in req.messages)
    for m in req.messages:
        for call in m.tool_calls or []:
            chars += len(call.function.name) + len(call.function.arguments)
    if req.tools:
        chars += len(json.dumps([t.model_dump() for t in req.tools]))
    return math.ceil(chars / 4)


def estimate_output(req: ChatRequest) -> int:
    return req.max_completion_tokens or req.max_tokens or DEFAULT_OUTPUT_TOKENS
