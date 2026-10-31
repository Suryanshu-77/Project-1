"""Provider-neutral request/response types.

Every adapter translates to and from these models, so no provider-specific
shape leaks past ``tierwise.providers``. The request side mirrors the OpenAI
Chat Completions format because that is the gateway's public API.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded, as in the OpenAI format


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDef(BaseModel):
    type: Literal["function"]
    function: FunctionDef


class ChatRequest(BaseModel):
    # Real SDK clients send fields we don't use (user, seed, n, ...); ignore them.
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    tools: list[ToolDef] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None


class Usage(BaseModel):
    """Token counts. ``input_tokens`` is the total prompt size; cache fields are subsets."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0


FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class ChatResponse(BaseModel):
    id: str
    model: str
    provider: str
    content: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: FinishReason
    usage: Usage
    created: int

    def to_openai(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
            "usage": {
                "prompt_tokens": self.usage.input_tokens,
                "completion_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.input_tokens + self.usage.output_tokens,
                "prompt_tokens_details": {"cached_tokens": self.usage.cached_input_tokens},
            },
        }
