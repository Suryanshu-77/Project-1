"""HTTP endpoints. In M1 the model id picks the provider; routing arrives in M3."""

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tierwise.api.errors import GatewayError
from tierwise.providers.base import Provider
from tierwise.schema import ChatRequest

router = APIRouter()

_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude-",),
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
}


class UnknownModel(GatewayError):
    def __init__(self, model: str) -> None:
        super().__init__(404, "model_not_found", f"The model '{model}' is not available.")


def resolve_provider(model: str) -> str:
    for provider, prefixes in _PREFIXES.items():
        if model.startswith(prefixes):
            return provider
    raise UnknownModel(model)


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request) -> JSONResponse:
    if req.stream:
        raise GatewayError(400, "unsupported_parameter", "stream=true is not supported yet.")
    providers: dict[str, Provider] = request.app.state.providers
    provider = providers.get(resolve_provider(req.model))
    if provider is None:
        raise UnknownModel(req.model)
    response = await provider.chat(req, req.model)
    headers = {
        "X-Tierwise-Model": req.model,
        "X-Tierwise-Provider": provider.name,
        "X-Tierwise-Reason": "override",
        "X-Tierwise-Request-Id": uuid.uuid4().hex,
    }
    return JSONResponse(response.to_openai(), headers=headers)


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    return {"status": "ok", "providers": sorted(request.app.state.providers)}
