"""HTTP endpoints. The registry decides which provider serves a model; routing arrives in M3."""

import logging
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from tierwise.api.errors import GatewayError, gateway_error_from
from tierwise.api.streaming import sse_response
from tierwise.costs import cost_usd
from tierwise.providers.base import Provider, ProviderError
from tierwise.registry import ModelSpec, Registry
from tierwise.schema import ChatRequest, Usage
from tierwise.store import RequestLog, RequestRecord

router = APIRouter()
logger = logging.getLogger(__name__)


class UnknownModel(GatewayError):
    def __init__(self, model: str) -> None:
        super().__init__(404, "model_not_found", f"The model '{model}' is not available.")


def resolve(
    model: str, registry: Registry, providers: dict[str, Provider]
) -> tuple[ModelSpec, Provider]:
    spec = registry.get(model)
    provider = providers.get(spec.provider) if spec is not None else None
    if spec is None or provider is None:
        raise UnknownModel(model)
    return spec, provider


class _Recorder:
    """Builds the one log record each request produces, whatever its outcome."""

    def __init__(self, log: RequestLog, registry: Registry, model: str, stream: bool) -> None:
        self.log = log
        self.registry = registry
        self.model = model
        self.stream = stream
        self.request_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.spec: ModelSpec | None = None

    def costs(self, usage: Usage | None) -> tuple[float | None, float | None]:
        if usage is None or self.spec is None:
            return None, None
        return cost_usd(usage, self.spec), cost_usd(usage, self.registry.baseline())

    def done(self, usage: Usage | None, status: int, error_code: str | None) -> None:
        cost, baseline = self.costs(usage)
        self.log.submit(
            RequestRecord(
                id=self.request_id,
                ts=time.time(),
                model=self.model,
                provider=self.spec.provider if self.spec else "",
                status=status,
                stream=self.stream,
                latency_ms=(time.perf_counter() - self.started) * 1000,
                tokens_in=usage.input_tokens if usage else None,
                tokens_cached=usage.cached_input_tokens if usage else None,
                tokens_out=usage.output_tokens if usage else None,
                cost_usd=cost,
                baseline_cost_usd=baseline,
                error_code=error_code,
            )
        )


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request) -> Response:
    state = request.app.state
    recorder = _Recorder(state.log, state.registry, req.model, req.stream)
    try:
        spec, provider = resolve(req.model, state.registry, state.providers)
        recorder.spec = spec
        headers = {
            "X-Tierwise-Model": req.model,
            "X-Tierwise-Provider": provider.name,
            "X-Tierwise-Reason": "override",
            "X-Tierwise-Request-Id": recorder.request_id,
        }
        if req.stream:
            chunks = provider.stream(req, spec.wire_id)
            try:
                # Wait for the first chunk so early failures become ordinary HTTP errors.
                first = await anext(chunks, None)
            except BaseException:
                await chunks.aclose()
                raise
            return sse_response(
                chunks,
                first,
                request_id=recorder.request_id,
                model=req.model,
                headers=headers,
                include_usage=bool((req.stream_options or {}).get("include_usage")),
                on_done=recorder.done,
            )
        response = await provider.chat(req, spec.wire_id)
    except (GatewayError, ProviderError) as exc:
        err = exc if isinstance(exc, GatewayError) else gateway_error_from(exc)
        recorder.done(None, err.status, err.code)
        raise err from exc
    except Exception as exc:
        # A bug, not an upstream condition: log it, but still answer in OpenAI's shape.
        logger.exception("unexpected error handling %s", recorder.request_id)
        recorder.done(None, 500, "internal_error")
        message = "Internal gateway error."
        raise GatewayError(500, "internal_error", message, "api_error") from exc
    cost, _ = recorder.costs(response.usage)
    recorder.done(response.usage, 200, None)
    if cost is not None:
        headers["X-Tierwise-Cost-USD"] = f"{cost:.6f}"
    return JSONResponse(response.to_openai(), headers=headers)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    registry: Registry = request.app.state.registry
    providers: dict[str, Provider] = request.app.state.providers
    data = [
        {"id": m.id, "object": "model", "owned_by": m.provider}
        for m in registry.models
        if m.provider in providers
    ]
    return {"object": "list", "data": data}


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    return {"status": "ok", "providers": sorted(request.app.state.providers)}
