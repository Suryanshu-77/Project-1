"""HTTP endpoints. ``auto``/``tier:*`` requests are routed; a concrete model id is served as-is."""

import logging
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from tierwise.affinity import AffinityTable, prefix_boundaries, split_prefix
from tierwise.api.errors import GatewayError, gateway_error_from
from tierwise.api.streaming import sse_response
from tierwise.costs import cost_usd
from tierwise.options import Option
from tierwise.providers.base import Provider, ProviderError
from tierwise.registry import ModelSpec, Registry
from tierwise.router.modes import VIRTUAL_MODELS, EmptyAfterTag, UnknownRoutingValue
from tierwise.router.router import RoutingResult, route
from tierwise.schema import ChatRequest, Usage
from tierwise.selector import NoEligibleModel, select
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
        self.source = "override"
        self.mode: str | None = None
        self.difficulty: float | None = None
        self.confidence: float | None = None
        self.prefix_hash: str | None = None
        self.effort: str | None = None

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
                source=self.source,
                mode=self.mode,
                difficulty=self.difficulty,
                confidence=self.confidence,
                prefix_hash=self.prefix_hash,
                effort=self.effort,
                tokens_reasoning=usage.reasoning_tokens if usage else None,
                error_code=error_code,
            )
        )


def _choose(
    result: RoutingResult, state: object, recorder: _Recorder
) -> tuple[ModelSpec, Provider, str, str | None, Option | None]:
    """Pick the serving model and effort.

    Returns (spec, provider, reason, prompt hash to remember, chosen option); the option
    is None for a concrete model id, which is served exactly as requested.
    """
    registry: Registry = state.registry  # type: ignore[attr-defined]
    providers: dict[str, Provider] = state.providers  # type: ignore[attr-defined]
    affinity: AffinityTable = state.affinity  # type: ignore[attr-defined]
    target = result.target
    recorder.prefix_hash = split_prefix(result.request)[0]
    if target.kind == "model":
        spec, provider = resolve(result.request.model, registry, providers)
        boundaries = prefix_boundaries(result.request)
        return spec, provider, "override", boundaries[-1][0] if boundaries else None, None
    recorder.mode = target.mode if target.kind == "auto" else f"tier:{target.tier}"
    if result.decision is not None:
        recorder.difficulty = result.decision.difficulty
        recorder.confidence = result.decision.confidence
    try:
        selection = select(result, registry, providers.keys(), affinity)
    except NoEligibleModel as exc:
        raise GatewayError(422, "no_eligible_model", exc.message) from exc
    option = selection.candidates[0]
    spec = option.spec
    return spec, providers[spec.provider], selection.reason, selection.prompt_hash, option


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request) -> Response:
    state = request.app.state
    recorder = _Recorder(state.log, state.registry, req.model, req.stream)
    affinity: AffinityTable = state.affinity
    try:
        try:
            result = route(req)
        except UnknownRoutingValue as exc:
            raise UnknownModel(req.model) from exc
        except EmptyAfterTag as exc:
            raise GatewayError(400, "invalid_request_error", str(exc)) from exc
        spec, provider, reason, prompt_hash, option = _choose(result, state, recorder)
        recorder.spec, recorder.model, recorder.source = spec, spec.id, reason
        effort_header: str
        if option is None:  # concrete model: forward the client's effort (if any) unchanged
            forward = result.request
            sent_effort = forward.reasoning_effort
            effort_header = sent_effort or "default"
        else:
            sent_effort = option.wire_effort
            forward = result.request.model_copy(update={"reasoning_effort": sent_effort})
            effort_header = option.header_value
        recorder.effort = effort_header
        headers = {
            "X-Tierwise-Model": spec.id,
            "X-Tierwise-Provider": provider.name,
            "X-Tierwise-Reason": reason,
            "X-Tierwise-Request-Id": recorder.request_id,
            "X-Tierwise-Effort": effort_header,
        }

        def remember() -> None:
            if prompt_hash is not None:
                affinity.record(prompt_hash, spec.id, sent_effort)

        if forward.stream:

            def stream_done(usage: Usage | None, status: int, code: str | None) -> None:
                recorder.done(usage, status, code)
                if status == 200:
                    remember()

            chunks = provider.stream(forward, spec.wire_id)
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
                model=spec.id,
                headers=headers,
                include_usage=bool((forward.stream_options or {}).get("include_usage")),
                on_done=stream_done,
            )
        response = await provider.chat(forward, spec.wire_id)
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
    remember()
    if cost is not None:
        headers["X-Tierwise-Cost-USD"] = f"{cost:.6f}"
    return JSONResponse(response.to_openai(), headers=headers)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    registry: Registry = request.app.state.registry
    providers: dict[str, Provider] = request.app.state.providers
    virtual = [{"id": v, "object": "model", "owned_by": "tierwise"} for v in VIRTUAL_MODELS]
    data = [
        {"id": m.id, "object": "model", "owned_by": m.provider}
        for m in registry.models
        if m.provider in providers
    ]
    return {"object": "list", "data": virtual + data}


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    return {"status": "ok", "providers": sorted(request.app.state.providers)}
