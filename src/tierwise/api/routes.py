"""HTTP endpoints. ``auto``/``tier:*`` requests are routed; a concrete model id is served as-is."""

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from tierwise.affinity import AffinityTable, prefix_boundaries, split_prefix
from tierwise.api.auth import authenticate
from tierwise.api.errors import GatewayError, gateway_error_from
from tierwise.api.streaming import (
    STATUS_UPSTREAM_FAILED_MID_STREAM,
    Primed,
    prime_stream,
    sse_response,
)
from tierwise.cache import (
    CachedAnswer,
    CacheKey,
    SemanticCache,
    cacheable,
    last_user_text,
    scope_key,
)
from tierwise.costs import cost_usd
from tierwise.options import Option
from tierwise.providers.base import Provider, ProviderError, RetryableError
from tierwise.registry import ModelSpec, Registry
from tierwise.reliability.breaker import Breakers
from tierwise.reliability.failover import serve
from tierwise.reliability.retry import RetryPolicy, with_retries
from tierwise.router.modes import VIRTUAL_MODELS, EmptyAfterTag, UnknownRoutingValue
from tierwise.router.router import RoutingResult, route
from tierwise.schema import ChatRequest, ChatResponse, FinishReason, StreamChunk, Usage
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
        self.failover_count = 0
        self.cache_hit = False
        self.key_id: str | None = None

    def costs(self, usage: Usage | None) -> tuple[float | None, float | None]:
        if usage is None or self.spec is None:
            return None, None
        cost = 0.0 if self.cache_hit else cost_usd(usage, self.spec)
        return cost, cost_usd(usage, self.registry.baseline())

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
                failover_count=self.failover_count,
                cache_hit=self.cache_hit,
                key_id=self.key_id,
                tokens_reasoning=usage.reasoning_tokens if usage else None,
                error_code=error_code,
            )
        )


@dataclass(frozen=True)
class _Plan:
    """What to call: a concrete model, or the selector's ordered candidates."""

    candidates: list[Option]
    concrete: ModelSpec | None
    reason: str
    prompt_hash: str | None
    clear_count: int | None = None


def _choose(result: RoutingResult, state: object, recorder: _Recorder) -> _Plan:
    registry: Registry = state.registry  # type: ignore[attr-defined]
    providers: dict[str, Provider] = state.providers  # type: ignore[attr-defined]
    affinity: AffinityTable = state.affinity  # type: ignore[attr-defined]
    target = result.target
    recorder.prefix_hash = split_prefix(result.request)[0]
    if target.kind == "model":
        spec, _ = resolve(result.request.model, registry, providers)
        boundaries = prefix_boundaries(result.request)
        return _Plan([], spec, "override", boundaries[-1][0] if boundaries else None)
    recorder.mode = target.mode if target.kind == "auto" else f"tier:{target.tier}"
    if result.decision is not None:
        recorder.difficulty = result.decision.difficulty
        recorder.confidence = result.decision.confidence
    try:
        selection = select(result, registry, providers.keys(), affinity)
    except NoEligibleModel as exc:
        raise GatewayError(422, "no_eligible_model", exc.message) from exc
    return _Plan(
        selection.candidates,
        None,
        selection.reason,
        selection.prompt_hash,
        selection.clear_count,
    )


def _forward(request: ChatRequest, option: Option | None) -> tuple[ChatRequest, str | None, str]:
    """(request to send, effort sent on the wire, X-Tierwise-Effort value)."""
    if option is None:  # concrete model: forward the client's effort (if any) unchanged
        return request, request.reasoning_effort, request.reasoning_effort or "default"
    sent = option.wire_effort
    return request.model_copy(update={"reasoning_effort": sent}), sent, option.header_value


def _cached_response(
    hit: CachedAnswer, result: RoutingResult, recorder: _Recorder, spec: ModelSpec
) -> Response:
    """Answer from the cache, as JSON or as a short SSE stream."""
    recorder.spec, recorder.model, recorder.source = spec, spec.id, "cache"
    recorder.cache_hit = True
    recorder.mode = result.request.model
    headers = {
        "X-Tierwise-Model": spec.id,
        "X-Tierwise-Provider": spec.provider,
        "X-Tierwise-Reason": "cache",
        "X-Tierwise-Request-Id": recorder.request_id,
        "X-Tierwise-Effort": "default",
        "X-Tierwise-Failovers": "0",
    }
    finish = cast(FinishReason, hit.finish_reason)
    if result.request.stream:

        async def replay() -> AsyncGenerator[StreamChunk, None]:
            yield StreamChunk(content=hit.content)
            yield StreamChunk(finish_reason=finish)
            yield StreamChunk(usage=hit.usage)

        return sse_response(
            Primed(replay(), []),
            request_id=recorder.request_id,
            model=spec.id,
            headers=headers,
            include_usage=bool((result.request.stream_options or {}).get("include_usage")),
            on_done=recorder.done,
        )
    response = ChatResponse(
        id=f"chatcmpl-{recorder.request_id}",
        model=spec.id,
        provider=spec.provider,
        content=hit.content,
        tool_calls=None,
        finish_reason=finish,
        usage=hit.usage,
        created=int(time.time()),
    )
    recorder.done(hit.usage, 200, None)
    headers["X-Tierwise-Cost-USD"] = "0.000000"
    return JSONResponse(response.to_openai(), headers=headers)


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request) -> Response:
    state = request.app.state
    recorder = _Recorder(state.log, state.registry, req.model, req.stream)
    affinity: AffinityTable = state.affinity
    breakers: Breakers = state.breakers
    policy: RetryPolicy = state.retry
    providers: dict[str, Provider] = state.providers
    try:
        key = authenticate(
            request.headers.get("authorization"), state.keys, state.require_api_key
        )
        recorder.key_id = key.id if key is not None else None
        if key is not None:
            state.limiter.check(key)
        try:
            result = await route(req, state.learned)
        except UnknownRoutingValue as exc:
            raise UnknownModel(req.model) from exc
        except EmptyAfterTag as exc:
            raise GatewayError(400, "invalid_request_error", str(exc)) from exc
        cache: SemanticCache | None = state.cache
        cache_key: CacheKey | None = None
        if cache is not None and cacheable(
            result, request.headers, state.cache_max_temperature
        ):
            cache_key = await cache.key(scope_key(result), last_user_text(result))
            hit = cache.lookup(cache_key)
            hit_spec = state.registry.get(hit.model_id) if hit is not None else None
            if hit is not None and hit_spec is not None:
                return _cached_response(hit, result, recorder, hit_spec)

        def keep(content: str, finish: str, usage: Usage, spec: ModelSpec) -> None:
            """Cache only complete text answers."""
            if cache is not None and cache_key is not None and content and finish == "stop":
                cache.store(cache_key, CachedAnswer(content, spec.id, spec.provider, usage, finish))

        plan = _choose(result, state, recorder)
        recorder.source = plan.reason

        def served_by(spec: ModelSpec, option: Option | None, failovers: int) -> dict[str, str]:
            _, sent_effort, effort_header = _forward(result.request, option)
            recorder.spec, recorder.model, recorder.failover_count = spec, spec.id, failovers
            # log the level that actually ran (the model's default when nothing was sent)
            recorder.effort = sent_effort if option is None else option.effort
            return {
                "X-Tierwise-Model": spec.id,
                "X-Tierwise-Provider": spec.provider,
                "X-Tierwise-Reason": plan.reason,
                "X-Tierwise-Request-Id": recorder.request_id,
                "X-Tierwise-Effort": effort_header,
                "X-Tierwise-Failovers": str(failovers),
            }

        def remember(spec: ModelSpec, option: Option | None) -> None:
            if plan.prompt_hash is not None:
                affinity.record(plan.prompt_hash, spec.id, _forward(result.request, option)[1])

        if result.request.stream:
            hold_s, hold_bytes = state.stream_hold

            async def open_stream(candidate: Option | None, target: ModelSpec) -> Primed:
                outgoing = _forward(result.request, candidate)[0]
                chunks = providers[target.provider].stream(outgoing, target.wire_id)
                return await prime_stream(chunks, hold_s, hold_bytes)

            if plan.concrete is not None:
                spec, option, failovers = plan.concrete, None, 0
                served_by(spec, None, 0)  # so an error is logged against this model
                try:
                    primed = await with_retries(partial(open_stream, None, spec), policy)
                except ProviderError as exc:
                    breakers.record(spec.provider, spec.id, exc)
                    raise
            else:
                streamed = await serve(
                    plan.candidates,
                    lambda c: open_stream(c, c.spec),
                    breakers,
                    policy,
                    clear_count=plan.clear_count,
                    record_success=False,  # stream_done gives the verdict when it ends
                )
                primed, option, failovers = streamed.value, streamed.option, streamed.failovers
                spec = option.spec
            headers = served_by(spec, option, failovers)
            forward = _forward(result.request, option)[0]

            streamed_chunks: list[StreamChunk] = []

            def stream_done(usage: Usage | None, status: int, code: str | None) -> None:
                recorder.done(usage, status, code)
                if status == 200:
                    breakers.record(spec.provider, spec.id, None)
                    remember(spec, option)
                    if usage is not None:
                        text = "".join(c.content or "" for c in streamed_chunks)
                        finishes = [c.finish_reason for c in streamed_chunks if c.finish_reason]
                        has_tools = any(c.tool_calls for c in streamed_chunks)
                        if not has_tools:
                            keep(text, finishes[-1] if finishes else "", usage, spec)
                elif status == STATUS_UPSTREAM_FAILED_MID_STREAM:  # failed after first content
                    failure = RetryableError(spec.provider, None, code or "mid-stream failure")
                    breakers.record(spec.provider, spec.id, failure)
                else:  # the client left: no verdict on the upstream, give back any probe
                    breakers.provider(spec.provider).release()
                    breakers.model(spec.provider, spec.id).release()

            return sse_response(
                primed,
                request_id=recorder.request_id,
                model=spec.id,
                headers=headers,
                include_usage=bool((forward.stream_options or {}).get("include_usage")),
                on_done=stream_done,
                collect=streamed_chunks,
            )

        if plan.concrete is not None:
            spec, option, failovers = plan.concrete, None, 0
            forward = _forward(result.request, None)[0]
            served_by(spec, None, 0)  # so an error is logged against this model
            try:
                response = await with_retries(
                    partial(providers[spec.provider].chat, forward, spec.wire_id), policy
                )
            except ProviderError as exc:
                breakers.record(spec.provider, spec.id, exc)
                raise
            breakers.record(spec.provider, spec.id, None)
        else:

            async def attempt(candidate: Option) -> ChatResponse:
                outgoing = _forward(result.request, candidate)[0]
                provider = providers[candidate.spec.provider]
                return await provider.chat(outgoing, candidate.spec.wire_id)

            served = await serve(
                plan.candidates, attempt, breakers, policy, clear_count=plan.clear_count
            )
            response, option, failovers = served.value, served.option, served.failovers
            spec = option.spec
        headers = served_by(spec, option, failovers)
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
    remember(spec, option)
    if not response.tool_calls:
        keep(response.content or "", response.finish_reason, response.usage, spec)
    if cost is not None:
        headers["X-Tierwise-Cost-USD"] = f"{cost:.6f}"
    return JSONResponse(response.to_openai(), headers=headers)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    state = request.app.state
    authenticate(request.headers.get("authorization"), state.keys, state.require_api_key)
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
