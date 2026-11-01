from collections.abc import Iterator

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.api.helpers import gateway, make_key, ok_upstream
from tests.api.test_routing_route import routed
from tierwise.telemetry import configure_tracing

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _provider() -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(provider)  # once per process: the API ignores later overrides


@pytest.fixture(autouse=True)
def spans() -> Iterator[InMemorySpanExporter]:
    EXPORTER.clear()
    yield EXPORTER


HI = {"model": "auto", "messages": [{"role": "user", "content": "the secret prompt words"}]}


async def test_request_produces_expected_spans(spans: InMemorySpanExporter) -> None:
    async with routed() as (_, client):
        resp = await client.post("/v1/chat/completions", json=HI)
    names = [s.name for s in spans.get_finished_spans()]
    for name in ("tierwise.request", "tierwise.route", "tierwise.select", "tierwise.provider"):
        assert name in names
    [provider_span] = [s for s in spans.get_finished_spans() if s.name == "tierwise.provider"]
    assert provider_span.attributes["tierwise.model"] == resp.headers["X-Tierwise-Model"]
    root = next(s for s in spans.get_finished_spans() if s.name == "tierwise.request")
    assert root.attributes["http.status_code"] == 200
    assert provider_span.parent is not None and provider_span.parent.span_id == root.context.span_id


async def test_failover_makes_one_provider_span_per_attempt(spans: InMemorySpanExporter) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:  # the first attempt is rate-limited
        calls.append(request.url.host)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "slow"})
        return ok_upstream(request)

    async with routed(handler=handler) as (_, client):
        await client.post("/v1/chat/completions", json=HI)
    attempts = [s for s in spans.get_finished_spans() if s.name == "tierwise.provider"]
    assert len(attempts) == 2
    assert attempts[0].status.status_code == trace.StatusCode.ERROR
    assert attempts[1].status.status_code != trace.StatusCode.ERROR


async def test_spans_never_contain_prompt_or_key(spans: InMemorySpanExporter) -> None:
    async with gateway(require_api_key=True) as (app, client):
        key, _ = await make_key(app)
        await client.post(
            "/v1/chat/completions",
            json={**HI, "model": "gpt-4.1-mini"},
            headers={"Authorization": f"Bearer {key}"},
        )
    for span in spans.get_finished_spans():
        for value in (span.attributes or {}).values():
            assert "secret prompt" not in str(value) and key not in str(value)


def test_tracing_off_without_endpoint() -> None:
    assert configure_tracing(None) is None
