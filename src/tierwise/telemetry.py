"""OpenTelemetry tracing: spans for routing, the cache, selection and each provider attempt.

Without ``OTEL_EXPORTER_OTLP_ENDPOINT`` nothing is installed and the API is a no-op. Spans
carry model, provider and routing attributes, never prompt or answer text, nor keys.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

tracer = trace.get_tracer("tierwise")


def configure_tracing(
    endpoint: str | None, service_name: str = "tierwise"
) -> TracerProvider | None:
    """Export spans over OTLP/HTTP to ``endpoint``; None leaves tracing off."""
    if not endpoint:
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def provider_span(provider: str, model: str) -> Iterator[Span]:
    """One span per upstream attempt; a raised error marks it failed and propagates."""
    with tracer.start_as_current_span(
        "tierwise.provider",
        attributes={"tierwise.provider": provider, "tierwise.model": model},
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            span.set_attribute("error.type", type(exc).__name__)
            raise
