"""Prometheus metrics, fed from the one record every request produces.

Each app owns its own registry, so several apps in one process (tests) never share counts.
Breaker state and dropped telemetry are read live at scrape time rather than mirrored.
"""

from collections.abc import Collection, Iterator

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.exposition import generate_latest
from prometheus_client.registry import Collector

from tierwise.reliability.breaker import Breakers
from tierwise.store import RequestLog, RequestRecord

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
HOLD_BUCKETS = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 3)
UNRESOLVED_MODEL = "auto"  # label for requests that never reached a registry model


class _Live(Collector):
    """Values that already live elsewhere: read them when Prometheus scrapes."""

    def __init__(self, breakers: Breakers, log: RequestLog) -> None:
        self._breakers = breakers
        self._log = log

    def collect(self) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
        open_circuits = GaugeMetricFamily(
            "tierwise_breaker_open",
            "1 for each model whose circuit breaker is open",
            labels=["provider", "model"],
        )
        for provider, model in self._breakers.open_models():
            open_circuits.add_metric([provider, model], 1.0)
        yield open_circuits
        yield CounterMetricFamily(
            "tierwise_telemetry_dropped",
            "Request-log rows dropped because the write queue was full",
            value=self._log.dropped,
        )


class Metrics:
    def __init__(self, breakers: Breakers, log: RequestLog, model_ids: Collection[str]) -> None:
        self.registry = CollectorRegistry()
        self._models = frozenset(model_ids)
        r = self.registry
        self._requests = Counter(
            "tierwise_requests", "Requests by model, routing source and HTTP status",
            ["model", "source", "status"], registry=r,
        )  # fmt: skip
        self._cost = Counter("tierwise_cost_usd", "USD spent on providers", registry=r)
        self._saved = Counter(
            "tierwise_cost_saved_usd", "USD saved against the baseline model", registry=r
        )
        self._latency = Histogram(
            "tierwise_latency_seconds", "Request latency", buckets=LATENCY_BUCKETS, registry=r
        )
        self._cache_hits = Counter("tierwise_cache_hits", "Semantic cache hits", registry=r)
        self._failovers = Counter("tierwise_failovers", "Candidates abandoned", registry=r)
        self._shadow = Counter(
            "tierwise_shadow_samples", "Routed answers sampled for flywheel labels", registry=r
        )
        self._hold = Histogram(
            "tierwise_ttft_buffer_seconds",
            "Time a stream was held before its first content",
            buckets=HOLD_BUCKETS,
            registry=r,
        )
        r.register(_Live(breakers, log))

    def observe(self, record: RequestRecord) -> None:
        model = record.model if record.model in self._models else UNRESOLVED_MODEL
        self._requests.labels(model, record.source, str(record.status)).inc()
        self._latency.observe(record.latency_ms / 1000)
        if record.cost_usd is not None:
            self._cost.inc(record.cost_usd)
            if record.baseline_cost_usd is not None:
                self._saved.inc(max(0.0, record.baseline_cost_usd - record.cost_usd))
        if record.cache_hit:
            self._cache_hits.inc()
        if record.failover_count:
            self._failovers.inc(record.failover_count)

    def shadow_sample(self) -> None:
        self._shadow.inc()

    def observe_hold(self, seconds: float) -> None:
        self._hold.observe(seconds)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
