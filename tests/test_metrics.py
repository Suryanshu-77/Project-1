from pathlib import Path

from tierwise.metrics import Metrics
from tierwise.providers.base import RetryableError
from tierwise.reliability.breaker import Breakers
from tierwise.store import RequestLog, RequestRecord


def _record(**kw: object) -> RequestRecord:
    fields: dict[str, object] = {
        "id": "r", "ts": 0.0, "model": "m", "provider": "p", "status": 200, "stream": False,
        "latency_ms": 120.0, "tokens_in": 1, "tokens_cached": 0, "tokens_out": 1,
        "cost_usd": 0.01, "baseline_cost_usd": 0.05, "source": "rules",
    }  # fmt: skip
    fields.update(kw)
    return RequestRecord(**fields)  # type: ignore[arg-type]


def _metrics(breakers: Breakers | None = None) -> Metrics:
    return Metrics(breakers or Breakers(), RequestLog("sqlite+aiosqlite://"), {"m", "n"})


def _text(metrics: Metrics) -> str:
    return metrics.render()[0].decode()


def _value(metrics: Metrics, name: str, labels: dict[str, str] | None = None) -> float:
    value = metrics.registry.get_sample_value(name, labels or {})
    return 0.0 if value is None else value


def test_observe_counts_and_costs() -> None:
    metrics = _metrics()
    metrics.observe(_record())
    metrics.observe(_record(cost_usd=0.02, baseline_cost_usd=0.02))
    assert _value(metrics, "tierwise_cost_usd_total") == 0.03
    assert abs(_value(metrics, "tierwise_cost_saved_usd_total") - 0.04) < 1e-12
    labels = {"model": "m", "source": "rules", "status": "200"}
    assert _value(metrics, "tierwise_requests_total", labels) == 2
    assert _value(metrics, "tierwise_latency_seconds_count") == 2


def test_negative_saving_adds_nothing() -> None:
    metrics = _metrics()
    metrics.observe(_record(cost_usd=0.09, baseline_cost_usd=0.05))
    assert _value(metrics, "tierwise_cost_saved_usd_total") == 0.0


def test_cache_hit_and_failovers_counted() -> None:
    metrics = _metrics()
    metrics.observe(_record(cache_hit=True, source="cache", cost_usd=0.0))
    metrics.observe(_record(failover_count=2))
    assert _value(metrics, "tierwise_cache_hits_total") == 1
    assert _value(metrics, "tierwise_failovers_total") == 2


def test_unknown_model_label_is_auto() -> None:
    metrics = _metrics()
    metrics.observe(_record(model="typo-xyz", status=404))
    assert "typo-xyz" not in _text(metrics)
    labels = {"model": "auto", "source": "rules", "status": "404"}
    assert _value(metrics, "tierwise_requests_total", labels) == 1


def test_breaker_open_gauge_reads_live_state() -> None:
    breakers = Breakers()
    metrics = _metrics(breakers)
    for _ in range(5):
        breakers.record("p", "m", RetryableError("p", 503, "down"))
    assert _value(metrics, "tierwise_breaker_open", {"provider": "p", "model": "m"}) == 1.0


def test_telemetry_dropped_reads_log() -> None:
    log = RequestLog("sqlite+aiosqlite://", max_queue=1)
    metrics = Metrics(Breakers(), log, {"m"})
    log.submit(_record())
    log.submit(_record())
    assert _value(metrics, "tierwise_telemetry_dropped_total") == 1


def test_hold_histogram() -> None:
    metrics = _metrics()
    metrics.observe_hold(0.02)
    assert _value(metrics, "tierwise_ttft_buffer_seconds_count") == 1


def test_registries_are_isolated() -> None:
    a, b = _metrics(), _metrics()
    a.observe(_record())
    assert _value(b, "tierwise_cost_usd_total") == 0.0


def test_render_is_prometheus_text(tmp_path: Path) -> None:
    body, content_type = _metrics().render()
    assert content_type.startswith("text/plain")
    assert b"# TYPE tierwise_requests_total counter" in body


def test_shadow_metric_exported() -> None:
    metrics = _metrics()
    metrics.shadow_sample()
    metrics.shadow_sample()
    assert _value(metrics, "tierwise_shadow_samples_total") == 2.0
