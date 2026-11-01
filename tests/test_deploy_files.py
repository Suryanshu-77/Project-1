"""Static checks on the compose stack, so CI catches a broken deploy without running Docker."""

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from starlette.testclient import TestClient

from tierwise.metrics import Metrics
from tierwise.providers.openai import from_openai
from tierwise.registry import load_registry
from tierwise.reliability.breaker import Breakers
from tierwise.store import RequestLog

ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "deploy"


def _compose() -> dict[str, Any]:
    return yaml.safe_load((DEPLOY / "docker-compose.yml").read_text(encoding="utf-8"))


def _dashboard() -> dict[str, Any]:
    path = DEPLOY / "grafana" / "dashboards" / "tierwise.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _exprs() -> dict[str, list[str]]:
    return {
        panel["title"]: [t["expr"] for t in panel.get("targets", [])]
        for panel in _dashboard()["panels"]
    }


def _metric_names() -> set[str]:
    """Every series name the gateway can expose, from a fresh Metrics registry."""
    metrics = Metrics(Breakers(), RequestLog("sqlite+aiosqlite://"), [])
    suffixes = {"counter": ["_total"], "histogram": ["_bucket", "_sum", "_count"], "gauge": [""]}
    return {
        family.name + suffix
        for family in metrics.registry.collect()
        for suffix in suffixes.get(family.type, [""])
    }


def test_compose_services_and_healthcheck() -> None:
    services = _compose()["services"]
    expected = {"postgres", "fake-upstream", "gateway", "prometheus", "grafana", "jaeger"}
    assert set(services) == expected
    assert "healthcheck" in services["postgres"]
    assert services["gateway"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    env = services["gateway"]["environment"]
    assert env["TIERWISE_REQUIRE_KEY"] == "true"
    assert env["TIERWISE_DB_URL"].startswith("postgresql+asyncpg://")


def test_dashboard_queries_use_real_metrics() -> None:
    known = _metric_names()
    used = {
        name
        for exprs in _exprs().values()
        for expr in exprs
        for name in re.findall(r"\btierwise_[a-z_]+\b", expr)
    }
    assert used, "the dashboard queries no tierwise metrics"
    assert used <= known, f"unknown metrics in dashboard: {sorted(used - known)}"


def test_dashboard_has_savings_panel() -> None:
    dashboard = _dashboard()
    assert dashboard["uid"] == "tierwise"
    [expr] = _exprs()["Saved vs all-flagship (%)"]
    assert "TODO" not in expr
    assert "tierwise_cost_usd_total" in expr and "tierwise_cost_saved_usd_total" in expr


def test_compose_registry_loads() -> None:
    registry = load_registry(str(DEPLOY / "registry.compose.yaml"))
    assert {m.provider for m in registry.models} == {"ollama"}
    assert {m.tier for m in registry.models} == {"basic", "moderate", "flagship"}


def _fake_upstream() -> ModuleType:
    path = ROOT / "scripts" / "fake_upstream.py"
    spec = importlib.util.spec_from_file_location("fake_upstream", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fake_upstream_reply_shape() -> None:
    client = TestClient(_fake_upstream().app)
    body = {"model": "small", "messages": [{"role": "user", "content": "hello there"}]}
    data = client.post("/v1/chat/completions", json=body).json()
    response = from_openai(data)
    assert response.content and response.usage.input_tokens > 0
    stream = client.post("/v1/chat/completions", json={**body, "stream": True}).text
    assert stream.rstrip().endswith("data: [DONE]") and '"usage"' in stream
    assert client.get("/stats").json()["requests"] == 2


def test_postgres_healthcheck_uses_tcp_and_gateway_restarts() -> None:
    services = _compose()["services"]
    check = " ".join(services["postgres"]["healthcheck"]["test"])
    assert "-h 127.0.0.1" in check  # the init server answers on the socket before TCP opens
    assert services["gateway"]["restart"] == "on-failure"


def test_published_ports_are_localhost_only() -> None:
    for name, service in _compose()["services"].items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} on all interfaces"
            assert not str(port).endswith(":4318"), "OTLP ingest needn't be reachable from the host"


def test_grafana_has_no_default_admin_login() -> None:
    env = _compose()["services"]["grafana"]["environment"]
    assert env["GF_AUTH_DISABLE_LOGIN_FORM"] == "true"
    assert env["GF_AUTH_BASIC_ENABLED"] == "false"


def test_image_runs_as_non_root() -> None:
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    users = [line.split()[1] for line in lines if line.startswith("USER ")]
    assert users and users[-1] not in {"root", "0"}


def test_release_publishes_only_after_the_test_suite() -> None:
    """A PyPI version can never be re-uploaded: the job publish waits on must run pytest."""
    release = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())
    needs = release["jobs"]["publish"]["needs"]
    gate = release["jobs"][needs]["steps"]
    assert any("pytest" in str(step.get("run", "")) for step in gate)


def test_fake_upstream_judges_hard_prompts_low() -> None:
    from tierwise.flywheel import JUDGE_MARKER, judge_messages

    module = _fake_upstream()
    assert module.JUDGE_MARKER == JUDGE_MARKER
    client = TestClient(module.app)

    def score(prompt: str) -> int:
        messages = [m.model_dump() for m in judge_messages(f"user: {prompt}", "a", "b", "s1")]
        reply = client.post("/v1/chat/completions", json={"model": "large", "messages": messages})
        return int(json.loads(reply.json()["choices"][0]["message"]["content"])["score"])

    assert score("Design a billing architecture") == 2
    assert score("Translate 'hello' into Spanish") == 5
