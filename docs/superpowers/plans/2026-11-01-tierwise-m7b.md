# tierwise M7b Implementation Plan: Metrics, Tracing, Dashboard, docker-compose

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docker compose up` starts the gateway with Postgres, Prometheus, Grafana and Jaeger, and the provisioned dashboard shows the model mix, cost saved against the flagship, p95 latency, cache hit rate, failovers and open circuits.

**Architecture:**
- **`metrics.py`:** Prometheus metrics on a per-app `CollectorRegistry`, so test apps don't collide. They are fed from the one place every request already ends, `_Recorder.done`. Open circuits are read from `Breakers` at scrape time.
- **`telemetry.py`:** OpenTelemetry spans for route, select, cache and the provider call. Tracing is a no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- **`deploy/` directory:**
  - The compose file and the Prometheus and Grafana provisioning.
  - The dashboard JSON.
  - A **fake OpenAI-compatible upstream** (`scripts/fake_upstream.py`), so the stack runs with no API keys and costs nothing.
- **Exit check:** a smoke script (`scripts/compose_smoke.py`) checks that every part works, locally and in a CI job.

**Tech Stack:** `prometheus-client`, `opentelemetry-api` / `-sdk` / `-exporter-otlp-proto-http`, `asyncpg`. Images: `postgres:16`, `prom/prometheus:v2.55.1`, `grafana/grafana:11.3.0` and `jaegertracing/all-in-one:1.62.0` (pinned).

**Spec:** `docs/superpowers/specs/2026-11-01-tierwise-design.md` §2 (`GET /metrics`), §3.9 (metrics list, spans, provisioned dashboard, Postgres via asyncpg), §7 (pool reuse), §8 M7 exit check

## Global Constraints
- Everything in the M3–M7a plans' Global Constraints still holds.
- **Metric names (spec §3.9, exact):**
  - `tierwise_requests_total{model,source,status}`
  - `tierwise_cost_usd_total`
  - `tierwise_cost_saved_usd_total`
  - `tierwise_latency_seconds` (histogram)
  - `tierwise_cache_hits_total`
  - `tierwise_failovers_total`
  - `tierwise_breaker_open{provider,model}`
  - `tierwise_ttft_buffer_seconds`
  - `tierwise_telemetry_dropped_total`
  - `tierwise_shadow_samples_total` belongs to M9 and is **not** added here.
- **Latency buckets:** `(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)` seconds. TTFT-hold buckets: `(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 3)`.
- **Label values are bounded:**
  - `model` is a registry id or `"auto"` (an unresolved request).
  - `source` comes from the existing reason set.
  - `status` is the HTTP status code as a string.
- **Spans never carry prompt or answer text, or keys.** Allowed attributes: `tierwise.model`, `tierwise.provider`, `tierwise.reason`, `tierwise.mode`, `tierwise.effort`, `tierwise.cache_hit`, `tierwise.failovers`, `http.status_code`.
- **`/metrics`** needs no API key. It serves the default `prometheus_client` text format, and the compose network keeps it internal. It is documented as such.
- **Compose:** the gateway sets `TIERWISE_REQUIRE_KEY=true`, `KEYS_REFRESH_S=2`, `TIERWISE_DB_URL=postgresql+asyncpg://tierwise:tierwise@postgres:5432/tierwise` and `OLLAMA_BASE_URL=http://fake-upstream:8080/v1`, with no real provider keys. Grafana allows anonymous viewing, and the dashboard uid is `tierwise`.
- **Registry for compose:** `deploy/registry.compose.yaml` lists three Ollama-served models at three price points and tiers, all served by the fake upstream, so routing and savings are visible.

## Decisions this plan makes
- **The fake upstream ships in the repo and in compose.** A demo stack that needs paid API keys to show a dashboard can't be checked in CI. Real providers are enabled by adding their keys to `.env`. Cost if wrong: none.
- **Per-app `CollectorRegistry`:** the global default registry breaks the many apps the test suite creates. Cost if wrong: none.
- **`/metrics` is unauthenticated:** this matches the Prometheus convention. Keep the port off the public network, which the README states. Cost if wrong: one dependency on the auth helper.
- **Postgres is tested in CI with a service container,** as a parametrized store test plus the compose smoke run. Local runs use SQLite. Cost if wrong: none.

## Review Focus
1. **Counters after failover, cache hits and stream outcomes:** a streamed request ending in 499 or 502 is counted exactly once with that status, and a cache hit counts in both `requests_total{source="cache"}` and `cache_hits_total`. Test in Task 1.
2. **Label explosion:** an unknown model id in a client request (404) must not create a label per typo. It uses `model="auto"` or the raw value only for registry ids. Test in Task 1.
3. **Tracing off by default:** no exporter, no network calls, and no measurable overhead when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset. Test in Task 2.
4. **Postgres upsert under concurrency:** two flushes for the same (key, month) are summed, not overwritten. Test in Task 3, on Postgres in CI.
5. **Compose start-up order:** the gateway starts before Postgres accepts connections, so it must retry or wait (a `depends_on: condition: service_healthy` healthcheck) and not crash-loop. Checked by the smoke run in Task 5.

---

### Task 1: Prometheus metrics and `/metrics`

**Files:**
- Create: `src/tierwise/metrics.py`
- Modify:
  - `pyproject.toml`: add `prometheus-client>=0.21`.
  - `src/tierwise/api/routes.py`: `_Recorder.done` calls `metrics.observe(record)`. The stream path observes the TTFT hold, and `GET /metrics` is added.
  - `src/tierwise/api/streaming.py`: `Primed` gains `held_s: float`, set by `prime_stream`.
  - `src/tierwise/api/app.py`: `app.state.metrics = Metrics(breakers=..., log=...)`.
- Test: `tests/test_metrics.py`, `tests/api/test_metrics_route.py`

**Interfaces:**
- Consumes: `RequestRecord`, `Breakers` and `RequestLog.dropped`.
- Produces:
  - `class Metrics`:
    - `__init__(self, breakers: Breakers, log: RequestLog)`. It owns `self.registry: CollectorRegistry`.
    - `observe(self, record: RequestRecord) -> None`:
      - `requests_total` += 1.
      - `latency` observes `record.latency_ms / 1000`.
      - `cost_usd_total` += `cost_usd`, and `cost_saved_usd_total` += `baseline − cost` (when both are known; a negative value adds 0).
      - `cache_hits_total` += 1 if `cache_hit`, and `failovers_total` += `failover_count`.
    - `observe_hold(self, seconds: float) -> None`
    - `render(self) -> tuple[bytes, str]`: the body and content type, from `generate_latest`.
  - `breaker_open{provider,model}` and `telemetry_dropped_total` are **collected at scrape time** by a custom collector reading `Breakers` (1 for each open model breaker) and `log.dropped`.
- [ ] **Step 1: Write the failing tests.**
  - `test_observe_counts_and_costs`: two records (cost 0.01, baseline 0.05; cost 0.02, baseline 0.02) → `cost_usd_total` is 0.03, `cost_saved_usd_total` is 0.04 and `requests_total{model,source,status}` is 2.
  - `test_cache_hit_and_failovers_counted`.
  - `test_breaker_open_gauge_reads_live_state`: trip a breaker → the rendered text contains `tierwise_breaker_open{model="m",provider="p"} 1.0`.
  - `test_registries_are_isolated`: two `Metrics` don't share counts.
  - API:
    - `test_metrics_endpoint_after_requests`: two requests, then `GET /metrics` → 200, `text/plain`, and `tierwise_requests_total` ≥ 2.
    - `test_streamed_request_counted_once_with_status`: a stream that fails in-band is counted once with `status="502"`.
    - `test_unknown_model_does_not_add_label`: a 404 for `model="typo-xyz"` → `typo-xyz` doesn't appear in `/metrics`.
- [ ] **Step 2: Run** `uv run pytest tests/test_metrics.py tests/api/test_metrics_route.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `observe`, use `record.model` only if it is in the registry, otherwise `"auto"`. Pass the registry ids in at construction.
- [ ] **Step 4: Run** `uv run pytest -q && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: Prometheus metrics and /metrics"`

### Task 2: OpenTelemetry spans

**Files:**
- Create: `src/tierwise/telemetry.py`
- Modify:
  - `pyproject.toml`: add `opentelemetry-api>=1.28`, `opentelemetry-sdk>=1.28` and `opentelemetry-exporter-otlp-proto-http>=1.28`.
  - `src/tierwise/api/routes.py`: add spans.
  - `src/tierwise/api/app.py`: call `configure_tracing(config)` at startup and `shutdown_tracing()` at stop.
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Produces:
  - `tracer = trace.get_tracer("tierwise")`
  - `configure_tracing(endpoint: str | None, service_name: str = "tierwise") -> TracerProvider | None`. With None: nothing is installed (no-op API). Otherwise a `BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))`. The endpoint comes from the `OTEL_EXPORTER_OTLP_ENDPOINT` setting.
  - Span names: `tierwise.request` (root), `tierwise.route`, `tierwise.cache`, `tierwise.select` and `tierwise.provider` (one per attempted candidate, attributes `tierwise.model` and `tierwise.provider`, and an `error.type` status on failure).
- [ ] **Step 1: Write the failing tests** (an SDK `TracerProvider` with `InMemorySpanExporter` installed by a fixture).
  - `test_request_produces_expected_spans`: one auto request → spans `tierwise.request`, `tierwise.route`, `tierwise.select` and `tierwise.provider`. The provider span has `tierwise.model` equal to the `X-Tierwise-Model` header.
  - `test_failover_makes_one_provider_span_per_attempt`: a 429 on the first host → two provider spans, the first with error status.
  - `test_spans_never_contain_prompt_or_key`: for every span, the attribute values don't contain the prompt text or the `tw-` key.
  - `test_tracing_off_without_endpoint`: `configure_tracing(None)` returns None and installs no provider.
- [ ] **Step 2: Run** `uv run pytest tests/test_telemetry.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: OpenTelemetry spans for route, cache, select and provider calls"`

### Task 3: Postgres support

**Files:**
- Modify: `pyproject.toml` (add `asyncpg>=0.30`), `tests/test_store.py`, `.github/workflows/ci.yml`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: the store's existing dialect switch (`postgresql.insert` for upserts).
- [ ] **Step 1: Write the tests.**
  - A `db_url` fixture, parametrized over `["sqlite", "postgres"]`. Postgres reads `TIERWISE_TEST_PG_URL` and **skips** when it is unset (local). Every test using `_url(tmp_path)` that needs no SQLite-specific checks switches to this fixture; the WAL and pragma tests stay SQLite-only.
  - `test_concurrent_flushes_sum_not_overwrite` (both dialects): two `RequestLog` instances on the same database each debit (k, m) by 0.10 and stop → `spent_usd == 0.20` and `requests == 2`.
  - In `ci.yml`, add a `postgres:16` service (user, password and db all `tierwise`, port 5432, with a `pg_isready` healthcheck) and set `TIERWISE_TEST_PG_URL=postgresql+asyncpg://tierwise:tierwise@localhost:5432/tierwise` for the pytest step.
- [ ] **Step 2: Run** `uv run pytest tests/test_store.py -q` locally. Expected: the SQLite variants pass and the Postgres ones are skipped. The real Postgres run happens in CI, on the PR.
- [ ] **Step 3: Fix any dialect issue the Postgres run surfaces.** Use the Postgres image locally (`docker run -p 5432:5432 -e POSTGRES_PASSWORD=tierwise …`) if Docker Desktop is available. Each fix gets its own failing test first.
- [ ] **Step 4: Run** `TIERWISE_TEST_PG_URL=… uv run pytest tests/test_store.py -q` against local Postgres if available. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: Postgres support via asyncpg; store tests run on Postgres in CI"`

### Task 4: Container image, fake upstream, compose and the dashboard (hand-off: the savings query)

**Files:**
- Create:
  - `Dockerfile`: `python:3.12-slim` with uv. It installs the project without dev or train groups, copies `models/router/` (no `model.onnx`, so the gateway runs rules-only), and runs `CMD ["tierwise"]`.
  - `.dockerignore`
  - `scripts/fake_upstream.py`, a small Starlette app on port 8080:
    - `POST /v1/chat/completions` returns a deterministic OpenAI-shaped reply (streaming too), with `usage` scaled to the prompt length.
    - `GET /stats` returns `{"requests": n, "connections": distinct (client host, port) pairs}`.
  - `deploy/docker-compose.yml`, with services:
    - `postgres`: with a healthcheck.
    - `fake-upstream`: the same image, `CMD python scripts/fake_upstream.py`.
    - `gateway`: port 8000, `depends_on` postgres healthy and fake-upstream started.
    - `prometheus`: port 9090, scraping `gateway:8000/metrics` every 5 s.
    - `grafana`: port 3000, anonymous viewer, provisioned datasource and dashboard.
    - `jaeger`: ports 16686 and 4318.
  - `deploy/prometheus.yml`
  - `deploy/grafana/provisioning/datasources/prometheus.yml`
  - `deploy/grafana/provisioning/dashboards/tierwise.yml`
  - `deploy/grafana/dashboards/tierwise.json`
  - `deploy/registry.compose.yaml`
- Test: `tests/test_deploy_files.py` (static checks that run in CI without Docker)

**Interfaces:**
- Consumes: Task 1's metric names.
- Produces: the dashboard, uid `tierwise`, titled "tierwise", with these panels:
  - "Requests by model" (`sum by (model) (rate(tierwise_requests_total[5m]))`)
  - "Cost vs flagship baseline" (cost and cost + saved)
  - **"Saved vs all-flagship (%)", the `TODO(human)` PromQL**
  - "p95 latency" (`histogram_quantile(0.95, sum by (le) (rate(tierwise_latency_seconds_bucket[5m])))`)
  - "Cache hit rate"
  - "Failovers / min"
  - "Open circuits" (`tierwise_breaker_open == 1`)
  - "Stream hold p95"
- [ ] **Step 1: Write the failing static tests.**
  - `test_compose_services_and_healthcheck`: the YAML has these 6 services, `postgres` has a healthcheck, and `gateway.depends_on.postgres.condition == "service_healthy"`.
  - `test_dashboard_queries_use_real_metrics`: every PromQL `expr` in the dashboard references only metric names produced by `Metrics(...)` (rendered from a fresh instance, with `_bucket`, `_sum`, `_count` and `_total` suffixes resolved).
  - `test_dashboard_has_savings_panel`: a panel titled "Saved vs all-flagship (%)" exists, and its expr uses both `tierwise_cost_usd_total` and `tierwise_cost_saved_usd_total`.
  - `test_compose_registry_loads`: `load_registry("deploy/registry.compose.yaml")` passes validation, and all its models use provider `ollama`.
  - `test_fake_upstream_reply_shape`: the Starlette test client → a valid `ChatResponse` through the OpenAI adapter's parser; `/stats` counts requests.
- [ ] **Step 2: Run** `uv run pytest tests/test_deploy_files.py -q`. Expected: FAIL.
- [ ] **Step 3: Write the files.** Put `"expr": "TODO(human)"` in the savings panel, request the contribution, and wait.
- [ ] **Step 4: Run** the same command after the user's PromQL. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: docker-compose stack with Postgres, Prometheus, Grafana, Jaeger and a fake upstream"`

### Task 5: Compose smoke check (the M7 exit check), in CI and locally

**Files:**
- Create: `scripts/compose_smoke.py` (stdlib and httpx only). It:
  - Waits for the gateway's `/healthz` (up to 120 s).
  - Creates a key with `docker compose exec gateway tierwise keys create --name smoke --budget 5 --rpm 600` and waits 3 s for the key refresh.
  - Sends 60 requests: easy, hard, a repeated prompt, and 10 streamed.
  - Asserts that:
    - All requests return 200 except the deliberate over-limit ones.
    - Prometheus's `/api/v1/query?query=sum(tierwise_requests_total)` is ≥ 60 (polled for up to 30 s).
    - Grafana's `/api/dashboards/uid/tierwise` returns 200.
    - Jaeger's `/api/services` lists `tierwise`.
    - **Pool reuse:** the fake upstream's `connections` ≤ 10 while `requests` ≥ 60, so connections are reused rather than opened per request (spec §7, deferred from M7a).
    - A key over its rpm gets a 429.
  - Exits non-zero on the first failure, printing the compose logs.
- Modify:
  - `.github/workflows/ci.yml`: a `compose-smoke` job: `docker compose -f deploy/docker-compose.yml up -d --build`, then `uv run python scripts/compose_smoke.py`, then `docker compose down -v` (in an `always()` step).
  - `README.md`: a "Run the full stack" section with `docker compose -f deploy/docker-compose.yml up --build`, the URLs (gateway :8000, Grafana :3000, Prometheus :9090, Jaeger :16686), how to switch from the fake upstream to real providers, and the note that `/metrics` is unauthenticated.
- [ ] **Step 1: Run the smoke check** locally if Docker Desktop is available (`docker compose -f deploy/docker-compose.yml up -d --build && uv run python scripts/compose_smoke.py`). Otherwise push and let the CI job run it. Expected: exit 0.
- [ ] **Step 2: Fix whatever the first real run surfaces.** Each fix that touches gateway code gets a failing test first. Compose and config fixes are verified by re-running the smoke check.
- [ ] **Step 3: Take a screenshot of the Grafana dashboard** after the smoke traffic, if Docker is local: open `http://localhost:3000/d/tierwise` in the browser pane and save it to `docs/benchmark/dashboard.png` for the README.
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check . && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "test: compose smoke check (M7 exit); docs: run the full stack"`
