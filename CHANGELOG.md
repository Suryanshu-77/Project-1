# Changelog

## 0.9.1: 2026-11-02

Hardening, from review findings a real user could hit:

- **Labels:** the flywheel's contrast model must be able to serve the stored request (capabilities and context, via the selector's own `fits`). A sample with no fitting model fails before any paid call, instead of after it.
- **Sampling:** streams that end without a usage chunk, as Ollama's always do, are now sampled. They are still not cached.
- **Startup:** the gateway starts even when no home directory can be resolved, as in containers that run under an arbitrary user ID.
- **Retrain:** a version folder appears only once it is complete, and a mismatched encoder gives a clear error.
- **`fetch-model`:** a download that grows past the pinned size is stopped.
- **The judge's score:** `true` is no longer accepted as 1, and `4.0` counts as 4.

## 0.9.0: 2026-11-02

**M9, the data flywheel.** The router can now learn from your own traffic.

- **Sample:** opted-in routed answers go into `shadow_samples`. That needs three opt-ins: `TIERWISE_STORE_PAYLOADS`, a key created with `--allow-shadow`, and an `auto*` request. Routes the router was unsure about are sampled 5× more.
- **Label:** `tierwise flywheel label` gets a contrast answer, then has a judge score the cheaper answer from 1 to 5, under a monthly budget. Both calls are retried, and empty, cut-off or tool-call contrast answers are never labeled.
- **Retrain:** `python -m training.retrain` trains a challenger. It's promoted only if all of these hold: at least 20 held-out rows of both labels, zero critical misroutes on the deployed router, no more hard prompts called easy by the classifier than the champion, and at least the champion's cost saved. On a cost tie, the better AUC wins. Versions are write-once in `models/router/versions/`, and promotion is a single atomic replace of `router.json`, which names its own weights. `/healthz` reports `router_version`.
- **CI:** a new `flywheel-e2e` job runs the whole loop against the fake upstream.

## 0.8.0 — 2026-11-02

First public release. Versions track milestones: 0.8.0 is M8.

- **M0–M1:** packaging, CI, OpenAI-compatible schema, Anthropic and OpenAI adapters, passthrough.
- **M2:** Gemini and Ollama, streaming, model registry, cost accounting, SQLite request log.
- **M3:** rules router, cache-aware selector, prefix affinity, `auto*` / `tier:*` models.
- **M3.5:** reasoning effort chosen together with the model.
- **M4:** learned router (MiniLM on ONNX + numpy classifier).
- **M5:** eval harness, Pareto curves, `BENCHMARK.md`, critical-set CI gate.
- **M6:** retry, failover, two-level circuit breakers, stream hold, opt-in semantic cache.
- **M7:** API keys, rate limits, monthly budgets; Prometheus, OpenTelemetry, Grafana, docker-compose with Postgres.
- **M8:**
  - `pip install tierwise` works without a checkout: the registry and router artifacts ship in the wheel.
  - `tierwise fetch-model` downloads the encoder at a pinned commit and checks its sha256.
  - A cancelled request now releases its budget reservation.
  - Releases publish to PyPI from a tag, and the Docker base image is pinned by digest.
  - `scripts/demo.py`.
