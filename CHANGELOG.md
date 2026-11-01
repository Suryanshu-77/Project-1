# Changelog

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
