# Contributing

Issues and pull requests are welcome. For anything larger than a small fix, open an issue first
so the approach can be agreed before you write the code.

## Setup

tierwise uses [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --group retrain       # dev tools plus scikit-learn, as CI runs it
uv sync --group train         # optional: torch and friends, for training and ONNX parity tests
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Postgres store tests are skipped unless `TIERWISE_TEST_PG_URL` points at a running Postgres 16
(for example `postgresql+asyncpg://tierwise:tierwise@localhost:5433/tierwise`).

## Where things live

| Path | What |
|---|---|
| `src/tierwise/api/` | the FastAPI app and the `/v1/chat/completions` flow |
| `src/tierwise/router/` | rules scorer, learned router, `route()` |
| `src/tierwise/selector.py` | the cheapest (model, effort) that clears the quality bar |
| `src/tierwise/reliability/` | retries, circuit breakers, failover |
| `src/tierwise/providers/` | Anthropic, OpenAI, Gemini and Ollama adapters |
| `src/tierwise/flywheel.py`, `training/retrain.py`, `evaluation/gate.py` | the data flywheel |
| `evaluation/`, `experiments/`, `BENCHMARK.md` | the routing benchmark and the flywheel experiment |
| `deploy/` | the compose demo stack and its Grafana dashboard |
| `docs/design.md` | the design document: read the relevant section before a larger change |

## Guidelines

- **Tests first.** A bug fix comes with a test that fails without it.
- **No real API calls in tests.** Use the fake upstream in `scripts/fake_upstream.py`.
- **Routing changes need evidence.** If a change can move routing decisions, re-run the
  evaluation (`uv run python -m evaluation.run_eval`) and report the before and after numbers
  in the pull request. The critical-set gate must still pass.
- **Benchmark claims need reproducible steps,** in `BENCHMARK.md` or under `experiments/`.
- **Changelog.** User-visible changes get a line in `CHANGELOG.md`.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
