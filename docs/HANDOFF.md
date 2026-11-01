# tierwise: project handoff

*State as of 2026-11-02: M8 merged and released; M9 being planned on `feat/flywheel`. Start a new chat by giving Claude this file. The authoritative
detail lives in the repo: the spec, the per-milestone plans, `BENCHMARK.md`, `README.md` and
`git log`. This file is the map.*

## What tierwise is

A **self-hosted, OpenAI-compatible LLM gateway** that sends each request to the cheapest model
expected to answer it well, across Anthropic, OpenAI, Gemini and local Ollama. It also picks a
**reasoning-effort level** together with the model. It is built as a resume-grade project:

- learned router
- measured benchmark
- reliability: retry, failover and circuit breakers
- semantic cache
- multi-tenant keys, budgets and rate limits
- Prometheus, Grafana and OpenTelemetry
- docker-compose stack

- **Repo:** https://github.com/Suryanshu-77/Project-1 (local:
  `C:\Users\Surya\OneDrive\Desktop\Claude Projects\Project 1`)
- **Spec (the authority):** `docs/superpowers/specs/2026-11-01-tierwise-design.md` (rev 4)
- **Plans, one per milestone:** `docs/superpowers/plans/2026-11-01-tierwise-*.md`
- **Resume headline (from `BENCHMARK.md`):** on 3,943 held-out prompts, the deployed router
  keeps **95% of GPT-4's quality at 44% of its cost (56% saved)**, against 37% for random
  routing.

## Milestones

| # | What | PR | Status |
|---|---|---|---|
| M0–M1 | Packaging, CI, schema, Anthropic + OpenAI adapters, passthrough | #2 | merged |
| M2 | Gemini + Ollama, streaming, registry, cost accounting, SQLite log | #3 | merged |
| M3 | Rules router, cache-aware selector, prefix affinity, `auto*`/`tier:*` | #4 | merged |
| M3.5 | Reasoning effort chosen jointly with the model | #5 | merged |
| M4 | Learned router: MiniLM on ONNX + numpy classifier (trained on routellm/gpt4_dataset) | #6 | merged |
| M5 | Eval harness, Pareto curves, `BENCHMARK.md`, critical-set CI gate | #7 | merged |
| M6 | Retry, failover, two-level breakers, stream hold, opt-in semantic cache | #8 | merged |
| M7a | API keys + CLI, token-bucket rate limits, reserve/settle monthly budgets | #9 | merged |
| M7b | Prometheus `/metrics`, OTel spans, Grafana dashboard, compose + Postgres | #10 | merged (all 5 CI checks green, incl. Postgres + `compose-smoke`) |
| M8 | Packaged data, `tierwise fetch-model`, cancellation fix, release workflow, demo, docs | #11 | merged; **tierwise 0.8.0 is live on PyPI**, and the repo is public |
| M9 | Data flywheel (spec §5.2): M9a collection, M9b learning | — | planned |

`main` is at `e453895`: M0 through M7 are all merged. Version is now 0.8.0, and versions track milestones.

## Architecture (the request path)

```
auth (Bearer tw-…, 401) → rate limit (token bucket, 429 + Retry-After)
→ route(): override tag → rules → learned router (when rules confidence < 0.8),
           difficulty = max(rules, learned) → fallback 0.7 on error
→ semantic cache lookup (opt-in; last user turn by meaning, everything else exact)
→ select(): (model, effort) options, quality bar, cache-aware cost, affinity
→ budget reserve (estimate) → serve(): breakers + retries + failover (≤3 models)
   streams: hold until first content (3 s / 4 KB), so early aborts fail over
→ settle actual cost (debits never dropped) → log + metrics + spans
```

| Area | Files |
|---|---|
| API | `src/tierwise/api/{app,routes,streaming,auth,errors}.py` |
| Routing | `router/{rules,features,learned,encoder,classifier,router}.py`, `selector.py`, `options.py`, `affinity.py` |
| Reliability | `reliability/{retry,breaker,failover}.py` |
| Cache, money, persistence | `cache.py`, `budgets.py`, `keys.py`, `store.py` |
| Observability | `metrics.py`, `telemetry.py` |
| CLI | `__main__.py` (`tierwise` serves; `tierwise keys create\|list\|revoke`; `tierwise fetch-model`) |
| Packaging | `config.py` resolvers (explicit → repo default → packaged `tierwise/data/`), `fetch.py`, `scripts/wheel_smoke.py`, `.github/workflows/release.yml` |
| Offline | `training/`, `evaluation/` |
| Deploy | `deploy/` (compose, Prometheus, Grafana), `Dockerfile`, `scripts/{fake_upstream,compose_smoke,bench_router,smoke,costs}.py` |

## Running things

```bash
uv sync                                   # dev only (what CI runs)
uv sync --group train                     # + torch/sklearn/etc. (training, eval, ONNX parity tests)
uv run pytest -q                          # 492 pass, 9 Postgres variants skip locally
uv run ruff check . && uv run mypy src    # both clean
docker compose -f deploy/docker-compose.yml up -d --build   # Docker Desktop is installed
uv run python scripts/compose_smoke.py    # M7 exit check
```

- **Postgres store tests:** set `TIERWISE_TEST_PG_URL=postgresql+asyncpg://tierwise:tierwise@localhost:5433/tierwise`
  with a `postgres:16` container on port 5433.
- **Model artifacts:** `models/router/model.onnx` (90 MB) is gitignored; regenerate it with
  `uv run python -m training.export_onnx` (needs the train group). Without it the gateway runs
  rules-only and has no cache.
- **Docker CLI on PATH in Git Bash:** `export PATH="$PATH:/c/Program Files/Docker/Docker/resources/bin"`.
- **The compose stack may still be running locally.** Stop it with
  `docker compose -f deploy/docker-compose.yml down` (add `-v` to wipe its data).
- **Grafana:** http://localhost:3000/d/tierwise. It is view-only, and every port is bound to
  127.0.0.1.

## How we work (the user's preferences)

- **The user** is learning as they go, and the Learning output style is on. Each milestone
  has one `TODO(human)` hand-off of 2–10 lines, followed by an insight on what they wrote.
  The user has written:
  - `expand` (M3.5)
  - `to_decision` (M4)
  - `cost_saved_at`, with interpolation (M5)
  - `next_candidates`, other-providers-first (M6)
  - `TokenBucket.take` (M7a)
  - the Grafana savings PromQL (M7b)
  - the `CancelledError` handler that releases the reservation (M8)
- **Per milestone:**
  1. The superpowers flow: brainstorming → spec → `writing-plans` → **Native** execution
     with `executing-plans` (ledger under `.superpowers/sdd/<plan>/`, `task-start` and
     `task-done` scripts).
  2. TDD on every step.
  3. A final whole-branch review by a fresh **Opus** subagent, then **one** fix pass, in
     which each Critical or Important finding gets a test that fails first.
  4. Rulings and deferred minors are reported to the user.
- **Integration:** the user says "Push and open PR", then "Turn on Auto-fix", then "merge
  it" once CI is green. PRs use the mattpocock `pr` template (Summary, Evidence, Merge
  Danger). Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, and
  PR bodies end with the Claude Code line.
- **Gotchas learned the hard way:**
  - Never chain `git commit` after a check piped into `tail`, or after a check on a separate
    line. Use `ruff check -q . && mypy src && pytest … && git commit`.
  - Bash heredocs with complex quoting fail. Write edit scripts to the scratchpad with the
    Write tool and run them with `python`.
- **Permissions:** the user permits installing tools the project needs. Ask before
  accepting license terms.

## Open threads and backlog

- **Next:** M9a, then M9b. Plans: `docs/superpowers/plans/2026-11-02-tierwise-m9a.md` and `-m9b.md`.
- **Known weaknesses worth knowing (each documented):**
  - **Out of distribution (SPROUT):** the learned router loses to rules at 98–99% quality.
    It learned *Mixtral's* weak spots, which motivates M9.
  - **`to_decision`:** gives about 42–62% of traffic low confidence, which raises the
    quality bar.
  - **Semantic cache:** opt-in because embeddings miss negation (safe vs unsafe: 0.99).
  - **Budgets:** can overshoot by the cost of requests in flight (the output estimate is
    512 tokens when `max_tokens` is unset). They are enforced only with
    `TIERWISE_REQUIRE_KEY=true`.
  - **Registry:** quality values and prices are uncalibrated placeholders (calibration is
    M9). Prices are marked "VERIFY".
  - **Datasets:** RouterBench and SPROUT have no stated license; only aggregates from SPROUT
    are published.
- **Deferred minors** (recorded in each milestone's final message; none are blocking). The
  main themes:
  - Log and metrics provenance is coarse: source labels on pre-routing errors, and lost
    rules-vs-learned attribution under affinity.
  - Cache lookup is O(n) on the event loop.
  - Smoke-script and dashboard polish.
