# tierwise — Design Spec

*Cost-aware, self-hosted LLM gateway with a learned router.*
Date: 2026-11-01 · Status: revision 3. Rev 2 added critical-path latency, provider prompt caching, breaker granularity, stream buffering and the data flywheel. Rev 3 adds token-space truncation, non-blocking ONNX inference, pooled HTTP clients, and a queued write path.

## 1. Problem and goal
Teams overpay by sending every request to a flagship model, or lose quality by defaulting to cheap ones. **tierwise** sits between applications and model providers and decides, per request, which model to use. It picks the **cheapest model expected to answer well enough**, and it measures whether that bet paid off.

### Success criteria
1. Any OpenAI-SDK client works by changing only `base_url` and `api_key`.
2. Four providers work through one API: Anthropic, OpenAI, Google Gemini, and Ollama (local).
3. A published benchmark (`BENCHMARK.md`) shows the cost vs. quality curve against an all-flagship baseline and a random baseline.
4. Zero routes of a request labeled *critical* to the cheapest model on the eval set.
5. The gateway survives one provider failing (failover), enforces per-key budgets, and exposes metrics in Grafana.
6. Median routing overhead is under 30 ms. Nothing on the critical path makes a network call except the chosen provider.

### Non-goals (v1)
Fine-tuning models, re-asking a stronger model when an answer looks weak, a hosted SaaS or billing, embeddings or image-generation endpoints, a full admin UI (Grafana and a CLI cover operations).

## 2. External API
`POST /v1/chat/completions` follows the OpenAI Chat Completions format: `messages`, `tools`, `temperature`, `max_tokens`, `stream`, `response_format`.

The `model` field selects the routing mode:
| value | behavior |
|---|---|
| `auto` | full routing (default) |
| `auto:cheap` / `auto:balanced` / `auto:quality` | routing with a shifted quality bar |
| `tier:basic` / `tier:moderate` / `tier:flagship` | skip difficulty prediction, cheapest healthy model in that tier |
| a concrete id, e.g. `claude-sonnet-5-5` | passthrough, no routing |

Prompt tags `[critical]` and `[easy]` in the last user message act as overrides and are stripped before forwarding.

Response headers: `X-Tierwise-Model`, `X-Tierwise-Provider`, `X-Tierwise-Reason` (override | rules | learned | affinity | fallback | cache), `X-Tierwise-Cost-USD`, `X-Tierwise-Request-Id`.

Other endpoints: `GET /v1/models` (registry plus virtual `auto*` / `tier:*` models), `GET /healthz`, `GET /metrics` (Prometheus).

Errors use the OpenAI error object `{error: {message, type, code}}`. Codes: `invalid_api_key` (401), `budget_exceeded` / `rate_limited` (429), `no_eligible_model` (422), `upstream_unavailable` (503).

## 3. Components
Each unit has one job, a typed interface, and can be tested in isolation.

### 3.1 Schema (`schema.py`)
Pydantic models `ChatRequest`, `Message` (text and image parts), `ToolDef`, `ChatResponse`, `Usage(input_tokens, cached_input_tokens, cache_write_tokens, output_tokens)`. Adapters map each provider's cache fields, such as Anthropic `cache_read_input_tokens` and OpenAI `prompt_tokens_details.cached_tokens`, so logged cost reflects real discounts, `StreamChunk`. All adapters translate to and from these types, so no provider-specific type leaks past `providers/`.

### 3.2 Provider adapters (`providers/`)
```python
class Provider(Protocol):
    name: str
    async def chat(self, req: ChatRequest, model_id: str) -> ChatResponse: ...
    def stream(self, req: ChatRequest, model_id: str) -> AsyncIterator[StreamChunk]: ...
```
- Implementations: `anthropic`, `openai` (also covers any OpenAI-compatible host through a `base_url`), `gemini`, `ollama`. Each uses `httpx` directly, which keeps dependencies small and the translation explicit.
- Adapters raise typed errors: `RetryableError` (429, 5xx, timeout), `FatalError` (4xx request problems), `AuthError`.
- Provider credentials come from environment variables only and are never logged.
- **Connection pooling.** Each provider gets one long-lived `httpx.AsyncClient`, created in the FastAPI `lifespan` startup hook, closed on shutdown, and injected into the adapter. No client is ever created per request, because that would mean a fresh TCP connection and TLS handshake every time (tens of milliseconds).
  - Limits are configurable per provider: `httpx.Limits(max_keepalive_connections=100, max_connections=250, keepalive_expiry=30)`.
  - Timeouts: `connect=5s`, `read` set per model (longer for flagship and streaming), `pool=2s`. A pool timeout counts as retryable.
  - HTTP/2 can be switched on per provider (`http2=True`) where the provider supports it.

### 3.3 Model registry (`registry.py`, `config/registry.yaml`)
```yaml
- id: claude-haiku-5-5
  provider: anthropic
  price_in: 1.00      # USD per 1M tokens (verify at build time)
  price_out: 5.00
  price_cache_read: 0.10   # cached input tokens; null if provider has no prompt caching
  price_cache_write: 1.25  # cost to write the cache (Anthropic); null where implicit/free
  cache_ttl_s: 300
  context: 200000
  caps: [tools, vision, json]
  quality: 0.62       # 0-1, calibrated from benchmark data (M5)
  tier: basic         # derived label for display; selection uses quality
```
Prices and quality scores are config, so new models need no code change. The registry is validated at startup.

### 3.4 Router (`router/`)
Output:
```python
RoutingDecision(difficulty: float, confidence: float, required_caps: set[str],
                source: Literal["override","rules","learned","fallback"], signals: dict)
```
Pipeline (`router.py`), first match wins:
1. **Override.** Explicit `tier:*`, a concrete id, or a `[critical]` / `[easy]` tag.
2. **Rules** (`rules.py`). A pure function over cheap features: token count, code or math blocks, high-stakes keywords, tool use, attachments, conversation depth. Very confident scores at either extreme return immediately.
3. **Learned** (`learned.py`).
   - **Input:** deterministic truncation only, with no LLM summarization: the system prompt plus the last two user turns. Order: last user turn first, then the previous user turn, then the head of the system prompt.
   - **Tokenization happens in token space, never on strings.** The model's own `tokenizer.json` is loaded with the Rust-backed HuggingFace `tokenizers` library. Each segment is encoded separately and token budgets are assigned by the priority order above. The result is joined into one sequence of at most 256 tokens including special tokens, which yields `input_ids`, `attention_mask` and `token_type_ids` (all three are BERT-family graph inputs). Truncating token IDs cannot split a multi-byte character or a sub-word.
   - **Encoder:** MiniLM exported to **ONNX** with dynamic batch and sequence axes, so a short prompt runs as a short tensor rather than being padded to 256. Padding is only needed when batching several requests. The ONNX graph returns token embeddings, so mean pooling over `attention_mask` and L2 normalization happen in numpy, matching `sentence-transformers`. A parity test checks this (§7).
   - **Runtime:** a single `onnxruntime.InferenceSession` on `CPUExecutionProvider`, created at startup with `graph_optimization_level=ORT_ENABLE_ALL` and fixed `intra_op_num_threads` (default 2), and warmed with a dummy run. `session.run` is blocking, so it runs in a dedicated small thread pool (`run_in_executor`) and never on the asyncio event loop. Target p50 5–10 ms. The execution provider is configurable, so a GPU can be used, but CPU is the default. `sentence-transformers` is used only offline, for training and export.
   - **Classifier:** logistic regression baseline, small MLP if it helps. Outputs P(strong model needed), which becomes `difficulty`. Confidence is the distance from the decision threshold.
4. **Fallback.** On any router error, use difficulty 0.7 (moderate-high) and mark it. The router never fails a request.

**No LLM judge on the request path.** Low confidence is handled without a network call: the selector raises the quality bar one step (§3.5). The judge moves offline into the data flywheel (§5.1), where it labels sampled traffic to train the next router version.

`required_caps` comes from the request itself: `tools` if tools are present, `vision` if image parts are present, `json` if `response_format` is set.

### 3.5 Selector (`selector.py`)
```python
def select(decision, req, registry, health, mode) -> list[ModelSpec]  # ordered candidates
```
1. Filter: has `required_caps`, has enough context for the estimated tokens, provider circuit not open.
2. Quality bar: `bar = f(difficulty, mode)`. When `confidence` is low the bar moves up one step (upward bias).
3. Eligible models have `quality >= bar`, sorted by **cache-aware estimated cost**:
   `cost(m) = cached_prefix_tokens(m) × price_cache_read + uncached_tokens × price_in + est_output × price_out`
   - `cached_prefix_tokens(m)` comes from the **prefix-affinity table**: a hash of the stable prompt prefix (system prompt, tools, earlier turns) mapped to the models that served it and when. A model gets cached pricing only if it served the same prefix within its `cache_ttl_s`. Otherwise the prefix is priced at the normal input rate (plus `price_cache_write` where it applies).
   - So in a long conversation, a warm cache on the current model can make it cheaper than a "cheaper" cold model. Switching models mid-conversation is then penalized correctly, rather than by a fixed token-count threshold.
   - **Affinity rule:** when the router's bar allows the incumbent model and its cache-aware cost is within 10% of the cheapest option, keep the incumbent (`reason=affinity`). This avoids flip-flopping and the cache misses it causes.
4. The cheapest model comes first. The rest form the **failover chain**, ending with models above the bar.
5. An empty list raises `no_eligible_model`.

### 3.6 Reliability (`reliability/`)
- **Retry.** Up to 2 retries with exponential backoff and jitter on `RetryableError`, for the same model.
- **Failover.** After retries are exhausted, try the next candidate from the selector. Never fail over on `FatalError`.
- **Circuit breakers at two levels.** Both feed the selector.
  - **Per `(provider, model)`:** trips on 429, 5xx and timeouts for that model. Opus being rate-limited does not block Haiku.
  - **Per provider:** trips only on provider-wide signals, such as connection refused, DNS/TLS failure, auth errors, or several models of that provider tripping together.
  - **Thresholds:** 5 consecutive failures within 60 s, a 30 s cooldown, then one half-open probe. All configurable.
- **Streaming: hold until first content.** The gateway buffers upstream events and sends nothing, not even the 200 header, until the first non-empty content or tool-call delta arrives, capped at 3 s or 4 KB. An abort, refusal or error inside that window can still fail over invisibly. After the first content is sent, failures end the stream with an OpenAI-style error event and are logged. This buffering adds only the time spent on the empty preamble events to time-to-first-token, and that is tracked as a metric.

### 3.7 Semantic cache (`cache.py`)
- **Key:** embedding of the normalized messages, scoped by routing mode and required caps.
- **Hit:** cosine similarity of at least 0.95 (configurable) and an entry younger than its TTL.
- **Skipped when:** `temperature > 0.3`, tools are present, the client sends `Cache-Control: no-cache`, or the mode is a passthrough.
- **Storage:** an in-memory vector index (numpy) by default, Redis when configured.
- A cache hit is still logged and counts toward savings.

### 3.8 Auth, budgets, rate limits (`api/auth.py`, `budgets.py`)
- Gateway API keys (`tw-...`) are stored as SHA-256 hashes and managed with the CLI `tierwise keys create|list|revoke`.
- Each key has a monthly USD budget and a requests-per-minute limit (token bucket).
- **Pre-check:** reject if spent + estimated cost exceeds the budget. **Post-debit:** charge the actual cost.
- Spend is checked against an **in-memory per-key counter**, loaded from the database at startup and updated on each debit. The request path never reads the database. Debits are written through the queued writer (§3.9).
- With several gateway replicas, counters move to Redis (`INCRBYFLOAT`), and Postgres keeps the durable ledger.

### 3.9 Store and telemetry (`store.py`, `telemetry.py`)
- SQLAlchemy (async) over SQLite via `aiosqlite` by default, and Postgres via `asyncpg` in compose.
- **Decoupled write path.** Request handlers never write to the database. They put records on a bounded `asyncio.Queue`, and one background writer task drains it in batches (up to 200 rows or every 250 ms, whichever comes first) in a single transaction.
  - A single writer means SQLite never has two writers competing for its lock.
  - At startup SQLite gets `PRAGMA journal_mode=WAL` (reads never block the writer), `synchronous=NORMAL` and `busy_timeout=5000`.
  - When the queue is full, **telemetry rows are dropped** and counted (`tierwise_telemetry_dropped_total`), so the request still succeeds. **Budget debit rows are never dropped**: they use a separate unbounded queue and are retried.
  - On shutdown the writer drains its queue before the process exits.
- Tables:
  - `requests`: id, ts, key_id, mode, source, difficulty, confidence, model, provider, tokens_in, tokens_cached, tokens_out, prefix_hash, cost_usd, baseline_cost_usd, latency_ms, cache_hit, failover_count, status.
  - `api_keys`
  - `usage_monthly`
- Prompts and outputs are **not** stored by default. `store_payloads: true` opts in and is needed for collecting training data.
- Prometheus metrics: `tierwise_requests_total{model,source,status}`, `tierwise_cost_usd_total`, `tierwise_cost_saved_usd_total`, `tierwise_latency_seconds` (histogram), `tierwise_cache_hits_total`, `tierwise_failovers_total`, `tierwise_breaker_open{provider,model}`, `tierwise_ttft_buffer_seconds`, `tierwise_shadow_samples_total`.
- OpenTelemetry spans: route, select, cache, provider call.
- The Grafana dashboard ships as provisioned JSON.

## 4. Data flow (non-streaming)
auth → rate-limit → cache lookup → route → select → budget pre-check → call candidate[0] (retry, then failover) → normalize → cache store → log + metrics → respond with headers.

## 5. Training and evaluation
- **Data** (`training/prepare_data.py`). Public pairwise preference data (LMSYS Arena human-preference / RouteLLM-released datasets) becomes a binary label "strong model needed". Exact datasets and licenses are confirmed at the start of M4. Optionally add real gateway logs collected with `store_payloads`.
- **Train** (`training/train.py`). Split train/val/test by prompt (deduplicated, no leakage). Model types: logistic regression baseline, then an MLP. The saved artifact (`router_model.joblib` plus the embedding model name) is loaded by `learned.py`.
- **Evaluate** (`eval/run_eval.py`). Use a RouterBench-style dataset with precomputed per-model outputs and quality scores, so the evaluation needs no API spend. Sweep thresholds and plot cost vs. quality for: all-flagship, all-cheapest, random mix, rules-only, learned, learned+cache-aware selection. Report the **cost saved at 95% of flagship quality**, plus a critical-set misroute count.
- **Calibration.** The `quality` values in the registry come from the same benchmark.

### 5.1 Data flywheel (drift control)
A static `router_model` decays as traffic and the registry change, so tierwise keeps collecting labels without touching what the user receives.
- **Shadow sampling, not traffic diversion.** For a configurable sample, default 2% of eligible requests, the user gets the routed answer as normal. A background worker then sends the same request to a contrasting model: the flagship if the request was routed cheap, a cheaper eligible model if it was routed high. Both answers go to the **offline judge**, a flagship model with a pairwise-grading rubric, which records whether the cheaper answer was good enough. That record is the ground-truth label.
  - Why not just route some `auto:cheap` traffic to the flagship? A single answer gives no comparison, so it yields no label. It would also override the cost mode the user chose.
- **Scope and guardrails:**
  - Low-confidence routes are sampled at a higher rate.
  - Opt-in only: requires `store_payloads: true` and a per-key `allow_shadow` flag.
  - Capped by a monthly shadow budget, run asynchronously, and never on the request path.
  - Skipped for tool-call and side-effecting requests.
- **Retraining loop (champion/challenger):** `training/retrain.py` merges fresh shadow labels with the base dataset and trains a challenger. It evaluates the challenger against the current champion on a fixed held-out set plus recent shadow data. The challenger is promoted (a versioned artifact plus a registry `router_version`) only if it is at least as good on quality-at-cost **and** has zero critical misroutes.
- **Registry drift:** adding a model triggers shadow sampling for that model at a raised rate, until it has enough labels to estimate its `quality` score.

## 6. Security and privacy
- Provider keys are read from env only and redacted in logs and traces.
- Gateway keys are stored hashed and compared in constant time.
- Payload logging is off by default.
- The gateway binds to localhost by default. Compose exposes it deliberately.
- Request size limits apply.

## 7. Testing strategy
- **Unit:** rules features, selector filtering and ordering, breaker state machine, token-bucket, budget math, cache threshold logic, schema translation per adapter.
- **Encoder:**
  - Token-budget truncation with emoji, CJK and very long inputs: never more than 256 tokens, special tokens intact, priority order respected.
  - ONNX vs. `sentence-transformers` embedding parity (cosine ≥ 0.999).
  - An event-loop-lag check confirms inference does not block other requests.
- **Concurrency:** a load test (`locust` or an `httpx` burst of 200 concurrent requests) against fake providers. It must produce no `database is locked` errors and no dropped debit rows, with a stable p50 routing overhead below 30 ms, and confirms the connection pool is reused (handshake count ≈ pool size, not request count).
- **Adapter contract:** recorded HTTP fixtures with `respx`, including error mapping (429→Retryable, 400→Fatal).
- **Integration:** the FastAPI `TestClient` drives the full pipeline against fake providers, covering failover, budget 429s and streaming.
- **Live smoke** (manual, opt-in): one prompt per provider through the OpenAI SDK.
- **Eval regression:** CI runs a small fixed eval subset and fails if the critical misroute count is above zero.
- **CI:** ruff, mypy (strict on `src/`), pytest with coverage.

## 8. Milestones
| # | Scope | Exit check |
|---|---|---|
| M0 | Packaging, lint/type/test tooling, CI, README | CI green |
| M1 | Schema, Anthropic and OpenAI adapters, passthrough endpoint (explicit model) | OpenAI SDK call works for both providers |
| M2 | Gemini and Ollama adapters, streaming, registry, cost accounting including cached-token pricing, SQLite log | 4 providers; logged cost matches provider-reported usage |
| M3 | Rules router, selector with cache-aware cost and prefix affinity, overrides, `auto*` / `tier:*` modes | Labeled prompts route sensibly; long conversation stays on its warm model |
| M4 | Data prep, training, ONNX export, learned router | Beats rules on held-out test; p50 encoder+classifier < 10 ms |
| M5 | Eval harness, Pareto plot, registry calibration, `BENCHMARK.md` | Resume numbers produced |
| M6 | Retry, failover, two-level circuit breakers, stream hold-until-first-content, semantic cache | Fault-injection tests pass, including an early stream abort that fails over |
| M7 | Keys, budgets, rate limits, metrics, Grafana, OTel, docker-compose | `docker compose up` shows dashboard |
| M8 | Docs, demo, PyPI release | Public, installable |
| M9 | Data flywheel: shadow sampling, offline judge, champion/challenger retrain | A challenger trained on shadow labels is promoted (or rejected) by the eval gate |

## 9. Learning hand-offs (`TODO(human)`)
- M1: the `Provider` protocol method shape
- M3: rule weights and thresholds
- M4: the confidence-escalation policy
- M6: failover candidate ordering

## 10. Open items (resolved during build, not blocking)
- Current prices and context windows for each model: verify against provider docs when writing `registry.yaml`.
- Dataset availability and licenses for training and eval: verify at the start of M4. Fallback is a self-labeled set of about 500 prompts judged by a flagship model.
