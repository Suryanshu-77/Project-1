# tierwise: design

*Cost-aware, self-hosted LLM gateway with a learned router.*
Date: 2026-11-01 · Status: revision 6. Rev 2 added critical-path latency, provider prompt caching, breaker granularity, stream buffering and the data flywheel. Rev 3 adds token-space truncation, non-blocking ONNX inference, pooled HTTP clients, and a queued write path. Rev 4 adds reasoning-effort selection (§3.5.1, milestone M3.5). Rev 5 adds packaging, model fetch and release (§3.10, milestone M8). Rev 6 moves the flywheel's contrast call and judge offline (§5.2, milestone M9). Rev 6.1 adds the 0.9.1 hardening fixes (§5.3).

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

### 3.5.1 Reasoning effort (milestone M3.5)
Reasoning ("thinking") tokens are billed as output, so an easy request that runs at a model's default effort wastes the most expensive tokens in the request. The selector therefore chooses **a model and an effort level together**.

**Normalized scale:** `minimal · low · medium · high · xhigh · max`. Each adapter translates a level into its provider's own control:

| Provider | Request field | Thinking tokens reported in |
|---|---|---|
| Anthropic | `output_config.effort` | included in `output_tokens` (not reported separately) |
| OpenAI | `reasoning_effort` | `completion_tokens_details.reasoning_tokens` (part of `completion_tokens`) |
| Gemini | `generationConfig.thinkingConfig.thinkingLevel` | `usageMetadata.thoughtsTokenCount` |
| Ollama | none | none |

When the chosen level equals the model's provider default, the adapter sends nothing.

**Registry.** A model may declare an `efforts` block. Without one, the model behaves as a single option at its default with zero thinking tokens.
```yaml
quality: 0.80                 # quality at the provider's default effort
effort_in_cache_key: true     # changing effort invalidates this provider's prompt cache
efforts:
  default: high               # what the provider does when no effort is sent
  levels:
    low:    {quality: -0.08, thinking_tokens: 150}
    medium: {quality: -0.03, thinking_tokens: 600}
    high:   {quality:  0.00, thinking_tokens: 2000}
```
- **Validated at startup:** levels must come from the normalized scale, `default` must be one of the listed levels, and every `quality` shift must keep the effective quality within 0–1.
- **Placeholder values:** the quality shifts and thinking-token estimates are uncalibrated until M5, which fits them from logged `tokens_reasoning` and benchmark scores.
- **`effort_in_cache_key`** defaults to `false`. It is `true` for Anthropic, where a top-level effort change invalidates the cached conversation.

**Selection.**
1. **Expand:** every model that passes the §3.5 filter becomes one option per declared level, or a single default option if it has no `efforts` block.
2. **Effective quality:** `clamp(model.quality + shift, 0, 1)`, where `shift` is the level's registry value for effort at or above the default, and `level.quality × difficulty / 0.5` for a level below it. Low effort costs little on easy requests and a lot on hard ones, which a fixed shift cannot express: no single fixed penalty met both M3.5 exit checks. This value is compared with the bar; the bar formula is unchanged.
3. **Estimated cost:** the §3.5 cost formula plus `level.thinking_tokens × price_out`.
4. **Order:** options that clear the bar by estimated cost, with ties going to the lower effort; then the remaining options by effective quality. The failover chain becomes a list of (model, effort) options.
5. **Affinity** remembers the (model, effort) pair that served each prompt.
   - For a model with `effort_in_cache_key: false`, any of its levels counts as warm.
   - For a model with `effort_in_cache_key: true`, only the same level counts as warm. Another level is priced cold: the whole history at the full input rate.
   - The selector therefore decides per turn whether a higher effort is worth the cache miss, so a long conversation tends to keep its effort level unless the bar requires more.
6. **`tier:*` targets** choose the cheapest option within the tier.

**Client-set effort.** `ChatRequest.reasoning_effort` (the OpenAI field name) is accepted on every request and always wins.
- For `auto` and `tier` targets, only options at exactly that level are considered. If none is eligible, the result is 422 `no_eligible_model`.
- For a concrete model id, the value is forwarded unchanged and the provider validates it.

**Visibility.**
- **Response header:** `X-Tierwise-Effort` carries the chosen level, or `default` when nothing was sent.
- **Log columns:** request rows gain `effort` and `tokens_reasoning`.
- **Usage:** gains `reasoning_tokens`, a subset of `output_tokens`, filled in where the provider reports it.
- **Report:** `scripts/costs.py` adds a breakdown by effort level.

**Schema upgrade.** SQLite's `create_all` never alters an existing table. At startup the request log compares `PRAGMA table_info(requests)` with the table definition and runs `ALTER TABLE … ADD COLUMN` for each missing nullable column. Existing rows are preserved and the operation is idempotent.

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

### 3.10 Packaging and release (milestone M8)
- **Installed package works on its own.** The default registry and the small router artifacts (`classifier.npz`, `router.json`, `tokenizer.json`, `NOTICE`) ship inside the wheel under `tierwise/data/`. When the configured `registry_path` or `router_dir` doesn't exist, the gateway uses the packaged copy. Explicit env overrides (`TIERWISE_REGISTRY`, `TIERWISE_ROUTER_DIR`) still win, and a missing explicit path is an error, not a silent fallback.
- **The 90 MB encoder is fetched, never bundled.** `tierwise fetch-model` downloads `onnx/model.onnx` from `sentence-transformers/all-MiniLM-L6-v2` at a pinned commit, verifies its sha256, and writes it atomically (temp file + rename) next to the router artifacts. Without it the gateway runs rules-only, as before. Startup never touches the network.
- **Client cancellation releases money.** A request cancelled mid-call (client disconnect) settles its budget reservation and is logged with status 499. Breaker probes are already released by `serve`.
- **Release.** A `v*` tag builds the sdist and wheel, smoke-installs the wheel in a clean venv, and publishes to PyPI through Trusted Publishing (no stored token), behind a `pypi` GitHub environment. Every CI run builds and smoke-installs the wheel too.
- **Demo.** `scripts/demo.py` sends prompts of rising difficulty through the compose stack and prints model, effort, reason and cost per prompt against an always-flagship baseline. The README shows that output and a Grafana screenshot.

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

### 5.2 Flywheel as built (milestone M9, rev 6)
Rev 6 moves the contrast call and the judge **offline**. The label is a property of the prompt, not of the moment it was sent, so nothing is lost. All label spend then sits in one job under one budget, and the gateway change is a single hook.
- **M9a: collection.**
  - **Gateway.** The sampler hooks in where complete text answers are cached (`keep()` in `routes.py`), so streams count and tool calls and cache hits never do.
    - A request is sampled only if all of these hold: `TIERWISE_STORE_PAYLOADS=true`, the key has `allow_shadow` (`tierwise keys create --allow-shadow`), and the request was routed with a decision.
    - The rate is `sample_rate(TIERWISE_SHADOW_RATE=0.02, confidence)`, raised for low-confidence routes.
    - A sample is one `shadow_samples` row (`status=pending`), written by the queued writer. Samples are droppable telemetry, counted in `tierwise_shadow_samples_total`.
  - **`tierwise flywheel label`, offline.** For each pending sample:
    1. Call the **contrast model**: the registry's `baseline_model` if a cheaper model answered, otherwise the cheapest `basic`-tier model.
    2. Ask the **judge** (`TIERWISE_JUDGE_MODEL`, default `baseline_model`) to score both answers from 1 to 5, each on its own merits (`tierwise-judge-v2`; until 0.10 it scored only the cheaper one). It sees them in an order seeded by the sample id and replies `{"a": n, "b": n}`.
    3. Set `needs_strong = cheap < 4 <= strong` (`flywheel.needs_strong`): the cheaper answer falls short where the stronger one doesn't. A prompt both models fail isn't labeled "needs strong": the flagship buys nothing there. Both scores are stored (`judge_score`, `judge_score_strong`). Retraining skips labels from before 0.10, which have no strong score. The base dataset's labels stay one-sided (Mixtral graded below 4), since its strong answer is GPT-4's, taken as correct.
    - Spend stops at `TIERWISE_SHADOW_BUDGET_USD` (default 5) per calendar month (UTC).
- **M9b: learning.**
  - **Retrain.** `training/retrain.py` (dependency group `retrain`: scikit-learn, no torch) merges labeled shadow rows with the base split when it exists. Shadow rows are split by prompt hash, using the base data's `split_of`. It trains a challenger with the same `fit` logic as `training/train.py`.
  - **Gate.** `evaluation/gate.py` scores the champion and the challenger on the same rows: the base test split if present, plus the shadow test split. It reports cost saved at `quality_target(rows)` (half the gap between all-cheap and all-strong quality, never below 95%; 0.9.2) and critical misroutes: how many of the 10 labeled hard prompts in `tests/router/labeled_prompts.jsonl` the deployed router would send to a basic model. The deployed router is the rules plus that classifier, so the count runs through the gateway's own `route()` and `select()`, the same check as the CI critical gate. Counting on the classifier alone would fail even the shipped router, at 7 of 10. The rules can only raise difficulty, so this count is a floor the rules guarantee. The gate (`decide`) therefore also refuses a challenger whose classifier alone calls more hard prompts easy than the champion's. Promotion also needs 20+ held-out rows with both labels. Each version's `router.json` names its own weights (`versions/<v>/classifier.npz`), so promotion is one atomic replace of `router.json`.
    - The challenger is promoted only if `should_promote(champion, challenger)` holds. At minimum: zero critical misroutes, and cost saved no worse than the champion's.
  - **Versions.** Every challenger is written to `models/router/versions/<UTC timestamp>/`. Promotion copies it over `router_dir`, and the replaced champion is kept in `versions/`. `router.json` carries `version`, which `/healthz` reports. This replaces the registry `router_version` of §5.1. The gateway loads a new router on restart.
- **Deferred:** the registry-drift raised rate, hot reload of the router, and a payload purge command. Samples contain prompts, which is why collection needs three opt-ins.
- **Exit check.** `scripts/flywheel_e2e.py` runs in CI against the fake upstream, whose fake judge scores hard-looking prompts 2 and easy ones 5. It sends traffic with a shadow key, then labels, retrains and gates. It passes when the gate prints a decision with its reasons, and `router_dir` changes if and only if the decision was to promote.

### 5.3 Hardening (0.9.1)
These fixes come from review findings that a real user could hit:
- **Contrast model.** It must be able to serve the stored request, using the selector's own capability and context check. It is `baseline_model` when a cheaper, non-flagship model answered, and the cheapest fitting `basic` model when any flagship answered: a flagship is the strong side of the label. A sample with no fitting contrast model fails before any paid call.
- **Sampling.** A stream that ends normally is sampled even without a usage chunk; Ollama never sends one. Caching still needs real usage.
- **Startup.** It never fails because the home directory can't be resolved (containers running under an arbitrary user ID).
- **Retrain.** A version folder appears only once complete. Retrain checks the encoder's dimension against `router.json`.
- **`fetch-model`.** It stops a download that grows past the pinned size.
- **The judge's score.** It must be a real integer, so `true` is rejected.

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
| M3.5 | Reasoning effort: normalized levels, per-model `efforts` in the registry, joint (model, effort) selection, effort-aware affinity, client override, `reasoning_tokens` accounting, log column migration | Easy prompts run at the lowest eligible effort and labeled hard prompts at `medium` or above; an Anthropic effort change is priced as a cache miss; an M3 database upgrades in place |
| M4 | Data prep, training, ONNX export, learned router | Beats rules on held-out test; p50 encoder+classifier < 10 ms |
| M5 | Eval harness, Pareto plot, registry calibration, `BENCHMARK.md` | Resume numbers produced |
| M6 | Retry, failover, two-level circuit breakers, stream hold-until-first-content, semantic cache | Fault-injection tests pass, including an early stream abort that fails over |
| M7 | Keys, budgets, rate limits, metrics, Grafana, OTel, docker-compose | `docker compose up` shows dashboard |
| M8 | Docs, demo, PyPI release | `pip install` of the built wheel serves a request with the packaged registry; release workflow publishes on a tag |
| M9 | Data flywheel: shadow sampling, offline judge, champion/challenger retrain | A challenger trained on shadow labels is promoted (or rejected) by the eval gate |

## 9. Open items (resolved during build, not blocking)
- Current prices and context windows for each model: verify against provider docs when writing `registry.yaml`.
- Effort controls (M3.5): confirm the OpenAI `reasoning_effort` levels and the Gemini `thinkingLevel` values per model against the official API references when writing each model's `efforts` block. Third-party sources disagree; for example, Gemini Pro accepts only `low`/`high` while Flash also accepts `minimal`/`medium`.
- Dataset availability and licenses for training and eval: verify at the start of M4. Fallback is a self-labeled set of about 500 prompts judged by a flagship model.
