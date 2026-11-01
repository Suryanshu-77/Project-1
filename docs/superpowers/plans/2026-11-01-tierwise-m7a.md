# tierwise M7a Implementation Plan: API Keys, Budgets and Rate Limits

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A multi-tenant gateway. Clients authenticate with `tw-…` keys created by a CLI, and each key gets a monthly USD budget and a requests-per-minute limit. Spend is checked in memory before each call and debited after it, through a write path that never drops a debit.

**Architecture:**
- **`store.py`** gains two tables, `api_keys` and `usage_monthly`, plus key CRUD. Debits go on a separate, unbounded accumulator that the existing writer flushes as upserts, and that is kept for retry when a write fails.
- **`keys.py`** generates and hashes keys and holds them in memory, refreshed from the database every 30 s.
- **`budgets.py`** holds the token-bucket rate limiter and the budget counters. A request **reserves** its estimated cost before calling and **settles** the actual cost afterwards, so concurrent requests can't overshoot a budget together.
- **`api/auth.py`** turns the `Authorization: Bearer` header into a key, or a 401.
- **`__main__.py`** gains `tierwise keys create|list|revoke`.

**Tech Stack:** no new dependencies (SQLAlchemy dialect upserts, `secrets`, `hashlib`, `hmac`).

**Spec:** `docs/superpowers/specs/2026-11-01-tierwise-design.md` §2 (error codes), §3.8, §3.9 (tables, never-dropped debits), §4 (flow order), §6 (hashed keys, constant-time compare), §7 (concurrency load test)

## Global Constraints
- Everything in the M3–M6 plans' Global Constraints still holds (ruff 100, strict mypy on `src/`, CI without the train group or `model.onnx`).
- **Keys:** `tw-` + `secrets.token_urlsafe(32)`. Only the SHA-256 hex of the full key is stored. The plaintext is printed once, at creation, and never logged. A row keeps the first 10 characters (`tw-` + 7) as `prefix` for display.
- **Auth:** `Authorization: Bearer <key>`. A missing, unknown or revoked key gets **401** `invalid_api_key`, type `invalid_request_error`. The hash check uses `hmac.compare_digest`. `/healthz` stays open. `/v1/models` and `/v1/chat/completions` require a key when `require_api_key` is true.
- **Rate limit:** a token bucket per key, with capacity `rpm` that refills at `rpm / 60` tokens per second. One request costs one token. When it is empty: **429** `rate_limited`, with a `Retry-After` header holding the whole seconds until a token is available (at least 1). A key with `rpm` None is unlimited.
- **Budget:**
  - **Period:** the UTC calendar month (`"YYYY-MM"`).
  - **Pre-check (reserve):** `spent + reserved + estimate > budget` gets **429** `budget_exceeded`.
  - **Post-debit (settle):** the reservation is released and the actual `cost_usd` is added. An error releases the reservation without a debit. Cache hits cost 0.
  - A key with budget None is unlimited.
- **Estimate:** `selector.estimated_cost(spec, estimate_tokens(req), estimate_output(req), 0) + thinking_tokens × price_out / 1e6` for the first candidate, the same formula the selector ranks by.
- **Flow order (§4):** auth → rate limit → cache lookup → route → select → budget pre-check → call.
- **Debits are never dropped:** they are accumulated per `(key_id, month)` in memory and flushed by the writer as upserts. On a failed write the amounts stay and are retried on the next flush. `stop()` flushes them before exiting.
- **The request path never reads the database:** keys and month spend are loaded at startup and refreshed in the background.
- **Settings:** `require_api_key: bool = False` (alias `TIERWISE_REQUIRE_KEY`) and `keys_refresh_s: float = 30.0`.

## Decisions this plan makes
- **Keys are refreshed every 30 s, not looked up on a miss.** A miss-triggered database read would let invalid keys hammer the database. Cost if wrong: a new or revoked key takes up to 30 s to take effect (documented).
- **Auth is off by default** (`require_api_key=false`, localhost bind); compose turns it on in M7b. Cost if wrong: one flag.
- **Reserve and settle instead of check and debit:** with a plain check, 50 concurrent requests on a key with $0.10 left could all pass the check. Cost if wrong: none (stricter than the spec).
- **Redis counters (multi-replica) are deferred:** the spec only needs them with several replicas, and M7b's compose runs one. Cost if wrong: a Redis backend added behind `Budgets` later.

## Review Focus
1. **Concurrent requests near the budget:** 20 requests in flight on a key with room for 3 must not all pass. Reservations make only about 3 pass. Test in Task 4.
2. **A streaming request that the client abandons:** the reservation is released, and a 499 with no usage debits nothing. Test in Task 4.
3. **Month rollover while running:** a request on the 1st of the month sees a fresh budget, even though the old month's counter is still in memory. Test in Task 4.
4. **A database outage during flush:** debits survive in memory and land on the next successful flush, with no double counting. Test in Task 1.
5. **The `Authorization` header in any casing, with extra spaces, or with a non-`Bearer` scheme:** the tolerable forms work and everything else is a 401, never a 500. Test in Task 2.

---

### Task 1: Key and usage tables, never-dropped debits

**Files:**
- Modify: `src/tierwise/store.py`
- Test: `tests/test_store.py` (extend)

**Interfaces:**
- Produces:
  - Table `api_keys`: `id` String PK (`uuid4().hex[:12]`), `name` String not null, `key_hash` String unique not null, `prefix` String not null, `monthly_budget_usd` Float nullable, `rpm` Integer nullable, `created_ts` Float not null, `revoked_ts` Float nullable.
  - Table `usage_monthly`: `key_id` String, `month` String (`"YYYY-MM"`), `spent_usd` Float not null, `requests` Integer not null. The primary key is `(key_id, month)`.
  - `@dataclass(frozen=True) class KeyRow`, mirroring the `api_keys` columns.
  - `RequestLog` methods:
    - `async create_key(name: str, key_hash: str, prefix: str, budget: float | None, rpm: int | None) -> KeyRow`
    - `async list_keys() -> list[KeyRow]`
    - `async revoke_key(key_id: str) -> bool`: False if the key is unknown or already revoked.
    - `async load_month_spend(month: str) -> dict[str, float]`
    - `submit_debit(key_id: str, month: str, cost_usd: float) -> None`: synchronous, never raises, never drops.
    - `async fetch_usage() -> list[tuple[str, str, float, int]]` (for tests)
    - `pending_debits: dict[tuple[str, str], tuple[float, int]]` (read-only view, for tests)
  - The writer loop flushes debits on every iteration, as one `INSERT … ON CONFLICT (key_id, month) DO UPDATE SET spent_usd = spent_usd + excluded.spent_usd, requests = requests + excluded.requests`, using `sqlalchemy.dialects.sqlite.insert` or `postgresql.insert` depending on the engine dialect. Amounts are removed from `pending_debits` only after the transaction commits. The loop exits only once both the queue and `pending_debits` are empty.
- [ ] **Step 1: Write the failing tests** (`tmp_path` SQLite).
  - `test_create_list_revoke_key`: created → listed with `revoked_ts` None; `revoke_key` → True, then a second call → False; `list_keys` shows `revoked_ts` set.
  - `test_duplicate_key_hash_rejected`: `sqlalchemy.exc.IntegrityError`.
  - `test_debits_aggregate_per_key_and_month`: debits of 0.01, 0.02 (key a, "2026-11") and 0.05 (key a, "2026-12"), then `stop()` → `fetch_usage()` has (a, 2026-11, 0.03, 2) and (a, 2026-12, 0.05, 1).
  - `test_debits_accumulate_across_flushes`: debit, wait for a flush, debit again → `spent_usd` is the sum.
  - `test_debits_survive_failed_flush`: monkeypatch the upsert to raise once → the amount stays in `pending_debits`, the next flush writes it exactly once, and the total is correct.
  - `test_debits_never_dropped_when_telemetry_queue_full`: `RequestLog(max_queue=1)`, fill the queue, then debit → `dropped` grows for telemetry only, and the debit is written.
  - `test_load_month_spend`.
- [ ] **Step 2: Run** `uv run pytest tests/test_store.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: api_keys and usage_monthly tables; debits are never dropped"`

### Task 2: Keys and authentication

**Files:**
- Create: `src/tierwise/keys.py`, `src/tierwise/api/auth.py`
- Modify: `src/tierwise/config.py` (`require_api_key`, `keys_refresh_s`), `src/tierwise/api/app.py` (load keys at startup and start the refresh task), `src/tierwise/api/routes.py` (authenticate first; `recorder.key_id`), `src/tierwise/api/errors.py` (`InvalidApiKey`)
- Test: `tests/test_keys.py`, `tests/api/test_auth_route.py`; `tests/api/helpers.py` gains `make_key(app, **kw) -> str`.

**Interfaces:**
- Consumes: Task 1's `KeyRow` and `RequestLog` key methods.
- Produces:
  - `KEY_PREFIX = "tw-"`
  - `generate_key() -> str`
  - `hash_key(key: str) -> str` (sha256 hex)
  - `class KeyStore`:
    - `replace(rows: list[KeyRow]) -> None`: active keys only.
    - `find(key: str) -> KeyRow | None`: hash, dict lookup, then `hmac.compare_digest` on the stored hash.
    - `async refresh_forever(log: RequestLog, interval_s: float) -> None`: a background task, cancelled at shutdown. A failed refresh is logged and the old set is kept.
  - `def authenticate(header: str | None, store: KeyStore, required: bool) -> KeyRow | None`:
    - Accepts `Bearer <key>` with any casing of `bearer` and surrounding whitespace.
    - Raises `InvalidApiKey` (401) when `required` and the key is missing or unknown.
    - When not required, it still resolves a valid key (so spend is tracked) and returns None otherwise.
  - `InvalidApiKey(GatewayError)`: 401, `invalid_api_key`, "Invalid or missing API key."
- [ ] **Step 1: Write the failing tests.**
  - Unit:
    - `test_generated_key_shape`: it starts with `tw-`, has length ≥ 40, and two keys differ.
    - `test_find_only_active_keys`.
    - `test_authenticate_variants` (parametrized):
      - `"Bearer K"`, `"bearer K"` and `"  Bearer   K  "` → found.
      - `None`, `"Basic K"`, `"Bearer"`, `"Bearer wrong"` → raise when required, None when not.
  - API:
    - `test_missing_key_is_401_when_required`: OpenAI error shape, `code == "invalid_api_key"`, and no upstream call.
    - `test_valid_key_logs_key_id`: the log row's `key_id` equals the key's id.
    - `test_revoked_key_rejected_after_refresh`: revoke, then call `KeyStore.replace` with fresh rows (no 30 s wait) → 401.
    - `test_models_requires_key_healthz_does_not`.
    - `test_auth_off_by_default_still_tracks_valid_key`.
- [ ] **Step 2: Run** `uv run pytest tests/test_keys.py tests/api -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: gateway API keys with hashed storage and bearer auth"`

### Task 3: Rate limiting (hand-off: the token bucket)

**Files:**
- Create: `src/tierwise/budgets.py` (rate-limit part)
- Modify: `src/tierwise/api/routes.py` (check right after authentication), `src/tierwise/api/errors.py` (`RateLimited` 429 `rate_limited` with a `retry_after: int` attribute; the error handler sets the `Retry-After` header)
- Test: `tests/test_budgets.py`, `tests/api/test_limits_route.py`

**Interfaces:**
- Produces:
  - `class TokenBucket`:
    - `__init__(self, rpm: int, clock: Callable[[], float] = time.monotonic)`. It starts full.
    - `take(self) -> float`. **`TODO(human)`:** refill by elapsed time, then take one token. Returns `0.0` if a token was taken, otherwise the seconds until one will be available. Nothing is taken in that case.
  - `class RateLimiter`:
    - `__init__(self, clock=time.monotonic)`
    - `check(self, key: KeyRow) -> None`: raises `RateLimited(retry_after=max(1, ceil(wait)))`. It does nothing for `rpm` None. Buckets are created per key id, and recreated if the key's `rpm` changes.
- [ ] **Step 1: Write the failing tests** (fake clock).
  - `test_full_bucket_allows_rpm_requests_then_blocks`: rpm=3 → three times 0.0, then about 20.0.
  - `test_refills_at_rpm_per_minute`: rpm=60, empty it, advance 1.0 s → one more allowed.
  - `test_never_exceeds_capacity`: rpm=2, idle for 10 minutes → only 2 in a burst.
  - `test_unlimited_key_never_limited`.
  - API: `test_rate_limited_is_429_with_retry_after`. rpm=1: the second request gets 429 `rate_limited`, `Retry-After` ≥ 1, and no upstream call.
- [ ] **Step 2: Run** `uv run pytest tests/test_budgets.py tests/api/test_limits_route.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement `RateLimiter` and the route check. Put a `TODO(human)` body in `TokenBucket.take`, request the contribution, and wait.**
- [ ] **Step 4: Run** the same command after the user's implementation, plus `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: per-key token-bucket rate limits"`

### Task 4: Budgets: reserve, settle, monthly reset

**Files:**
- Modify: `src/tierwise/budgets.py` (budget part), `src/tierwise/api/routes.py`, `src/tierwise/api/app.py` (load this month's spend at startup), `src/tierwise/api/errors.py` (`BudgetExceeded`: 429 `budget_exceeded`)
- Test: `tests/test_budgets.py`, `tests/api/test_limits_route.py`

**Interfaces:**
- Consumes: `RequestLog.submit_debit` and `load_month_spend`, plus `selector.estimated_cost`.
- Produces:
  - `month_of(ts: float) -> str` (UTC `"YYYY-MM"`)
  - `@dataclass class Reservation`, with fields `key_id: str`, `month: str` and `amount: float`.
  - `class Budgets`:
    - `__init__(self, log: RequestLog, clock: Callable[[], float] = time.time)`
    - `load(self, month: str, spend: dict[str, float]) -> None`
    - `spent(self, key_id: str, month: str | None = None) -> float`
    - `reserve(self, key: KeyRow, estimate: float) -> Reservation | None`: None for unlimited keys. Raises `BudgetExceeded` with a message showing spent, reserved and budget to the cent.
    - `settle(self, reservation: Reservation | None, key: KeyRow | None, actual: float | None) -> None`: releases the reservation. If the key exists and `actual` is not None, it adds the amount to the in-memory spend and calls `log.submit_debit(key.id, month_of(now), actual)`. It also debits keys without a budget, so their spend shows up.
  - Route: after `_choose`, `estimate = _estimate(plan, result.request)`, then `reservation = budgets.reserve(...)`. Every exit settles exactly once:
    - Non-streaming success: the actual cost.
    - Error: `None`.
    - Stream: in `stream_done`, the actual cost from `usage` when the status is 200; for a 499 or 502 without usage, `None`.
    - Cache hit: settles 0.0 with no reservation (the cache is checked before reserving, §4).
- [ ] **Step 1: Write the failing tests.**
  - Unit (a fake log recording `submit_debit` calls):
    - `test_reserve_within_budget_then_settle_debits_actual`.
    - `test_reserve_over_budget_raises`.
    - `test_reservations_count_against_budget`: budget 1.00 and estimates of 0.40 → the 3rd reserve raises before any settles.
    - `test_settle_none_releases_without_debit`.
    - `test_month_rollover_starts_fresh`: the clock moves from 2026-11-30T23:59Z to 2026-12-01T00:01Z, and spent for the new month is 0.
    - `test_unlimited_key_is_debited_but_never_blocked`.
  - API:
    - `test_over_budget_is_429_before_upstream`: budget 0.0 → 429 `budget_exceeded`, no upstream call.
    - `test_success_debits_actual_cost`: after the request, `budgets.spent(key) == row.cost_usd` and the `usage_monthly` row matches after `stop()`.
    - `test_failed_request_is_not_debited`.
    - `test_abandoned_stream_releases_reservation`: drive `sse_response` cancellation as in `test_streaming_route` → reserved back to 0, no debit.
    - `test_concurrent_requests_cannot_overshoot`: budget = 3 × estimate, 20 concurrent requests through `asyncio.gather` → at most 3 succeed and the rest get 429.
- [ ] **Step 2: Run** `uv run pytest tests/test_budgets.py tests/api/test_limits_route.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: monthly budgets with reserve-then-settle debits"`

### Task 5: `tierwise keys` CLI

**Files:**
- Modify: `src/tierwise/__main__.py`
  - `main(argv: list[str] | None = None) -> int`
  - No subcommand (or `serve`) runs the server as today.
  - `keys create --name N [--budget USD] [--rpm N]` prints the key once, plus the id and prefix.
  - `keys list` prints a table: id, name, prefix, budget, spent this month, rpm, status. **No secrets.**
  - `keys revoke ID` exits 1 for an unknown or already revoked key.
- Modify: `pyproject.toml`. Add `[project.scripts] tierwise = "tierwise.__main__:main"`.
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `generate_key`, `hash_key`, `RequestLog` key methods and `load_month_spend`.
- [ ] **Step 1: Write the failing tests** (`capsys`, `TIERWISE_DB_URL` pointed at `tmp_path`).
  - `test_create_prints_key_once_and_stores_only_hash`: the output contains a `tw-…` key, and the database row holds `hash_key(key)` but not the key.
  - `test_list_shows_no_secrets`: the printed text contains no key and no hash.
  - `test_revoke_then_revoke_again_exits_1`.
  - `test_list_shows_month_spend`: after a `submit_debit` of 1.25 and `stop()` → "1.25" appears.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: tierwise keys create|list|revoke"`

### Task 6: Concurrency load test and docs

**Files:**
- Create: `tests/api/test_load.py`
- Modify: `README.md` (a "Keys, budgets and rate limits" section), `.env.example`

**Interfaces:**
- Consumes: everything above.
- [ ] **Step 1: Write the test.** `test_200_concurrent_requests` (spec §7) uses a real SQLite file under `tmp_path`, three keys with ample budgets, and `asyncio.gather` over 200 `/v1/chat/completions` calls against fake providers. It then asserts:
  - All 200 get status 200.
  - After `stop()`: 200 rows in `requests`, `dropped == 0`, and no `database is locked` text in the captured logs (`caplog`).
  - The sum of `usage_monthly.spent_usd` equals the sum of `requests.cost_usd` to 1e-9, so no debit was lost or doubled.
  - The p50 of `latency_ms` is below 30.
  - The "pool reuse" part of the spec's check needs real sockets; it moves to M7b's compose smoke run (a ruling).
- [ ] **Step 2: Run** `uv run pytest tests/api/test_load.py -q`. Expected: PASS on the first run, since the behaviour exists. Record that in the ledger. A failure is a real bug: fix it with its own failing test.
- [ ] **Step 3: Write the docs:**
  - Creating keys.
  - Pointing the OpenAI SDK at the gateway with `api_key="tw-…"`.
  - Budgets: monthly UTC, reserve then settle.
  - The 401 and 429 responses and `Retry-After`.
  - The 30 s refresh delay for new or revoked keys.
  - `TIERWISE_REQUIRE_KEY`.
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check . && uv run mypy src`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "test: 200-request concurrency check; docs: keys, budgets, rate limits"`
