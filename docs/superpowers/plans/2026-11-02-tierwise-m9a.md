# tierwise M9a Implementation Plan: Shadow Sampling and Offline Labeling

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** opted-in traffic leaves pending `shadow_samples` rows, and `tierwise flywheel label` turns them into `needs_strong` labels with a contrast model and a judge, within a monthly budget, at $0 against the fake upstream.

**Architecture:**
- **Gateway.** One hook, in `keep()`, decides whether to sample (three opt-ins, then a confidence-dependent rate) and submits a row through `RequestLog`'s queued writer.
- **Offline.** A new module, `src/tierwise/flywheel.py`, does the contrast call and the judge call through the existing provider adapters, which come from a shared `build_providers()`.
- **Fake judge.** The fake upstream answers the judge rubric deterministically.

**Tech Stack:** existing only: SQLAlchemy Core, httpx, the provider adapters, argparse.

**Spec:** `docs/superpowers/specs/2026-11-01-tierwise-design.md` §5.2 (M9a), §3.9 (metrics, payload opt-in)

## Global Constraints
- Everything in the M3–M8 plans' Global Constraints still holds.
- **Settings (exact names and defaults):**
  - `store_payloads: bool = False` (env `TIERWISE_STORE_PAYLOADS`)
  - `shadow_rate: float = 0.02` (env `TIERWISE_SHADOW_RATE`)
  - `shadow_budget_usd: float = 5.0` (env `TIERWISE_SHADOW_BUDGET_USD`)
  - `judge_model: str | None = None` (env `TIERWISE_JUDGE_MODEL`; `None` means the registry's `baseline_model`)
- **Sampling needs all of:**
  - `store_payloads`
  - a key with `allow_shadow`
  - `result.decision is not None`
  - the hook is only reached for complete text answers (`keep()`'s existing condition)
- **The request path never awaits the sampler.** A full queue drops the sample and counts it in `tierwise_telemetry_dropped_total`.
- **Label:** `needs_strong = judge_score < 4`. The judge rubric carries the marker `tierwise-judge-v1`, and the judge's reply is JSON `{"score": <int 1-5>}`.
- **Metric:** `tierwise_shadow_samples_total`, a counter with no labels.
- **Spans and logs never carry prompt or answer text**, and that includes the label CLI's output.

## Decisions this plan makes
- **`shadow_samples` is a new table, not columns on `requests`:** samples are rare, hold large text, and have their own lifecycle (pending → labeled / failed). Cost if wrong: one join.
- **Samples get their own bounded queue in `RequestLog`,** drained in the same writer loop. Mixing them into the `RequestRecord` queue would force a union type through every batch. Cost if wrong: none.
- **The month for the budget is UTC `labeled_ts`,** using the existing `month_of`.
- **Judge answer order is randomized with a seeded `random.Random(sample.id)`,** so a re-run is reproducible.

## Review Focus
1. **Stream with a sampled key:** the streamed text is what gets stored, not an empty string. Test in Task 2.
2. **Judge returns non-JSON, or a score outside 1–5:** the sample becomes `status="failed"` with `error`, its cost is still counted, and the run continues. Test in Task 3.
3. **Budget hit mid-run:** stop before the next sample, leave the rest `pending`, exit code 0, and print how many are left. Test in Task 3.
4. **Routed model is the flagship:** the contrast is the cheapest basic model, and the judge scores *that* answer. Test in Task 3.
5. **A key without `allow_shadow`, or `store_payloads` off:** never sampled, even at `shadow_rate=1.0`. Test in Task 2.

---

### Task 1: Storage, keys flag, settings

**Files:**
- Modify: `src/tierwise/store.py` (the `shadow_samples` table, `KeyRow.allow_shadow`, `api_keys.allow_shadow` column, sample queue and APIs), `src/tierwise/config.py`, `src/tierwise/__main__.py` (`keys create --allow-shadow`), `src/tierwise/metrics.py`
- Test: `tests/test_store.py`, `tests/test_cli.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces, in `store.py`:
  - `@dataclass(frozen=True) class ShadowSample` with the fields `id: str, ts: float, key_id: str, request_json: str, routed_model: str, routed_answer: str, difficulty: float | None, confidence: float | None, source: str, status: str = "pending", contrast_model: str | None = None, contrast_answer: str | None = None, judge_score: int | None = None, needs_strong: bool | None = None, label_cost_usd: float = 0.0, labeled_ts: float | None = None, error: str | None = None`.
  - `RequestLog.submit_sample(sample: ShadowSample) -> bool`: queued; `False` means it was dropped.
  - `async RequestLog.pending_samples(limit: int) -> list[ShadowSample]`, oldest first.
  - `async RequestLog.save_label(sample: ShadowSample) -> None`: updates the row by `id`.
  - `async RequestLog.labeled_samples() -> list[ShadowSample]`, where `needs_strong` is not null.
  - `async RequestLog.label_spend(month: str) -> float`
  - `RequestLog.create_key(..., allow_shadow: bool = False)`, and `KeyRow.allow_shadow: bool`.
- Produces: `Metrics.shadow_sample() -> None`.

- [ ] **Step 1: Write the failing tests:**
  - `test_sample_roundtrip`: submit, flush with `stop()`, then `pending_samples(10)` returns it.
  - `test_save_label_moves_out_of_pending`
  - `test_label_spend_sums_month`: two months, only one is counted.
  - `test_sample_queue_full_drops_and_counts`
  - `test_existing_db_gains_allow_shadow_column`: through the existing `_add_missing_columns` path.
  - `test_keys_create_allow_shadow` (CLI)
  - `test_shadow_metric_exported`
  - The store tests run on SQLite and on Postgres, like the existing parametrized ones.
- [ ] **Step 2: Run** `uv run pytest tests/test_store.py tests/test_cli.py tests/test_metrics.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** The writer loop drains up to 200 samples per batch, after the request records.
- [ ] **Step 4: Run** `uv run ruff check -q . && uv run mypy src && uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: shadow_samples table, allow_shadow keys, payload settings`.

### Task 2: Sampling hook (TODO(human): `sample_rate`)

**Files:**
- Create: `src/tierwise/shadow.py`
- Modify: `src/tierwise/api/routes.py` (`keep()`), `src/tierwise/api/app.py` (store the settings on `app.state`)
- Test: `tests/api/test_shadow_route.py`, `tests/test_shadow.py`

**Interfaces:**
- Consumes: `ShadowSample`, `submit_sample`, `Metrics.shadow_sample`, `KeyRow.allow_shadow` (Task 1).
- Produces, in `shadow.py`:
  - `sample_rate(base: float, confidence: float | None) -> float`. This is the `TODO(human)`. Its contract: the result is in `[0, 1]`; `base` when confidence is high or `None`; higher than `base` when confidence is low.
  - `should_sample(key: KeyRow | None, decision: RoutingDecision | None, store_payloads: bool, base: float, rng: Callable[[], float] = random.random) -> bool`

- [ ] **Step 1: Write the failing tests:**
  - `test_sample_rate_contract` checks the contract above, at confidence `None`, 0.1, 0.5 and 0.95, and that the result is always `<= 1`.
  - `test_should_sample_requires_all_opt_ins`: a table of 4 cases, with `rng=lambda: 0.0`.
  - `test_sampled_non_stream_request_stores_row` and `test_sampled_stream_stores_full_text`: these use `gateway(...)` with `store_payloads=True, shadow_rate=1.0` and a key from `make_key(..., allow_shadow=True)`. The stored row has `routed_answer` equal to the reply, and `request_json` round-trips to a `ChatRequest`.
  - `test_no_allow_shadow_never_samples`, at rate 1.0.
  - `test_concrete_model_request_never_samples` (it has no decision).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** `should_sample` and the `keep()` hook. `sample_rate` has a `TODO(human)` body that raises `NotImplementedError`. The human writes it, then runs the tests.
- [ ] **Step 4: Run** the full checks. Expected: PASS.
- [ ] **Step 5: Commit** `feat: sample opted-in routed answers into shadow_samples`.

### Task 3: `tierwise flywheel label`, plus the fake judge

**Files:**
- Create: `src/tierwise/flywheel.py`
- Modify: `src/tierwise/api/app.py` (pull provider construction out into `build_providers`), `src/tierwise/__main__.py` (subcommand `flywheel label [--limit N]`), `scripts/fake_upstream.py` (fake judge)
- Test: `tests/test_flywheel.py`, `tests/test_deploy_files.py` (fake judge)

**Interfaces:**
- Produces:
  - `build_providers(config: Settings, client_for: Callable[[str], httpx.AsyncClient]) -> dict[str, Provider]`, in `api/app.py`.
  - `contrast_for(routed: ModelSpec, registry: Registry, providers: set[str]) -> ModelSpec | None`
  - `JUDGE_MARKER = "tierwise-judge-v1"`
  - `judge_messages(prompt: str, first: str, second: str) -> list[Message]`
  - `parse_score(text: str) -> int`, which raises `ValueError`.
  - `async label_pending(log: RequestLog, registry: Registry, providers: dict[str, Provider], judge: ModelSpec, budget_usd: float, limit: int, clock: Callable[[], float] = time.time) -> LabelRun`
  - `@dataclass LabelRun(labeled: int, failed: int, left: int, spent_usd: float, stopped_by_budget: bool)`

- [ ] **Step 1: Write the failing tests:**
  - These use a `RequestLog` on SQLite and providers backed by `httpx.MockTransport`, with a judge handler that returns `{"score": 2}`.
  - `test_cheap_routed_gets_flagship_contrast_and_label`: `needs_strong is True`, and `label_cost_usd` is greater than 0.
  - `test_flagship_routed_contrasts_cheapest_basic`
  - `test_bad_judge_json_marks_failed_and_continues`
  - `test_budget_stops_before_next_sample`: the budget is smaller than one label's cost plus the spend so far.
  - `test_judge_order_is_seeded`: run twice, and get the same order for the same id.
  - `test_cli_flywheel_label_prints_summary_only`: the output has counts and dollars, and no prompt text.
  - `test_fake_upstream_judges_hard_prompts_low`: the fake upstream returns `{"score": 2}` for a rubric whose prompt contains "design"/"prove"/"debug"/"production"/"architecture", and `{"score": 5}` otherwise.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Costs come from `cost_usd(usage, spec)`. Provider errors on either call mark the sample `failed`. Run it under `asyncio.run` with one `httpx.AsyncClient` per provider, closed at the end.
- [ ] **Step 4: Run** the full checks. Expected: PASS.
- [ ] **Step 5: Commit** `feat: tierwise flywheel label (contrast + judge, monthly budget)`.

### Task 4: Compose wiring and docs

**Files:**
- Modify: `deploy/docker-compose.yml` (gateway env `TIERWISE_STORE_PAYLOADS: "true"`), `README.md` ("Data flywheel" section), `docs/HANDOFF.md`
- Test: `tests/test_deploy_files.py`

- [ ] **Step 1: Test** `test_compose_enables_payloads`. Expected: FAIL, then add the env setting. Expected: PASS.
- [ ] **Step 2: Manual check.**
  1. Bring the compose stack up.
  2. `tierwise keys create --name fw --allow-shadow` inside the gateway container.
  3. Send 20 requests with `TIERWISE_SHADOW_RATE=1.0`.
  4. `docker compose exec gateway tierwise flywheel label`.

  Expected: 20 labeled, with $ spent printed.
- [ ] **Step 3: README.** Cover the three opt-ins, the privacy note (samples store prompts; there's no purge command yet), the cost of real labels (two calls per sample), and the commands.
- [ ] **Step 4: Run** the full checks, then commit `docs: data flywheel collection`.
