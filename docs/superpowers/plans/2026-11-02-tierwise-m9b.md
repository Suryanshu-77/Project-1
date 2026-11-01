# tierwise M9b Implementation Plan: Retrain, Gate, Promote

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python -m training.retrain` trains a challenger on the labeled shadow samples (plus the base data when it is present), gates it against the champion, and promotes it only when the gate says so. `scripts/flywheel_e2e.py` proves the whole loop in CI at $0.

**Architecture:**
- `training/train.py`'s fitting logic becomes a reusable `fit`.
- `evaluation/gate.py` scores two classifiers that share one set of embeddings, on the same rows.
- `training/retrain.py` orchestrates: load → embed → fit → gate → version → promote. The champion and the challenger share the encoder, so every prompt is embedded only once.

**Tech Stack:** scikit-learn and numpy, in a new dependency group `retrain = ["scikit-learn>=1.5"]` (no torch). The gateway's own `OnnxEncoder`, `Classifier` and tokenizer.

**Spec:** `docs/superpowers/specs/2026-11-01-tierwise-design.md` §5.2 (M9b, exit check)

## Global Constraints
- Everything in the M3–M9a plans' Global Constraints still holds.
- **Labels:** `needs_strong` from M9a. For shadow eval rows, `q_cheap = 0.0 if needs_strong else 1.0` and `q_strong = 1.0`. Costs are registry prices times `len(text) / 4` tokens, for the prompt and for each answer.
- **Critical set:** the 10 `label == "hard"` prompts in `tests/router/labeled_prompts.jsonl`. A critical misroute is `p < threshold` on one of them.
- **Versions:** `models/router/versions/<YYYYMMDDTHHMMSSZ>/` holds `classifier.npz`, `router.json` (with `"version"`) and `metrics.json` (with the gate report). Promotion copies those three files into the router dir. The replaced champion is first saved to `versions/<its version, or "initial">/` if it isn't already there.
- **`/healthz`** gains `"router_version"`. That is `router.json`'s `version`, `None` when it's absent, and `"rules-only"` when no learned router is loaded.
- **Retrain never touches the gateway at runtime.** A promotion takes effect on restart, and the CLI says so.

## Decisions this plan makes
- **The gate compares classifiers, not the full hybrid route:** the rules part is the same for both, so it cancels out. The existing CI critical gate still checks the full route. Cost if wrong: a promotion that the hybrid would have caught, which the existing CI gate still catches.
- **Small or one-class data:** `fit` falls back to `threshold = 0.5` and `val_auc = None` when the validation split has one class. `retrain` refuses (exit 2, "not enough labels") below 20 labeled rows, or with fewer than 2 rows of each class in train.
- **The decision is an outcome, not an error:** a rejection exits 0 and prints its reasons. Errors exit non-zero.

## Review Focus
1. **Shadow-only data (no `training/data`)**, the CI case: shadow rows supply train, val and test by prompt hash, with no prompt in two splits. Test in Task 2.
2. **Promote is atomic enough:** a crash between copying files must not leave `classifier.npz` and `router.json` from different versions. Write all three to a temp dir inside the router dir, then `os.replace` each file, `router.json` last, so a mismatch fails `Classifier.load`'s dim check rather than routing silently. Test in Task 3.
3. **The champion's `model.onnx` lives in the cache dir, not the router dir:** retrain must find the encoder the same way the gateway does. Test in Task 2.
4. **Re-running retrain with no new labels:** it produces a challenger equal to the champion. `should_promote` on a tie must be deterministic, and the result is documented. Test in Task 1.
5. **Critical prompts duplicated in the shadow data:** they are excluded from training so the gate stays honest. Test in Task 2.

---

### Task 1: `fit` and the gate (TODO(human): `should_promote`)

**Files:**
- Modify: `training/train.py` (pull `fit` out; `main` uses it), `pyproject.toml` (`retrain` group; `train` includes it)
- Create: `evaluation/gate.py`
- Test: `tests/evaluation/test_gate.py`, `tests/training/test_fit.py`

**Interfaces:**
- Produces, in `training/train.py`:
  - `@dataclass Fitted(model, kind: str, threshold: float, val_auc: dict[str, float] | None)`
  - `fit(x_train, y_train, x_val, y_val) -> Fitted`
- Produces, in `evaluation/gate.py`:
  - `@dataclass(frozen=True) GateResult(cost_saved_95: float | None, critical_misroutes: int, auc: float | None)`
  - `evaluate_classifier(p_rows: np.ndarray, rows: Sequence[EvalRow], p_critical: np.ndarray, threshold: float) -> GateResult`. It reuses `sweep`, `cost_saved_at` and `misroutes`.
  - `should_promote(champion: GateResult, challenger: GateResult) -> tuple[bool, str]`. This is the `TODO(human)`. Its contract: never promote when `challenger.critical_misroutes > 0`; never promote when the challenger's cost saved is worse than the champion's; return a human-readable reason either way.

- [ ] **Step 1: Write the failing tests:**
  - `test_fit_picks_better_of_logreg_and_mlp`, on separable synthetic data.
  - `test_fit_one_class_val_falls_back`
  - `test_evaluate_classifier_counts_critical_misroutes`
  - The `should_promote` contract tests: rejects on any critical misroute; rejects when cost saved is worse; promotes when strictly better with 0 misroutes; `None` cost saved counts as worst; the reason is non-empty. On a tie, assert only that two calls return the same result.
- [ ] **Step 2: Run** `uv run --group retrain pytest tests/evaluation/test_gate.py tests/training/test_fit.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** everything except `should_promote`, which gets a `TODO(human)` body raising `NotImplementedError`. The human writes it.
- [ ] **Step 4: Run** the full checks (as in M9a). Expected: PASS.
- [ ] **Step 5: Commit** `feat: reusable fit and champion/challenger gate`.

### Task 2: `training/retrain.py`

**Files:**
- Create: `training/retrain.py`
- Test: `tests/training/test_retrain.py`

**Interfaces:**
- Consumes: `fit`, `evaluate_classifier`, `should_promote` (Task 1); `RequestLog.labeled_samples()` (M9a); `split_of` (`training/prepare_data.py`); `router_directory`, `CACHE_ROUTER_DIR` (M8).
- Produces:
  - `shadow_rows(samples: list[ShadowSample], registry: Registry) -> dict[Split, list[tuple[str, int, EvalRow]]]`
  - `async run(db_url: str, router_dir: Path, registry: Registry, promote: bool, now: Callable[[], float] = time.time) -> RetrainReport`
  - `@dataclass RetrainReport(decision: bool, reason: str, champion: GateResult, challenger: GateResult, version: str, promoted: bool, n_labels: int)`
  - CLI: `python -m training.retrain [--db-url URL] [--router-dir DIR] [--registry PATH] [--dry-run]`. It prints the report as JSON and exits 0; exit 2 means not enough labels.

- [ ] **Step 1: Write the failing tests.** They use a tiny fake encoder (monkeypatch the embed function to a deterministic hash-to-vector map), so the tests need no ONNX:
  - `test_shadow_rows_split_by_prompt_hash`
  - `test_critical_prompts_excluded_from_training`
  - `test_refuses_below_20_labels` (exit 2)
  - `test_dry_run_writes_version_but_not_router_dir`
  - `test_promote_replaces_three_files_and_keeps_old_champion`
  - `test_reject_leaves_router_dir_byte_identical`
  - `test_encoder_found_in_cache_dir` (skipped without a real `model.onnx`)
  - `test_healthz_reports_router_version`, in `tests/api/`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** The base data, when `training/data/{train,val,test}.jsonl` exist, is merged split by split. Eval rows come from the base test split via `load_gpt4_test`, plus the shadow test split.
- [ ] **Step 4: Run** the full checks. Expected: PASS.
- [ ] **Step 5: Commit** `feat: retrain a challenger from shadow labels; gate and versioned promotion`.

### Task 3: End-to-end exit check, CI, docs

**Files:**
- Create: `scripts/flywheel_e2e.py`
- Modify: `.github/workflows/ci.yml` (job `flywheel-e2e`), `README.md`, `CHANGELOG.md` (`0.9.0`), `pyproject.toml` (version `0.9.0`), `docs/HANDOFF.md`

- [ ] **Step 1: `scripts/flywheel_e2e.py`.** It runs entirely in-process, with no Docker:
  1. Start `scripts/fake_upstream.py`'s app with uvicorn on a free port, in a thread.
  2. Copy the repo `models/router` to a temp router dir.
  3. Build `create_app` with a SQLite temp db, `store_payloads=True`, `shadow_rate=1.0`, `require_api_key=True`, Ollama pointed at the fake, and `deploy/registry.compose.yaml`.
  4. Create a key with `allow_shadow`, and send 120 distinct prompts: 60 hard-looking ones from templates like "Design a {thing} architecture …" and 60 easy ones like "Translate '{word}' into Spanish".
  5. Run `label_pending`, then `retrain.run(promote=True)`.
  6. Assert three things: the report has a decision and a non-empty reason; `router_dir`'s `router.json` changed if and only if `promoted`; and `versions/` has the challenger.

  Print the report.
- [ ] **Step 2: Run it locally.** Expected: exit 0 and a printed decision. Run it twice: the second run, with no new labels, must also exit 0.
- [ ] **Step 3: CI job `flywheel-e2e`.** Steps: `uv sync --group retrain`, then `uv run tierwise fetch-model`, then `uv run python scripts/flywheel_e2e.py`.
- [ ] **Step 4: README "Data flywheel" section** (retrain, gate, versions, rollback by copying a `versions/` dir back), CHANGELOG `0.9.0`, and HANDOFF.
- [ ] **Step 5: Run** the full checks, then commit `test: flywheel end-to-end exit check (M9); docs`.
