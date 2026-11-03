# Protocol 4: the product's calibration, run for real

*Written and committed before the run. Its decision rule is fixed. A result that misses it is
reported as a miss.*

## Why

tierwise 0.11 turned Protocol 3's finding into a product step: `tierwise flywheel calibrate`
chooses the needs-strong cut for a judge, using the user's `choose_cut` rule (label rate within
½× to 2× of SPROUT's, best F1, ties to the smaller cut).

On the 400 pilot judgments stored by Protocol 2, that rule picks c = 2, and a test checks it.
That test runs the rule on judgments the experiment scripts collected. It has never run the
shipped path end to end:
- the hash-only reference
- the pinned parquet download and its checks
- the product's judge call
- the cuts file

This protocol runs that path once with the real judge.

## The run

- **Judge:** Claude Haiku (`anthropic/claude-haiku-5.5`) through AICredits, the same judge as
  Protocols 2 and 3. It is registered as `claude-haiku-5-5`, the product registry's id, in
  `registry.judge.yaml`.
- **Reference:** the 400 packaged pairs: the Protocol 2 pilot pool at SPROUT commit `d54e3cd`.
- **Command:** `uv run --group train python experiments/flywheel_local/calibrate_haiku.py`. This
  maps `.env`'s judge settings onto tierwise's OpenAI-compatible provider, then runs
  `tierwise flywheel calibrate` unchanged.
- **Output:** the cut is saved to `experiments/flywheel_local/judge_cuts_haiku.json`, not to the
  user's own `~/.cache`.
- **Budget:** `--max-usd 0.20` at list price, which is about ₹22 real. The new key's limit
  (about ₹25) is a second backstop.
  - **Expected cost:** about $0.16 at list, measured on the same 400 calls in Protocol 2, which
    is about ₹17.5 real.
  - **The tool's own dry-run estimate** is $0.166.

## Decision rule

- **Prediction:** calibrate saves **c = 2** for `claude-haiku-5-5`.
- **SUCCESS:** the command exits 0 and the saved cut is 2.
- **MISS:** the command exits 0 with a different cut, or no cut qualifies (exit 1).
- **INCOMPLETE:** any of the following, which is reported as incomplete, not as a miss. A rerun
  needs a note added here first.
  - the spend cap is hit
  - more than 5% of the pairs are unscored
  - a download or check fails
- **Reported alongside, not as tests:**
  - the chosen cut's rate, precision, recall and F1
  - Protocol 3's stored-judgment figures for c = 2, for comparison: rate 4.5%, precision 94%,
    recall 65%
  - the amount spent

## Result (2026-11-03): SUCCESS

`calibrate_haiku.py` exited 0 and saved **cut 2** for `claude-haiku-5-5`
([judge_cuts_haiku.json](judge_cuts_haiku.json)):

| | Rate (G 6.5%) | Precision | Recall | F1 | Unscored | Spent (list) |
|---|---|---|---|---|---|---|
| This run | 4.25% | 94% | 61.5% | 0.74 | 0 / 400 | $0.1588 |
| Protocol 3, stored judgments | 4.5% | 94% | 65% | 0.77 | n/a | n/a |

The dry run estimated $0.17 (0.11.1's corrected estimator), and the real spend came within 5% of
it.

## What this can't show

- **These are the same pairs Protocol 3 chose c = 2 from.** A SUCCESS shows that the product
  path reproduces the experiment's choice. It doesn't show that c = 2 holds for other prompts
  or other model pairs.
- **The judge runs at temperature 0, but it is a hosted model.** Its scores can drift from
  Protocol 2's stored ones. Small differences in the reported figures are expected.
