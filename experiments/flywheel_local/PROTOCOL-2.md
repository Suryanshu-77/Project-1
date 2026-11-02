# Protocol 2: the same-pair test with the two-score judge

*Written and committed before any of its runs, like [PROTOCOL.md](PROTOCOL.md). Its decision
rules are fixed. A result that misses them is reported as a miss, not re-analysed until it
passes.*

## What changed since Protocol 1

Protocol 1's pilot (stage 1) came back NO-GO. Claude Haiku agreed with SPROUT's verdict (AUC
0.897), but it scored identical answers below 4 in 6 of 20 pairs. In 5 of those, SPROUT says the
gpt-4o answer itself was wrong. A judge that scores only the cheaper answer therefore calls
prompts that *both* models fail "needs strong".

tierwise 0.10 fixed that. The judge (`tierwise-judge-v2`) scores **both** answers, and
`flywheel.needs_strong(cheap, strong) = cheap < 4 <= strong`. That is SPROUT's definition G
(`cheap < 0.5 <= strong`) on the judge's 1–5 scale.

**Carried over unchanged from Protocol 1 and its Amendment 1:**
- **Data:** the label pool, which is the 2,000 shuffled SPROUT train prompts in `stage0.build_pool`.
- **The eval and its split**, including the enlarged confirm set. The confirm set has never been
  loaded by any script.
- **The metrics:** cost saved at 99%, ROC-AUC, the shuffled control and the 5-seed noise band.
- **Stage 0 and its decision (definition G, N = 2,000, weight 5).** Stage 0 used SPROUT's own
  labels, not a judge, so a new judge doesn't change it.

**Not reused:** Protocol 1's 440 pilot judgments. They came from the v1 rubric, so every pair is
judged again.

## Stage 1: does the two-score judge behave? (pilot, about ₹18)

Haiku 5.5 judges, under the tierwise 0.10 rubric and seeded A/B order:
- **The first 400 pool pairs:** gpt-4o-mini against gpt-4o, from SPROUT's stored answers.
- **20 identical pairs:** pool prompts 1–20, with gpt-4o's answer against itself.
- **20 order flips:** pool prompts 1–20 judged again with A and B swapped, so 440 calls in all.

**Go** to stage 2 only if all four hold:

1. **Identical pairs never ask for the strong model:** `needs_strong` is False in at least
   **19 of 20**. Their two scores are also equal in at least **18 of 20**.
2. **The order of A and B doesn't matter:** after mapping back to (cheap, strong), the label is
   the same in at least **17 of 20** flips. Each score moves by at most 1 in at least 17 of 20.
3. **It tracks SPROUT's verdict:** on the 400 pairs, the gap `strong − cheap` separates SPROUT's
   G labels with ROC-AUC **≥ 0.70**.
4. **Its positive rate is plausible:** the share of the 400 that `needs_strong` labels lies
   between **half and twice** G's share on the same 400.

The label rule is the product's `needs_strong`. No cutoff is tuned on the pilot: there is
nothing left to choose.

Failed calls are retried by the script and stay unscored if they still fail. If more than 5% of
the 440 calls are unscored, the pilot is a **NO-GO**, because the checks would be judged on an
incomplete sample.

## Stage 2: the test (about ₹64)

Haiku judges pool pairs 401–2,000 the same way, and the 400 pilot pairs count toward the 2,000.

**One primary test, on the confirm set:** the classifier refit with Haiku's `needs_strong`
labels, weighted ×5, against "without". It **succeeds** if both hold:
1. The 95% interval of the paired bootstrap difference in cost saved at 99% (1,000 resamples)
   lies above 0.
2. Its mean beats the same prompts with shuffled labels by more than the confirm set's 5-seed
   noise band.

Reported alongside, but not as tests:
- SPROUT's own G labels on the same 2,000 prompts (the ceiling)
- ROC-AUC on the confirm set
- the label rate
- agreement with G
- the cost

A pair that stays unscored is left out of training, and the count is reported. If more than 5% of
the 2,000 are unscored, the result is reported as **incomplete**, not as a pass or a fail.

## Budget

Protocol 1 measured $0.00036 per call at list price, about ₹0.040 at this account's real rate
(₹33.79 for $0.308).

| Stage | Calls | Expected | Hard cap in the script (list-price USD) |
|---|---|---|---|
| 1 | 440 | about ₹18 | $0.20 |
| 2 | 1,600 | about ₹64 | $0.65 |

The key's own limit is ₹100, with usage reset to ₹0 on 2026-11-03, and it stops everything before
the wallet runs out. If a cap is reached, the run stops and nothing is concluded. Already-scored
pairs are stored, so a re-run never pays twice.

## What this still can't show

- **"Better" means "agrees with SPROUT's LLM judge":** neither judge is ground truth.
- **One model pair (gpt-4o-mini vs gpt-4o), single-turn prompts.** A pass shows that the
  flywheel's labels can improve routing on that pair. It doesn't size the gain on other traffic.
- **The confirm set is used once.** After stage 2 it is no longer untouched.
