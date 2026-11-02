# Protocol 3: the same-pair test with a judge-calibrated cut point

*Written and committed before any of its runs. It extends [PROTOCOL-2.md](PROTOCOL-2.md) and
changes one thing, the label rule. Its decision rules are fixed. A result that misses them is
reported as a miss.*

## Why

Protocol 2's pilot came back NO-GO on one check. The two-score judge behaved well:
- identical answers never asked for the strong model (20/20)
- swapping A and B kept the label in 19/20 pairs
- its score gap tracked SPROUT's verdict at AUC 0.914

But `cheap < 4 <= strong` labeled 21.5% of the 400 pairs, 3.3× SPROUT's rate for G (6.5%).
Claude Haiku grades harder than SPROUT's judge: 42 of its 86 positives were a 3 against a 4, and
SPROUT scored most of those cheap answers 1.0.

## The one change: where "falls short" starts

The label becomes **`cheap < c` and `strong >= 4`**, with **c = 2**: the cheaper answer is wrong or
misses important parts, and the stronger one is acceptable.

**How c was chosen,** from the 400 stored pilot judgments only (the confirm set has never been
loaded):
- The candidates were c = 2, 3 and 4.
- Only c = 2 (4.5%) and c = 3 (8.8%) pass Protocol 2's rate check (½× to 2× of G's 6.5%).
- Of those two, c = 2 agrees with G better:

| c | Rate | Precision vs G | Recall vs G | F1 |
|---|---|---|---|---|
| 2 | 4.5% | 94% | 65% | 0.77 |
| 3 | 8.8% | 51% | 69% | 0.59 |

c = 2 is fixed from here on. It is chosen from about 26 true positives, so the protocol tests c = 2
itself; it doesn't claim c = 2 is the best cut for every judge.

**What this means for the product:** tierwise 0.10 ships `needs_strong = cheap < 4 <= strong`.
This protocol tests whether a judge-calibrated cut makes the flywheel's labels improve routing.
If it does, the cut should become something calibrated per judge rather than a constant. That's a
separate product change, not part of this run.

## Stage 1: the stored pilot, re-checked under c = 2 (free, no calls)

Protocol 2's checks apply unchanged to the 440 stored judgments, with the new label:
1. Identical pairs: the label fires in at most 1 of 20, and the two scores are equal in at least 18.
2. Order flips: the same label in at least 17 of 20, and each score within 1 in at least 17.
3. The gap `strong − cheap` gives AUC ≥ 0.70 against G.
4. The label rate lies between half and twice G's rate.
5. At most 5% of calls are unscored.

**Go** to stage 2 only if all of them pass. The first two scoring checks and the AUC don't depend
on c; they passed in Protocol 2 and are recomputed anyway.

## Stage 2: the test (about ₹64)

Stage 2 is Protocol 2's stage 2, with the c = 2 label:
- **Judging:** Claude Haiku judges pool pairs 401–2,000. The 400 pilot pairs count toward the 2,000.
- **Training:** the label is used with weight 5.
- **The test:** one primary test on the 29,055-row confirm set (Amendment 1), against "without".
- **Success:** the 95% interval of the paired bootstrap difference in cost saved at 99% (1,000
  resamples) lies above 0, **and** its mean beats the shuffled control by more than the confirm
  set's 5-seed noise band.
- **Reported alongside, not as tests:** SPROUT's G labels on the same prompts (the ceiling), and
  for comparison the `cheap < 4` labels from the same judgments. The table shows that second
  row, but it is not a test.
- **Incomplete:** more than 5% of the 2,000 pairs unscored means the result is reported as
  incomplete.

**Budget:** 1,600 calls at Protocol 1's measured ₹0.040 per call, about ₹64. The script's hard cap
is $0.65 at list price. The key has about ₹81 left of its ₹100.

## What this still can't show

Everything listed in Protocol 2 still applies, and:
- **c = 2 was chosen after seeing the pilot.** The confirm set guards the test against that, but a
  pass shows c = 2 works for this judge on this model pair, not that it transfers.
