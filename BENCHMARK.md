# tierwise routing benchmark

**Headline:** on 3,943 held-out prompts, tierwise's deployed router keeps **95% of GPT-4's quality at 44% of its cost (56.0% saved)**. Random routing saves 36.6% at the same quality. The learned classifier on its own saves 60.0%; the deployed router gives up 4 points of that so that rules can still push hard prompts upward.

Every number here comes from precomputed answers, so reproducing it costs nothing in API calls.

## Setup

Each prompt has an answer from a **cheap** and a **strong** model, with a known quality and cost for each. Every router gives each prompt a score, and a prompt goes to the strong model when its score is at or above a threshold *t*. Sweeping *t* from "everything cheap" to "everything strong" traces the router's cost/quality curve.

| | Held-out (headline) | SPROUT (out-of-distribution) |
|---|---|---|
| **Data** | Test split of [`routellm/gpt4_dataset`](https://huggingface.co/datasets/routellm/gpt4_dataset) (Apache-2.0). Split by prompt hash, never trained on. | Test split of [`CARROT-LLM-Routing/SPROUT`](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT). **No license is stated**; it is downloaded at eval time, nothing from it is committed, and only aggregates are reported. |
| **Prompts** | 3,943 | 6,637 (0 dropped) |
| **Cheap / strong** | Mixtral-8x7B / GPT-4 (`gpt-4-1106-preview`) | gpt-4o-mini / gpt-4o (`2024-08-06`) |
| **Quality** | Cheap: GPT-4 judge's grade of Mixtral's answer, (score − 1) / 4. Strong: 1.0. **Assumption:** the judge graded only Mixtral, so GPT-4's own answer is the reference. | The dataset's per-model `score` in [0, 1], from an LLM judge, for both models. |
| **Cost** | Prompt and answer length ÷ 4 chars per token × list price | The dataset's token counts × list price |
| **Prices** (USD / 1M tokens, in / out) | GPT-4 10 / 30 (OpenAI); Mixtral 0.60 / 0.60 (Together AI) | gpt-4o 2.50 / 10; gpt-4o-mini 0.15 / 0.60 (OpenAI) |

**The routers:**
- **random:** a uniform random score, seed 0.
- **rules:** the rules scorer.
- **learned:** the learned classifier's raw P(strong model needed).
- **hybrid:** what the gateway actually runs: `route()` with rules first, the learned router when rules are unsure, and difficulty = max(rules, learned).

**Metrics:**
- **Cost saved at X%:** the share of the all-strong cost saved at the cheapest point that reaches X% of all-strong quality. It is interpolated between neighbouring thresholds, which in practice means randomly mixing two thresholds. *Single threshold* is the conservative version: one fixed threshold.
- **Misroutes:** prompts the cheap model failed (quality < 0.5) while the strong one didn't, that were still routed cheap at the single-threshold 95% point.
- **AUC:** the area under quality-fraction over cost-fraction.

## Results

### Held-out: Mixtral-8x7B vs GPT-4

![held-out Pareto curve](docs/benchmark/heldout_pareto.png)

| Router | Saved @ 95% | @ 98% | @ 99% | Single threshold @ 95% | Misroutes @ 95% | AUC |
|---|---|---|---|---|---|---|
| random | 36.6% | 14.9% | 7.0% | 36.6% | 110 | 0.905 |
| rules | 38.9% | 10.1% | 5.1% | 38.3% | 98 | 0.905 |
| learned | **60.0%** | **28.3%** | **18.0%** | 60.0% | 68 | 0.930 |
| **hybrid (deployed)** | **56.0%** | 20.2% | 9.4% | 55.9% | 73 | 0.920 |

All-Mixtral already reaches 85.6% of GPT-4's quality at 2.7% of its cost.

### SPROUT: gpt-4o-mini vs gpt-4o

![SPROUT Pareto curve](docs/benchmark/sprout_pareto.png)

| Router | Saved @ 95% | @ 98% | @ 99% | AUC |
|---|---|---|---|---|
| random | 93.2% | 35.2% | 21.0% | 0.909 |
| rules | 93.2% | **58.5%** | **53.4%** | 0.915 |
| learned | 93.2% | 51.7% | 40.9% | 0.914 |
| hybrid (deployed) | 93.2% | 52.0% | 48.8% | 0.915 |

**gpt-4o-mini alone reaches 95.2% of gpt-4o's quality** (mean judged score 0.807 vs 0.848) at 6.8% of its cost, so every router "saves" 93.2% at the 95% target just by sending everything cheap. The stricter targets are where the routers differ:
- **All routers beat random.**
- **Rules beat the learned classifier out of distribution.** The classifier learned which prompts *Mixtral* struggles with, in 2024-era data; gpt-4o-mini fails on different prompts.
- **The hybrid's recovery depends on the target.** Because rules can raise its difficulty, at 99% it recovers most of the gap between learned and rules (48.8% vs 40.9% and 53.4%). At 98% it recovers almost none of it (52.0% vs 51.7% and 58.5%).

## The deployed hybrid

- **Who decides:** the rules router decides alone only when it is confident (confidence ≥ 0.8). On every other prompt the learned classifier decides, and rules can only raise its difficulty.
- **The max rule:** this was added after the critical-set gate caught the classifier sending 4 of 10 labeled hard prompts (a proof, a derivation, an architecture design and a legal review) to basic-tier models. The rule costs 4 points on the held-out headline (60.0% → 56.0%), and it brings the out-of-distribution result closer to the rules router.
- **Low confidence:** 42.0% of held-out prompts (61.6% on SPROUT) get a decision with confidence below 0.3, so the gateway raises their quality bar one step. That comes from `to_decision` measuring distance from a low threshold (0.15) in probability space. It is a deliberate upward bias, and the next thing to tune when real traffic exists.

## The flywheel: do judge labels improve the router?

The first run was fully local, at $0, on an RTX 4060 laptop GPU with Ollama. The follow-ups used a paid judge. Scripts and steps: [`experiments/flywheel_local`](experiments/flywheel_local/README.md).

**Setup:**
- **Traffic:** 744 SPROUT prompts (none in the base data) went through the gateway: qwen2.5:1.5b as basic, qwen2.5:7b as flagship.
- **Labeling:** `tierwise flywheel label` got the contrast answers. **mistral:7b**, which wrote neither answer, judged them, and `needs_strong = score < 4`.
- **Retraining:** the classifier was refit on the base data plus the labels from train-hash prompts. It was then scored on **3,156 other SPROUT prompts against SPROUT's own gpt-4o-mini / gpt-4o quality scores**, so the scoring doesn't depend on our judge.

**What went wrong first, and the fixes in 0.9.2:**
- **A saturated test:** the first evaluation used 156 held-out prompts at the 95% target. There, all-cheap already reached 97.6% of all-strong quality, so perfect, random and constant routing all scored exactly 92.4%. The test couldn't have shown any effect. The gate now measures at a target set from its eval rows (`quality_target`: half the gap between all-cheap and all-strong, at least 95%).
- **Dropped labels:** mistral:7b quoted 216 of its scores (`{"score": "5"}`), and `parse_score` rejected them. 171 of the 216 were 5s, so the labels that survived were skewed toward "needs strong". Quoted numbers are now accepted.

**Results at a fixed 99% quality target** (the experiment's, set before the gate's `quality_target`, which would pick about 97.6% here). Δ is the mean of 300 paired bootstraps against the refit without new labels, so it differs slightly from the difference of the point estimates:

| Labels added | Rate | Saved @ 99% | ROC-AUC vs SPROUT verdict | Δ vs none [95% interval] | Wins |
|---|---|---|---|---|---|
| none | n/a | 42.6% | 0.670 | n/a | n/a |
| **judge `score < 4`, once (0.9's rule)** | 31% | **53.4%** | **0.693** | **+10.4 [+0.3, +21.2]** | **294/300** |
| judge `score < 4`, 10× weight | 31% | 59.7% | 0.698 | +18.5 [+5.2, +33.8] | 297/300 |
| judge `score < 3`, once | 8% | 40.8% | 0.659 | +0.0 [−9.1, +10.4] | 147/300 |
| judge `score < 3`, 10× weight | 8% | 50.6% | 0.658 | +7.6 [−5.0, +20.3] | 269/300 |
| SPROUT's own verdict, once | 7% | 39.0% | 0.654 | −3.0 [−12.4, +9.5] | 67/300 |
| SPROUT's own verdict, 10× weight | 7% | 48.0% | 0.687 | +6.9 [−4.3, +21.1] | 256/300 |
| perfect routing (ceiling) | n/a | 90.6% | 1.000 | +47.3 | 300/300 |

ROC-AUC here is a different measure from the AUC column above. It is the chance that a prompt SPROUT says needs the strong model (cheap < 0.5 ≤ strong) scores above one that doesn't.

**What later checks showed: this run can't tell whether the flywheel helps.**
- **The +10.4 isn't robust.** On the 401 prompts whose answer pairs were stored, mistral's labels give **+4.3 [−4.5, +12.8]**, which isn't significant. The +10.4 needed 175 recovered labels, mostly "the cheap answer was fine".
- **A stronger judge made the router worse** (a follow-up check, not pre-registered). Claude Haiku 5.5 re-judged the stored pairs for $0.15. It called 83% of them "needs strong", and its labels cut cost saved by **18.7 points [−30.1, −9.3]**. That doesn't mean it judged badly. A label belongs to one model pair: these compare qwen2.5:1.5b with qwen2.5:7b, a gap Haiku saw clearly, while the eval asks whether *gpt-4o-mini* falls short of *gpt-4o*.
- **So the labels and the eval describe different model pairs.** Mistral's lenient labels happened to suit the gpt-4o pair better.

### The same-pair test (pre-registered)

[`PROTOCOL.md`](experiments/flywheel_local/PROTOCOL.md) fixed every rule before its runs.
- **Labels:** Claude Haiku judges SPROUT's own gpt-4o-mini vs gpt-4o answer texts, the pair the eval scores.
- **Prompts:** labels come from SPROUT's train split. The eval uses the test split, halved: one half to choose a setup, the other to test it once. Amendment 1, made before any paid call, adds the train prompts never labeled to the test half.
- **Controls:** every comparison is checked against the same prompts with shuffled labels, and against a 5-seed retraining noise band.

| Stage | Cost | Result |
|---|---|---|
| **0: can perfect labels help?** SPROUT's own verdicts, 500 / 1,000 / 2,000 prompts, added once or ×5 | $0 | Only at **2,000 ×5**: +12.9 [+1.5, +27.7] (G) and +17.5 [+3.5, +33.3] (S). Added once, even 2,000 perfect labels beat the noise band, but their intervals include 0. |
| **1: does Haiku judge like SPROUT?** 400 pairs, plus two 20-pair checks | $0.16 | AUC **0.897** against SPROUT's verdict, and order flips agree (18/20). But identical answers scored 4 or more only **14/20** (18 needed), so **NO-GO**: stage 2 wasn't run. |

**Why the identical-answer check failed:**
- **What happened:** in 5 of the 6 misses, SPROUT says gpt-4o's answer was itself wrong, and Haiku scored the "cheaper" copy 1.
- **The flaw it exposed:** labels from a judge that scores only the cheaper answer call prompts that *both* models fail "needs strong", so the router learns to pay for the flagship where it buys nothing.
- **The fix, in 0.10:** the judge scores both answers, and a prompt needs the strong model only when the cheaper answer falls short and the stronger one doesn't.

### The two-score judge: Protocols 2 and 3

**[Protocol 2](experiments/flywheel_local/PROTOCOL-2.md)** re-ran the pilot with tierwise 0.10's
judge, which scores both answers and labels `cheap < 4 <= strong`. It cost $0.17.
- **The judge behaved:** identical answers never asked for the strong model (20/20), swapping A
  and B kept the label in 19 of 20 pairs, and the score gap tracked SPROUT at AUC **0.914**.
- **Its label rate failed:** 21.5% against G's 6.5%, so the pilot was **NO-GO**. Haiku grades
  harder than SPROUT's judge: 42 of its 86 positives were a 3 against a 4, where SPROUT scored
  the cheap answer fine.

**[Protocol 3](experiments/flywheel_local/PROTOCOL-3.md)** changed only the cut point, to
`cheap < 2` and `strong >= 4`.
- **How the cut was chosen:** from the stored pilot. It agrees best with G (F1 0.77) among the
  cuts that pass the rate check.
- **Stage 1:** the stored judgments re-checked under the new cut passed every check, at no cost.
- **Stage 2:** Haiku judged the remaining 1,600 pairs ($0.64). Then came **the one pre-registered
  test, on 29,055 SPROUT prompts that no step had touched**:

| Labels on the same 2,000 prompts (weight 5) | Label rate | Saved @ 99% | Δ vs none [95% interval] | Shuffled labels | Verdict |
|---|---|---|---|---|---|
| none | n/a | 39.3% | n/a | n/a | n/a |
| **Haiku, `cheap < 2 <= 4 <= strong` (the test)** | 4.9% | **44.7%** | **+5.8 [+1.1, +10.8]** | −6.9 | **SUCCESS** |
| SPROUT's own G (the ceiling) | 7.1% | 55.1% | +15.7 [+10.7, +21.0] | −14.0 | passes |
| Haiku, `cheap < 4 <= strong` (0.10's shipped cut, comparison only) | 21.4% | 28.5% | **−10.9 [−15.6, −6.2]** | −14.2 | n/a |

The test row clears both pre-registered rules. Its interval is above 0, and it beats the shuffled
control by 12.7 points against a 5.3-point retraining noise band.

**What it shows:**
- **The flywheel can improve the router.** About 2,000 labels from a paid judge raised cost saved
  at 99% quality by 5.8 points, a little over a third of what perfect labels gave.
- **The cut point decides the sign.** The same judgments with 0.10's shipped cut made the router
  significantly *worse*. A stricter judge needs a stricter "falls short", so the cut should be
  calibrated per judge rather than fixed at 4.
- **AUC went down, not up** (0.649 → 0.631). The labels improved the decision that matters at a 99%
  target, not the overall ranking.

**Caveats:**
- **c = 2 was chosen from 400 pilot pairs.** The untouched confirm set guards the test, but this
  shows c = 2 works for Claude Haiku on gpt-4o-mini vs gpt-4o, not that it transfers.
- **"Quality" means SPROUT's LLM judge.** It is single-turn, two-model traffic.
- **Weight 5 and 2,000 labels were fixed by stage 0.** Fewer labels, or weight 1, didn't help even
  with perfect labels.

### The product's calibration, run for real (Protocol 4)

tierwise 0.11 turned the finding above into a command: `tierwise flywheel calibrate` chooses each
judge's cut against 400 packaged SPROUT reference pairs (rate within ½× to 2× of SPROUT's, then
best F1). **[Protocol 4](experiments/flywheel_local/PROTOCOL-4.md)** ran it once, end to end,
with Claude Haiku, and pre-registered the prediction **c = 2**. Result: **SUCCESS.**

| | Cut | Label rate (G 6.5%) | Precision | Recall | Unscored | Spent |
|---|---|---|---|---|---|---|
| `calibrate`, live (Protocol 4) | **2** | 4.25% | 94% | 61.5% | 0 / 400 | $0.159 |
| Stored pilot judgments (Protocol 3) | 2 | 4.5% | 94% | 65% | n/a | n/a |

The hosted judge's scores drifted a little between runs, as expected, but not enough to change the
choice. These are the pairs Protocol 3 chose from, so this shows the product reproduces it, not
that c = 2 transfers.

**Spent in all:** $1.28 at list price, about ₹141 through AICredits, for every paid run above: ₹33.79 before Protocol 2 (from the dashboard), about ₹90 for Protocols 2 and 3, and about ₹17.5 for Protocol 4 (both estimated from the measured rate).

## Limitations

- **Older model pairs.** Neither dataset contains tierwise's registry models (Claude 5.5, GPT-5.x, Gemini 3). The benchmark measures how well the router *ranks* prompts by how much they need a strong model, and that ranking is what carries over. The registry's `quality` values stay uncalibrated placeholders until flywheel labels exist for current models.
- **Two models, single-turn.** The gateway chooses among many models and caches multi-turn prefixes, and neither shows up here. Cache-aware selection is covered by the selector tests instead.
- **Judge-based quality.** Both datasets grade answers with an LLM judge, and the held-out set assumes GPT-4's answer is always right.
- **SPROUT's license is unstated.** It is used for aggregate numbers only.

## Reproduce

```bash
uv sync --group train
uv run python -m training.export_onnx
uv run python -m training.prepare_data --limit 40000
uv run python -m evaluation.run_eval --dataset heldout   # docs/benchmark/heldout.{json,png}
uv run python -m evaluation.run_eval --dataset sprout    # downloads ~580 MB into the HF cache
```

The critical-set gate (`tests/evaluation/test_critical_gate.py`) checks that the 10 labeled hard prompts never route to a basic-tier model. Coverage differs by environment:
- **CI** runs the rules path only, because the 90 MB ONNX encoder isn't there. The max(rules, learned) logic is covered separately by unit tests with fake routers.
- **The learned path**, with the trained classifier, runs wherever the model files exist. Run it after every retrain.
