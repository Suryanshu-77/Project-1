# Protocol: does a strong judge's flywheel make the router better?

*Written and committed before any of the runs it describes. Its decision rules are fixed;
results that miss them are reported as misses, not re-analysed until they pass.*

## Why a new experiment

The first local run labeled **qwen2.5:1.5b vs qwen2.5:7b** answers but scored the router on
SPROUT's **gpt-4o-mini vs gpt-4o** quality. A "needs the strong model" label belongs to one model
pair. A stronger judge (Claude Haiku 5.5) judged the qwen pair faithfully (83% "needs strong")
and its labels made the router worse on the gpt-4o eval. So the eval measured the wrong pair.

SPROUT stores both models' answer *texts*. A judge can label the **same pair the eval scores**,
with nothing regenerated.

## Question

Do labels from Claude Haiku 5.5, judging gpt-4o-mini against gpt-4o on SPROUT prompts the eval
never sees, make tierwise's classifier save more cost at 99% of gpt-4o's quality than training
without them? The gain has to be larger than what the same prompts with shuffled labels give,
and larger than retraining noise.

## Data

| Set | Source | Use |
|---|---|---|
| Base | routellm train/val (as shipped) | always in training |
| Label pool | SPROUT **train** split (30,968), minus the 49 prompts also in test, the 10 critical prompts, and prompts outside 20–1,500 chars | sampled with a fixed seed |
| Eval | SPROUT **test** split (6,637), same exclusions, split in two by prompt hash | **dev** half: stage 0 only. **confirm** half: the stage 2 test only |

Splitting the eval keeps choosing a setup (stage 0) apart from testing it (stage 2). The
confirm half is not looked at until stage 2.

## Metrics

- **Primary:** cost saved at 99% of all-gpt-4o quality, as a paired bootstrap difference
  against "without" (1,000 resamples). At 95% every router ties, since gpt-4o-mini alone reaches
  95.2%.
- **Secondary:** ROC-AUC against SPROUT's verdict (`cheap < 0.5 <= strong`).
- **Controls**, on every comparison:
  - **shuffled:** the same prompts with their labels permuted, at the same positive rate. This
    separates "in-domain prompts help" from "the labels help".
  - **noise band:** "without" refit with 5 seeds. Its spread in cost saved is the retraining
    noise.

## Stage 0: is any label set worth buying? (free)

SPROUT's own scores label the pool. That is the best any judge could match.

- **Definitions:**
  - **G:** cheap < 0.5 <= strong (about 8% positive)
  - **S:** strong strictly better (about 13% positive)
- **Sizes:** N = 500, 1,000, 2,000
- **Weights:** 1 and 5 (the pool is small next to 32k base rows)

A setup **passes** on the dev half if all of these hold:
1. The 95% interval of the difference is above 0.
2. Its mean beats its shuffled control by more than the noise band.
3. Its AUC is no lower than the "without" AUC.

**Decision:**
- If one or more setups pass, take the one with the smallest N, then weight 1 before 5, then G
  before S. That fixes the definition, N and weight for stage 2.
- **If none passes: stop.** Even perfect labels don't help at a size worth paying for. No money
  is spent, and the negative result is recorded.

## Stage 1: does Haiku measure what SPROUT measures? (pilot, about $0.10)

Haiku scores 200 pool pairs with tierwise's own judge prompt and seeded A/B order, plus two
checks:
- **Identical pair:** 20 pairs where both answers are gpt-4o's. These should score 4 or 5.
- **Order flip:** 20 pairs judged a second time with A and B swapped. The two scores should
  differ by at most 1.

**Go** to stage 2 only if all of these hold:
1. At least 18 of the 20 identical pairs score 4 or more.
2. At least 17 of the 20 order flips differ by 1 or less.
3. Haiku's score separates the chosen definition's labels with AUC of at least 0.70 on the 200.

**The cutoff is fixed here:** the Haiku score threshold whose positive rate on the 200 is
closest to the chosen definition's rate. It doesn't change after this.

## Stage 2: the test (paid; budget fixed after stage 1)

Haiku scores the chosen N pool pairs. The pilot pairs count toward N. The budget is N times the
pilot's measured cost per call, plus 20%, enforced by `--cap-usd`.

**One primary test, on the confirm half:** Haiku labels (chosen definition, cutoff, weight)
against "without". It **succeeds** if both of these hold:
1. The 95% interval is above 0.
2. The mean beats the shuffled control by more than the noise band.

Reported alongside, not as tests:
- SPROUT's own labels on the same prompts (the ceiling)
- AUC
- the cost

## What could still go wrong

- **SPROUT's judge isn't ground truth.** "Better" here means "agrees with SPROUT's LLM judge".
  Stage 1 measures how closely Haiku follows it.
- **Single-turn, two models, one domain.** A pass shows the mechanism works on this pair. It
  doesn't show the size of the gain on other traffic.
- **The confirm half is used once.** Further runs against it would turn it into a dev set.
