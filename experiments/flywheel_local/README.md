# Local flywheel experiment

Do labels from the data flywheel make the router better? This runs the whole loop on free local
models and scores the result against [SPROUT](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT)'s
own gpt-4o-mini / gpt-4o quality scores, so the answer doesn't depend on our judge. Results and
their caveats: [BENCHMARK.md](../../BENCHMARK.md#the-flywheel-do-judge-labels-improve-the-router).

## Needs

- [Ollama](https://ollama.com), running, with the three models:

  ```bash
  ollama pull qwen2.5:1.5b   # basic: answers easy prompts
  ollama pull qwen2.5:7b     # flagship: answers hard prompts
  ollama pull mistral:7b     # judge: wrote neither answer
  ```

  On an 8 GB GPU (RTX 4060) all three run fully on the GPU, at about 3 s per 200 tokens.
- The learned router's encoder: `tierwise fetch-model`, or `uv run python -m training.export_onnx`.
- The base training data: `uv run python -m training.prepare_data --limit 40000`.

## Run

From the repo root:

```bash
uv run --group retrain --group train python experiments/flywheel_local/collect.py 800
uv run --group retrain --group train python experiments/flywheel_local/audit.py
uv run --group retrain --group train python experiments/flywheel_local/analyze.py
```

| Script | What it does | Time |
|---|---|---|
| `collect.py` | 800 SPROUT prompts through the gateway (`registry.yaml`), then `tierwise flywheel label` with mistral:7b as judge, then the first evaluation at 95% quality | about 3 h |
| `audit.py` | Checks every factor that could decide the result: whether the eval can tell routers apart, the judge's agreement with SPROUT, the labels it dropped, how much weight the labels carry | 1 min |
| `analyze.py` | Refits the classifier with each label set (judge `score < 4` and `< 3`, SPROUT's own verdict; once and 10×) and scores them on about 3,150 unseen SPROUT prompts at 99% quality, with paired bootstraps | 10 min (1 min once the embeddings are cached) |

**The paid follow-up** ([PROTOCOL.md](PROTOCOL.md), committed before each run; results in
`BENCHMARK.md`). The paid scripts read `JUDGE_API_KEY`, `JUDGE_BASE_URL` and `JUDGE_MODEL` from
`.env` (any OpenAI-compatible endpoint) and stop at a hard `--cap-usd`.

| Script | What it does | Cost |
|---|---|---|
| `rejudge.py` | Claude Haiku 5.5 re-scores the 497 stored qwen pairs, with the v1 judge | $0.15 |
| `analyze.py --judge MODEL` | Compares those labels with mistral's on the same prompts | free |
| `stage0.py` | Stage 0: can SPROUT's own perfect labels help at all, and at what size? | free |
| `samepair.py pilot` | Stage 1: Haiku judges SPROUT's gpt-4o-mini vs gpt-4o answers, plus the identical-answer and order-flip checks (`--dry-run` makes no calls) | $0.16 |
| `samepair2.py check` | Must reproduce stage 0 on the dev half before the confirm set is touched | free |
| `samepair2.py pilot [--cut c]` | [Protocol 2](PROTOCOL-2.md)/[3](PROTOCOL-3.md) stage 1 with tierwise 0.10's two-score judge | $0.17 |
| `samepair2.py test [--cut c]` | Stage 2: the remaining 1,600 pairs and the single test on the confirm set (refuses without a pilot GO) | $0.64 |

`rejudge.py` and `samepair.py` ran with the v1 judge, which scored only the cheaper answer. tierwise
0.10 replaced it, so they refuse to run on later versions; `git checkout f121ae2` reruns them.

`collect.py`'s own phase 3 trains on its labels through `retrain.shadow_rows`. From 0.10 that skips labels made before 0.10, so it finds none in a database labeled under 0.9.

`collect.py` is resumable: traffic is skipped once samples exist, and labeling picks up whatever
is still pending. Everything it writes (`flywheel.db`, `router/`, `*.npz`, `*.log`) stays in this
folder and out of git.

## Why the first evaluation is kept

`collect.py`'s own evaluation reports no gain, and that is the point: at 95% quality the cheap
model alone already reaches 97.6% of the strong one on these prompts, so perfect, random and
constant routing all score 92.4%. `audit.py` shows it, and `analyze.py` measures where the
routers can differ. The same trap is why tierwise's gate now sets its target from the eval rows
(`evaluation/gate.py::quality_target`).
