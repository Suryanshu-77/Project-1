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

`collect.py` is resumable: traffic is skipped once samples exist, and labeling picks up whatever
is still pending. Everything it writes (`flywheel.db`, `router/`, `*.npz`, `*.log`) stays in this
folder and out of git.

## Why the first evaluation is kept

`collect.py`'s own evaluation reports no gain, and that is the point: at 95% quality the cheap
model alone already reaches 97.6% of the strong one on these prompts, so perfect, random and
constant routing all score 92.4%. `audit.py` shows it, and `analyze.py` measures where the
routers can differ. The same trap is why tierwise's gate now sets its target from the eval rows
(`evaluation/gate.py::quality_target`).
