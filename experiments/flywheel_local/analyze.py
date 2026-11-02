"""Analysis (no model calls): same prompts, different labels, an eval that can discriminate.

Labels on the SPROUT train-hash prompts already judged by collect.py:
  judge<4 / judge<3  mistral's scores (incl. quoted scores tierwise < 0.9.2 dropped)
  gold               SPROUT's real verdict (cheap<0.5<=strong): the label upper bound
each added once (x1) or duplicated 10 times (x10, since 400 rows are 2% of base train).
Eval: up to 3000 SPROUT test prompts never sent, plus collect.py's held-out ones, scored by real
quality at 99% (95% is saturated: all-cheap already reaches 97.6%) and ROC-AUC vs gold.

--judge MODEL compares rejudge.py's scores from MODEL with mistral's on the same prompts: only
the pairs both judged (the 497 whose answers were stored), so the judge is all that differs.
"""

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent

import collect  # noqa: E402

from evaluation.datasets import load_sprout  # noqa: E402
from evaluation.metrics import cost_saved_at, sweep  # noqa: E402
from training import retrain  # noqa: E402
from training.prepare_data import DATA_DIR, normalize, split_of  # noqa: E402
from training.train import _request, fit  # noqa: E402

TARGET, BOOTSTRAPS, EVAL_N = 0.99, 300, 3000
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--judge", help="a rejudge.py judge to compare with mistral, e.g. "
                    "anthropic/claude-haiku-5.5")  # fmt: skip
args = parser.parse_args()
CACHE = HERE / "emb_cache.npz"

chosen = collect.sprout()
sent = {normalize(r.prompt) for r in chosen}
by_prompt = {normalize(r.prompt): r for r in chosen}

rows, _ = load_sprout("test")
critical = {normalize(p) for p in retrain._critical(retrain.CRITICAL)}
unsent = [r for r in rows if 20 <= len(r.prompt) <= 1500 and normalize(r.prompt) not in sent
          and normalize(r.prompt) not in critical]  # fmt: skip
evals = random.Random(1).sample(unsent, min(EVAL_N, len(unsent)))
evals += [r for r in chosen if split_of(r.prompt) != "train"]

db = sqlite3.connect(HERE / "flywheel.db")
judged: list[tuple[str, int, str]] = []  # (prompt, mistral score, sample id)
for sid, request_json, score, error in db.execute(
        "select id, request_json, judge_score, error from shadow_samples"):  # fmt: skip
    prompt = json.loads(request_json)["messages"][-1]["content"]
    if score is None and error and (m := re.match(r"judge score '(\d)'", error)):
        score = int(m.group(1))
    if score is not None and split_of(prompt) == "train" and normalize(prompt) in by_prompt:
        judged.append((prompt, int(score), sid))
print(
    f"SPROUT unsent pool {len(unsent)}; eval rows {len(evals)}; "
    f"gold positives in eval {sum(collect.gold(r) for r in evals)}; train labels {len(judged)}"
)

if CACHE.exists():
    z = np.load(CACHE)
    x_tr, x_va, x_sh, x_ev = z["tr"], z["va"], z["sh"], z["ev"]
    assert len(x_sh) == len(judged) and len(x_ev) == len(evals), "stale cache: delete it"
else:
    embed = retrain.default_embedder(collect.SHIPPED)
    base_tr, _ = retrain._base(DATA_DIR, "train")
    base_va, _ = retrain._base(DATA_DIR, "val")
    x_tr, x_va = embed(base_tr), embed(base_va)
    x_sh = embed([_request(p) for p, _, _ in judged])
    x_ev = embed([_request(r.prompt) for r in evals])
    np.savez(CACHE, tr=x_tr, va=x_va, sh=x_sh, ev=x_ev)
_, y_tr = retrain._base(DATA_DIR, "train")
_, y_va = retrain._base(DATA_DIR, "val")
y_tr, y_va = np.array(y_tr), np.array(y_va)

if args.judge:
    other = dict(db.execute("select sample_id, score from rejudge where judge = ? and score is "
                            "not null", (args.judge,)).fetchall())  # fmt: skip
    # agreement with SPROUT's verdict on every pair both judged, held-out prompts included
    both = [(json.loads(rj)["messages"][-1]["content"], s, other[sid]) for sid, rj, s in
            db.execute("select id, request_json, judge_score from shadow_samples "
                       "where status = 'labeled'") if sid in other]  # fmt: skip
    gold_both = np.array([collect.gold(by_prompt[normalize(p)]) for p, _, _ in both])
    for name, col in (("mistral:7b", 1), (args.judge, 2)):
        pred = np.array([row[col] < 4 for row in both])
        print(f"{name}: agrees with SPROUT on {np.mean(pred == gold_both):.0%} of {len(both)}; "
              f"needs-strong rate {pred.mean():.0%} (SPROUT {gold_both.mean():.0%})")
    keep = [i for i, (_, _, sid) in enumerate(judged) if sid in other]
    judged, x_sh = [judged[i] for i in keep], x_sh[keep]
    score = np.array([s for _, s, _ in judged])
    theirs = np.array([other[sid] for _, _, sid in judged])
    labels = {
        "mistral<4": (score < 4).astype(int),
        "strong<4": (theirs < 4).astype(int),
        "strong<3": (theirs < 3).astype(int),
        "gold": np.array([collect.gold(by_prompt[normalize(p)]) for p, _, _ in judged]),
    }
    print(f"train labels both judged: {len(judged)} (strong = {args.judge})")
else:
    score = np.array([s for _, s, _ in judged])
    labels = {
        "judge<4": (score < 4).astype(int),
        "judge<3": (score < 3).astype(int),
        "gold": np.array([collect.gold(by_prompt[normalize(p)]) for p, _, _ in judged]),
    }
gold_eval = np.array([collect.gold(r) for r in evals])

probs: dict[str, np.ndarray] = {}
without = fit(x_tr, y_tr, x_va, y_va)
probs["without"] = without.model.predict_proba(x_ev)[:, 1]
for name, y in labels.items():
    for k in (1, 10):
        fitted = fit(np.vstack([x_tr, *[x_sh] * k]), np.concatenate([y_tr, *[y] * k]), x_va, y_va)
        probs[f"{name} x{k}"] = fitted.model.predict_proba(x_ev)[:, 1]
probs["oracle(gold)"] = np.array([r.q_strong - r.q_cheap for r in evals])

print("\nlabel rates: " + ", ".join(f"{n} {y.mean():.0%}" for n, y in labels.items()))
print(
    f"{'model':<14}{'saved@99':>10}{'saved@95':>10}{'auc(gold)':>11}"
    f"{'d@99 vs without [95% CI]':>34}{'wins':>8}"
)
rng = np.random.default_rng(0)
idx_all = [rng.integers(0, len(evals), len(evals)) for _ in range(BOOTSTRAPS)]
base_boot = [cost_saved_at(sweep([evals[i] for i in idx], probs["without"][idx]), TARGET) or 0.0
             for idx in idx_all]  # fmt: skip
for name, p in probs.items():
    s99 = cost_saved_at(sweep(evals, p), TARGET)
    s95 = cost_saved_at(sweep(evals, p), 0.95)
    auc = roc_auc_score(gold_eval, p)
    if name == "without":
        ci, wins = "", ""
    else:
        diffs = [(cost_saved_at(sweep([evals[i] for i in idx], p[idx]), TARGET) or 0.0) - b
                 for idx, b in zip(idx_all, base_boot, strict=True)]  # fmt: skip
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        ci = f"{np.mean(diffs) * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"
        wins = f"{sum(d > 0 for d in diffs)}/{BOOTSTRAPS}"
    print(f"{name:<14}{s99 or 0:>10.4f}{s95 or 0:>10.4f}{auc:>11.4f}{ci:>34}{wins:>8}", flush=True)
