"""Audit every factor that could have decided the first evaluation's result. Read-only: no
model calls.
Run from the repo root: uv run --group retrain --group train python <this file>
"""

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent

import collect  # noqa: E402  (same prompt sample, same seed)

from evaluation.metrics import cost_saved_at, sweep  # noqa: E402
from training.prepare_data import DATA_DIR, normalize, split_of  # noqa: E402

chosen = collect.sprout()
by_prompt = {normalize(r.prompt): r for r in chosen}
held = [r for r in chosen if split_of(r.prompt) != "train"]


def section(title: str) -> None:
    print(f"\n=== {title} ===")


section("1. gold scores: scale and base rate")
qc = np.array([r.q_cheap for r in chosen])
qs = np.array([r.q_strong for r in chosen])
print("distinct q values (cheap):", sorted(Counter(np.round(qc, 2)).items())[:12])
print(f"mean q_cheap {qc.mean():.3f}, q_strong {qs.mean():.3f}")
print(f"gold needs-strong (cheap<0.5<=strong): {np.mean((qc < 0.5) & (qs >= 0.5)):.1%}")
print(
    f"strong strictly better: {np.mean(qs > qc):.1%}; cheap strictly better: {np.mean(qc > qs):.1%}"
)

section("2. is the eval metric saturated on the 156 held-out prompts?")
hq_c = np.mean([r.q_cheap for r in held])
hq_s = np.mean([r.q_strong for r in held])
hc_c = np.mean([r.cost_cheap for r in held])
hc_s = np.mean([r.cost_strong for r in held])
print(f"all-cheap quality = {hq_c / hq_s:.1%} of all-strong; all-cheap cost = {hc_c / hc_s:.1%}")
print(f"=> cost saved by routing EVERYTHING cheap: {1 - hc_c / hc_s:.4f}")
oracle = np.array([r.q_strong - r.q_cheap for r in held])
rand = np.random.default_rng(0).random(len(held))
for name, scores in {"oracle": oracle, "random": rand, "constant": np.zeros(len(held))}.items():
    print(f"{name:<9} cost_saved_95 = {cost_saved_at(sweep(held, scores), 0.95)}")
for target in (0.98, 0.99, 1.0):
    o = cost_saved_at(sweep(held, oracle), target)
    r = cost_saved_at(sweep(held, rand), target)
    print(f"target {target:.2f}: oracle {o}, random {r}")

section("3. judge labels, including the ones parse_score rejected")
db = sqlite3.connect(HERE / "flywheel.db")
rows = db.execute(
    "select request_json, judge_score, error, routed_model from shadow_samples"
).fetchall()
scores: list[tuple[int, int]] = []  # (judge score, gold)
recovered = 0
for request_json, judge_score, error, _ in rows:
    prompt = json.loads(request_json)["messages"][-1]["content"]
    row = by_prompt.get(normalize(prompt))
    if row is None:
        continue
    if judge_score is None and error and (m := re.match(r"judge score '(\d)'", error)):
        judge_score, recovered = int(m.group(1)), recovered + 1
    if judge_score is not None:
        scores.append((int(judge_score), collect.gold(row)))
print(f"labels usable: {len(scores)} ({recovered} recovered from string scores)")
print("routed:", Counter(r[3] for r in rows))
s = np.array(scores)
for cut in (3, 4, 5):
    pred = s[:, 0] < cut
    print(
        f"needs_strong = score < {cut}: rate {pred.mean():.0%}, "
        f"agree {np.mean(pred == s[:, 1]):.0%}, "
        f"recall on gold positives {pred[s[:, 1] == 1].mean():.0%}"
    )
print("gold rate among judged:", f"{s[:, 1].mean():.1%}")
print("score histogram gold=0:", sorted(Counter(s[s[:, 1] == 0, 0]).items()))
print("score histogram gold=1:", sorted(Counter(s[s[:, 1] == 1, 0]).items()))

section("4. how much weight can 400 labels carry?")
labels = [json.loads(line)["label"] for line in (DATA_DIR / "train.jsonl").open(encoding="utf-8")]
print(f"base train rows {len(labels)}, needs-strong rate {np.mean(labels):.1%}")
n_shadow = sum(1 for r in chosen if split_of(r.prompt) == "train")
print(
    f"SPROUT train-hash prompts sent: {n_shadow} = {n_shadow / len(labels):.2%} of the training set"
)
val = [json.loads(line)["label"] for line in (DATA_DIR / "val.jsonl").open(encoding="utf-8")]
print(f"threshold picked on base val only: {len(val)} rows, rate {np.mean(val):.1%}")
