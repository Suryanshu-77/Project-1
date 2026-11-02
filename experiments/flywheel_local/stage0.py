"""Stage 0 of PROTOCOL.md (free, no API calls): can any label set on gpt-4o-mini vs gpt-4o help?

SPROUT's own scores label prompts from its train split: the best labels any judge could match.
Each set (definition x N x weight) is scored on the eval's dev half against "without", with a
shuffled-label control and a 5-seed noise band; the protocol's rule picks a setup or says stop.
The confirm half is never loaded here.

Run from the repo root: uv run --group retrain --group train python <this file>
"""

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent

from evaluation.datasets import load_sprout  # noqa: E402
from evaluation.metrics import EvalRow, cost_saved_at, sweep  # noqa: E402
from training import retrain  # noqa: E402
from training.prepare_data import DATA_DIR, normalize  # noqa: E402
from training.train import _request, fit  # noqa: E402

TARGET, BOOTSTRAPS, SEEDS = 0.99, 1000, (0, 1, 2, 3, 4)
SIZES, WEIGHTS = (500, 1000, 2000), (1, 5)
SHIPPED = REPO / "models" / "router"
BASE_CACHE, CACHE = HERE / "emb_cache.npz", HERE / "stage0_cache.npz"
PROBS = HERE / "stage0_probs.npz"  # fitted predictions: fixing the statistics never refits


def usable(prompt: str, exclude: set[str]) -> bool:
    return 20 <= len(prompt) <= 1500 and normalize(prompt) not in exclude


def half(prompt: str) -> str:
    """dev or confirm, by prompt hash: fixed before any result is seen."""
    digest = hashlib.sha256(normalize(prompt).encode("utf-8")).digest()
    return "dev" if digest[0] % 2 == 0 else "confirm"


def gold(r: EvalRow) -> int:  # G: SPROUT says gpt-4o-mini fell short where gpt-4o didn't
    return int(r.q_cheap < 0.5 <= r.q_strong)


def better(r: EvalRow) -> int:  # S: SPROUT scores gpt-4o strictly higher
    return int(r.q_strong > r.q_cheap)


def saved_at(cols: tuple[np.ndarray, ...], scores: np.ndarray, target: float) -> float | None:
    """evaluation.metrics.cost_saved_at(sweep(rows, scores), target), vectorized: same cuts at
    tie groups, same (cost, frac_strong) order, same tolerance and interpolation."""
    qc, qs, cc, cs = cols
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    dq = np.concatenate([[0.0], np.cumsum((qs - qc)[order])])
    dc = np.concatenate([[0.0], np.cumsum((cs - cc)[order])])
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(ordered) != 0) + 1, [n]])
    quality, cost = (qc.sum() + dq[cuts]) / n, (cc.sum() + dc[cuts]) / n
    strong_q, strong_c = qs.mean(), cs.mean()
    qf = quality / strong_q if strong_q else np.zeros_like(quality)
    cf = cost / strong_c if strong_c else np.zeros_like(cost)
    o = np.lexsort((cuts / n, cost))
    qf, cf = qf[o], cf[o]
    hit = np.flatnonzero(qf >= target - 1e-9)
    if not len(hit):
        return None
    i = int(hit[0])
    if i == 0 or qf[i] <= target:
        return float(1.0 - cf[i])
    t = (target - qf[i - 1]) / (qf[i] - qf[i - 1])
    return float(1.0 - (cf[i - 1] + t * (cf[i] - cf[i - 1])))


def main() -> None:
    train_rows, _ = load_sprout("train")
    test_rows, _ = load_sprout("test")
    both = {normalize(r.prompt) for r in train_rows} & {normalize(r.prompt) for r in test_rows}
    exclude = both | {normalize(p) for p in retrain._critical(retrain.CRITICAL)}
    pool, seen = [], set()
    for r in train_rows:  # duplicates within the split would count one prompt twice
        if usable(r.prompt, exclude) and (key := normalize(r.prompt)) not in seen:
            seen.add(key)
            pool.append(r)
    random.Random(0).shuffle(pool)
    pool = pool[: max(SIZES)]  # nested: the first 500 are in the 1,000, and so on
    dev = [r for r in test_rows if usable(r.prompt, exclude) and half(r.prompt) == "dev"]
    print(f"train/test overlap excluded: {len(both)}; pool {len(pool)}; dev rows {len(dev)}, "
          f"gold positives {sum(map(gold, dev))}")  # fmt: skip

    base = np.load(BASE_CACHE)
    x_tr, x_va = base["tr"], base["va"]
    _, y_tr = retrain._base(DATA_DIR, "train")
    _, y_va = retrain._base(DATA_DIR, "val")
    y_tr, y_va = np.array(y_tr), np.array(y_va)
    assert len(x_tr) == len(y_tr) and len(x_va) == len(y_va), "stale emb_cache.npz"
    if CACHE.exists():
        z = np.load(CACHE)
        x_pool, x_dev = z["pool"], z["dev"]
        assert len(x_pool) == len(pool) and len(x_dev) == len(dev), "stale cache: delete it"
    else:
        embed = retrain.default_embedder(SHIPPED)
        x_pool = embed([_request(r.prompt) for r in pool])
        x_dev = embed([_request(r.prompt) for r in dev])
        np.savez(CACHE, pool=x_pool, dev=x_dev)

    cols = tuple(np.array([getattr(r, c) for r in dev], dtype=np.float64)
                 for c in ("q_cheap", "q_strong", "cost_cheap", "cost_strong"))  # fmt: skip
    y_dev = np.array([gold(r) for r in dev])

    def score(x_add: np.ndarray | None, y_add: np.ndarray | None, w: int = 1, seed: int = 0
              ) -> np.ndarray:  # fmt: skip
        if x_add is None or y_add is None:
            fitted = fit(x_tr, y_tr, x_va, y_va, seed=seed)
        else:
            fitted = fit(np.vstack([x_tr, *[x_add] * w]), np.concatenate([y_tr, *[y_add] * w]),
                         x_va, y_va, seed=seed)  # fmt: skip
        return np.asarray(fitted.model.predict_proba(x_dev)[:, 1])

    if PROBS.exists():
        probs = dict(np.load(PROBS))
    else:
        probs = {f"without s{s}": score(None, None, seed=s) for s in SEEDS}
        rng = np.random.default_rng(0)
        for d, label in (("G", gold), ("S", better)):
            y_pool = np.array([label(r) for r in pool])
            for n in SIZES:
                y, shuffled = y_pool[:n], rng.permutation(y_pool[:n])
                for w in WEIGHTS:
                    probs[f"{d} N{n} w{w}"] = score(x_pool[:n], y, w)
                    probs[f"{d} N{n} w{w} shuf"] = score(x_pool[:n], shuffled, w)
                    print(f"  fitted {d} N{n} w{w} (positive rate {y.mean():.1%})", flush=True)
        np.savez(PROBS, **probs)
    assert all(len(p) == len(dev) for p in probs.values()), "stale stage0_probs.npz"

    # the fast metric must agree with the reference before anything relies on it
    for p in probs.values():
        ref = cost_saved_at(sweep(dev, p), TARGET)
        assert ref is not None and abs(saved_at(cols, p, TARGET) - ref) < 1e-12  # type: ignore[operator]
    boot_rng = np.random.default_rng(1)  # one generator: each resample is a fresh draw
    boots = [boot_rng.integers(0, len(dev), len(dev)) for _ in range(BOOTSTRAPS)]
    assert len({b[:20].tobytes() for b in boots}) == BOOTSTRAPS, "resamples are not distinct"
    for idx in boots[:5]:
        sample = [dev[i] for i in idx]
        ref = cost_saved_at(sweep(sample, probs["without s0"][idx]), TARGET)
        assert abs(saved_at(tuple(c[idx] for c in cols), probs["without s0"][idx], TARGET)
                   - (ref or 0.0)) < 1e-12  # fmt: skip
    print("fast metric matches evaluation.metrics on every setup and 5 resamples")

    def boot(p: np.ndarray) -> np.ndarray:
        return np.array([saved_at(tuple(c[idx] for c in cols), p[idx], TARGET) or 0.0
                         for idx in boots])  # fmt: skip

    point = {k: saved_at(cols, p, TARGET) or 0.0 for k, p in probs.items()}
    auc = {k: float(roc_auc_score(y_dev, p)) for k, p in probs.items()}
    seed_points = [point[f"without s{s}"] for s in SEEDS]
    band = max(seed_points) - min(seed_points)
    base_boot = boot(probs["without s0"])
    print(f"\nwithout, 5 seeds: saved@99 {', '.join(f'{v:.4f}' for v in seed_points)}; "
          f"noise band {band * 100:.2f} pts; AUC {auc['without s0']:.4f}")  # fmt: skip
    print(f"\n{'setup':<14}{'saved@99':>9}{'AUC':>8}{'d point':>9}{'d vs without [95% CI]':>28}"
          f"{'d shuffled':>12}{'margin':>9}  pass")  # fmt: skip
    results, passing = [], []
    for d in ("G", "S"):
        for n in SIZES:
            for w in WEIGHTS:
                key = f"{d} N{n} w{w}"
                diff = boot(probs[key]) - base_boot
                diff_shuf = boot(probs[f"{key} shuf"]) - base_boot
                lo, hi = np.percentile(diff, [2.5, 97.5])
                assert hi > lo, f"{key}: a zero-width interval means the bootstrap is broken"
                margin = diff.mean() - diff_shuf.mean()
                rules = (lo > 0, margin > band, auc[key] >= auc["without s0"])
                ok = all(rules)
                results.append({"setup": key, "saved_at_99": point[key], "auc": auc[key],
                                "diff_mean": diff.mean(), "ci": [lo, hi],
                                "diff_shuffled_mean": diff_shuf.mean(), "margin": margin,
                                "rules": rules, "pass": ok})  # fmt: skip
                if ok:
                    passing.append((n, w, d, key))
                ci = f"{diff.mean() * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"
                marks = "".join("y" if r else "n" for r in rules)
                d_point = (point[key] - point["without s0"]) * 100
                print(f"{key:<14}{point[key]:>9.4f}{auc[key]:>8.4f}{d_point:>+9.2f}{ci:>28}"
                      f"{diff_shuf.mean() * 100:>+12.2f}{margin * 100:>+9.2f}  {marks}"
                      f"{'  PASS' if ok else ''}", flush=True)  # fmt: skip

    chosen = min(passing)[3] if passing else None  # smallest N, then w1, then G before S
    print("\nrules: CI above 0 / beats shuffled by more than the noise band / AUC not lower")
    print(f"decision: {'go to stage 1 with ' + chosen if chosen else 'STOP: no setup passes'}")
    out = {"dev_rows": len(dev), "noise_band": band, "without_auc": auc["without s0"],
           "results": results, "chosen": chosen}  # fmt: skip
    (HERE / "stage0_results.json").write_text(json.dumps(out, indent=2, default=bool) + "\n",
                                              encoding="utf-8")  # fmt: skip


if __name__ == "__main__":
    main()
