"""PROTOCOL-2.md: the same-pair test with tierwise 0.10's two-score judge.

    uv run --group retrain --group train python <this file> check
    uv run --group retrain --group train python <this file> pilot [--dry-run]
    uv run --group retrain --group train python <this file> test [--dry-run]

`check` (free): the analysis pipeline must reproduce stage 0's G N2000 w5 result on the dev half
before it is trusted on the confirm set. `pilot` is stage 1, `test` is stage 2 (it refuses to run
unless the pilot said GO). Every judgment is stored in `same_pair_v2` of flywheel.db, so a re-run
never pays twice; each stage stops at its hard cap (list-price USD).
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from functools import partial
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent

from pairs import answers, env, flipped_seed, pid  # noqa: E402
from stage0 import build_pool, gold, half, saved_at, usable  # noqa: E402

from evaluation.datasets import load_sprout  # noqa: E402
from evaluation.metrics import EvalRow  # noqa: E402
from tierwise.flywheel import _transcript, judge_messages, needs_strong, parse_scores  # noqa: E402
from tierwise.providers.base import ProviderError  # noqa: E402
from tierwise.providers.openai import OpenAIProvider  # noqa: E402
from tierwise.reliability.retry import RetryPolicy, with_retries  # noqa: E402
from tierwise.schema import ChatRequest  # noqa: E402
from training import retrain  # noqa: E402
from training.prepare_data import DATA_DIR, normalize  # noqa: E402
from training.train import _load, _request, fit  # noqa: E402

PILOT, CHECKS, N, WEIGHT = 400, 20, 2000, 5
TARGET, BOOTSTRAPS, SEEDS = 0.99, 1000, (0, 1, 2, 3, 4)
CAPS = {"pilot": 0.20, "test": 0.65}  # list-price USD, PROTOCOL-2.md
PRICE_IN, PRICE_OUT = 0.10, 0.50  # per 1M tokens: claude-haiku-5.5 on AICredits
MAX_UNSCORED = 0.05
DB = HERE / "flywheel.db"
SHIPPED = REPO / "models" / "router"
Job = tuple[str, str, str, str, str]  # key, prompt, cheap answer, strong answer, seed


def table(db: sqlite3.Connection) -> None:
    db.execute("create table if not exists same_pair_v2 (key text primary key, stage text, "
               "cheap integer, strong integer, error text, cost_usd real, input_tokens integer, "
               "ts real)")  # fmt: skip


async def judge(jobs: list[Job], stage: str) -> None:
    cfg = env()
    db = sqlite3.connect(DB)
    table(db)
    done = {k for (k,) in db.execute("select key from same_pair_v2")}
    spent = db.execute("select coalesce(sum(cost_usd), 0) from same_pair_v2 where stage = ?",
                       (stage,)).fetchone()[0]  # fmt: skip
    todo = [j for j in jobs if j[0] not in done]
    cap = CAPS[stage]
    print(f"{stage}: {len(todo)} calls to make ({len(jobs) - len(todo)} stored); "
          f"spent ${spent:.4f} of the ${cap:.2f} cap", flush=True)  # fmt: skip
    lock, gate = asyncio.Lock(), asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=120) as client:
        provider = OpenAIProvider(client, cfg["JUDGE_API_KEY"], cfg["JUDGE_BASE_URL"])

        async def one(key: str, prompt: str, cheap: str, strong: str, seed: str) -> None:
            nonlocal spent
            async with gate:
                if spent >= cap:
                    return
                request = ChatRequest.model_validate(
                    {"messages": [{"role": "user", "content": prompt}]})  # fmt: skip
                messages, cheap_key = judge_messages(_transcript(request), cheap, strong, seed)
                grading = ChatRequest(model=cfg["JUDGE_MODEL"], temperature=0, messages=messages)
                scores: tuple[int | None, int | None] = (None, None)
                error, cost, tokens = None, 0.0, 0
                try:
                    call = partial(provider.chat, grading, cfg["JUDGE_MODEL"])
                    reply = await with_retries(call, RetryPolicy())
                    tokens = reply.usage.input_tokens
                    cost = (tokens * PRICE_IN + reply.usage.output_tokens * PRICE_OUT) / 1e6
                    a, b = parse_scores(reply.content or "")
                    scores = (a, b) if cheap_key == "a" else (b, a)
                except (ProviderError, ValueError) as exc:
                    error = str(exc)[:300]
                async with lock:
                    spent += cost
                    db.execute("insert or replace into same_pair_v2 values (?,?,?,?,?,?,?,?)",
                               (key, stage, *scores, error, cost, tokens, time.time()))  # fmt: skip
                    db.commit()

        await asyncio.gather(*(one(*j) for j in todo))
    print(f"{stage}: spent ${spent:.4f}", flush=True)
    if spent >= cap:
        print(f"stopped at the ${cap:.2f} cap: nothing is concluded; re-run to finish")


def stored() -> dict[str, tuple[int, int]]:
    db = sqlite3.connect(DB)
    table(db)
    return {k: (c, s) for k, c, s in db.execute(
        "select key, cheap, strong from same_pair_v2 where cheap is not null")}  # fmt: skip


def label_jobs(pool: list[EvalRow], texts: dict[str, tuple[str, str]]) -> list[Job]:
    return [(f"label:{pid(r.prompt)}", r.prompt, *texts[pid(r.prompt)], pid(r.prompt))
            for r in pool]  # fmt: skip


def dry(jobs: list[Job]) -> None:
    chars = np.array([len(j[1]) + len(j[2]) + len(j[3]) for j in jobs])
    print(f"dry run: {len(jobs)} calls, empty answers {sum(not j[2] or not j[3] for j in jobs)}, "
          f"~{chars.mean() / 4 + 250:.0f} input tokens per call, ~${len(jobs) * 0.00036:.2f} "
          f"(~Rs {len(jobs) * 0.040:.0f}) at Protocol 1's rate; no calls made")  # fmt: skip


def pilot(dry_run: bool) -> None:
    pool = build_pool(*(load_sprout(s)[0] for s in ("train", "test")))[0]
    texts = answers(pool[:PILOT])
    jobs = label_jobs(pool[:PILOT], texts)
    for r in pool[:CHECKS]:
        p, (cheap, strong) = pid(r.prompt), texts[pid(r.prompt)]
        jobs.append((f"identical:{p}", r.prompt, strong, strong, p))
        jobs.append((f"flip:{p}", r.prompt, cheap, strong, flipped_seed(p)))
    if dry_run:
        flips = [j for j in jobs if j[0].startswith("flip:")]
        swapped = all(judge_messages("q", "c", "s", j[4])[1] != judge_messages("q", "c", "s",
                      j[0][5:])[1] for j in flips)  # fmt: skip
        print(f"every flip shows the answers in the other order: {swapped}")
        return dry(jobs)
    asyncio.run(judge(jobs, "pilot"))

    got = stored()
    unscored = sum(j[0] not in got for j in jobs)
    labeled = [r for r in pool[:PILOT] if f"label:{pid(r.prompt)}" in got]
    cs = np.array([got[f"label:{pid(r.prompt)}"] for r in labeled])
    y = np.array([gold(r) for r in labeled])
    ident = [got.get(f"identical:{pid(r.prompt)}") for r in pool[:CHECKS]]
    flips = [(got.get(f"label:{pid(r.prompt)}"), got.get(f"flip:{pid(r.prompt)}"))
             for r in pool[:CHECKS]]  # fmt: skip
    never = sum(s is not None and not needs_strong(*s) for s in ident)
    equal = sum(s is not None and s[0] == s[1] for s in ident)
    same_label = sum(a is not None and b is not None and needs_strong(*a) == needs_strong(*b)
                     for a, b in flips)  # fmt: skip
    close = sum(a is not None and b is not None and max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
                for a, b in flips)  # fmt: skip
    auc = float(roc_auc_score(y, cs[:, 1] - cs[:, 0]))
    rate = float(np.mean([needs_strong(c, s) for c, s in cs]))
    checks = {
        "identical: needs_strong never fires (>= 19/20)": (never, never >= 19),
        "identical: equal scores (>= 18/20)": (equal, equal >= 18),
        "flip: same label (>= 17/20)": (same_label, same_label >= 17),
        "flip: each score within 1 (>= 17/20)": (close, close >= 17),
        f"gap AUC vs SPROUT's G (>= 0.70)": (round(auc, 3), auc >= 0.70),  # noqa: F541
        f"label rate {rate:.1%} within 0.5-2x of G's {y.mean():.1%}":
            (round(rate, 3), 0.5 * y.mean() <= rate <= 2 * y.mean()),
        f"unscored calls {unscored}/{len(jobs)} (<= 5%)":
            (unscored, unscored <= MAX_UNSCORED * len(jobs)),
    }  # fmt: skip
    go = all(ok for _, ok in checks.values())
    for name, (value, ok) in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {value}")
    print(f"\ndecision: {'GO to stage 2' if go else 'NO-GO: stop, stage 2 is not run'}")
    out = {"checks": {k: [v, bool(ok)] for k, (v, ok) in checks.items()}, "auc": auc, "rate": rate,
           "g_rate": float(y.mean()), "unscored": unscored, "go": bool(go)}  # fmt: skip
    (HERE / "stage1_v2_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def compare(name: str, rows: list[EvalRow], x_eval: np.ndarray, x_add: np.ndarray,
            labels: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:  # fmt: skip
    """Refit with each label set (x WEIGHT) and its shuffled control; paired bootstrap against
    "without" (seed 0); 5-seed noise band. PROTOCOL.md's metrics, on `rows`."""
    base = np.load(HERE / "emb_cache.npz")
    _, y_tr = retrain._base(DATA_DIR, "train")
    _, y_va = retrain._base(DATA_DIR, "val")
    x_tr, x_va, y_tr, y_va = base["tr"], base["va"], np.array(y_tr), np.array(y_va)
    assert len(x_tr) == len(y_tr), "stale emb_cache.npz"

    def predict(y_add: np.ndarray | None, seed: int = 0) -> np.ndarray:
        if y_add is None:
            fitted = fit(x_tr, y_tr, x_va, y_va, seed=seed)
        else:
            x = np.vstack([x_tr, *[x_add] * WEIGHT])
            fitted = fit(x, np.concatenate([y_tr, *[y_add] * WEIGHT]), x_va, y_va, seed=seed)
        return np.asarray(fitted.model.predict_proba(x_eval)[:, 1])

    cols = tuple(np.array([getattr(r, c) for r in rows], dtype=np.float64)
                 for c in ("q_cheap", "q_strong", "cost_cheap", "cost_strong"))  # fmt: skip
    rng = np.random.default_rng(1)
    boots = [rng.integers(0, len(rows), len(rows)) for _ in range(BOOTSTRAPS)]
    assert len({b[:20].tobytes() for b in boots}) == BOOTSTRAPS, "resamples are not distinct"

    def boot(p: np.ndarray) -> np.ndarray:
        return np.array([saved_at(tuple(c[i] for c in cols), p[i], TARGET) or 0.0 for i in boots])

    without = {s: predict(None, s) for s in SEEDS}
    points = [saved_at(cols, without[s], TARGET) or 0.0 for s in SEEDS]
    band = max(points) - min(points)
    base_boot = boot(without[0])
    g_eval = np.array([gold(r) for r in rows])
    print(f"\n{name}: {len(rows)} rows, {int(g_eval.sum())} G positives; without saved@99 "
          f"{points[0]:.4f}, AUC {roc_auc_score(g_eval, without[0]):.4f}; noise band "
          f"{band * 100:.2f} pts")  # fmt: skip
    shuffle = np.random.default_rng(0)
    out: dict[str, dict[str, float]] = {}
    for key, y in labels.items():
        p, p_shuf = predict(y), predict(shuffle.permutation(y))
        diff, diff_shuf = boot(p) - base_boot, boot(p_shuf) - base_boot
        lo, hi = np.percentile(diff, [2.5, 97.5])
        assert hi > lo, "a zero-width interval means the bootstrap is broken"
        margin = float(diff.mean() - diff_shuf.mean())
        out[key] = {"saved_at_99": saved_at(cols, p, TARGET) or 0.0,
                    "auc": float(roc_auc_score(g_eval, p)), "rate": float(y.mean()),
                    "diff_mean": float(diff.mean()), "lo": float(lo), "hi": float(hi),
                    "diff_shuffled": float(diff_shuf.mean()), "margin": margin, "band": band,
                    "pass": bool(lo > 0 and margin > band)}  # fmt: skip
        r = out[key]
        print(f"  {key:<14} rate {r['rate']:.1%}  saved@99 {r['saved_at_99']:.4f}  AUC "
              f"{r['auc']:.4f}  d {r['diff_mean'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]  "
              f"shuffled {r['diff_shuffled'] * 100:+.2f}  margin {margin * 100:+.2f}  "
              f"{'PASS' if r['pass'] else 'no'}", flush=True)  # fmt: skip
    return out


def check() -> None:
    """The pipeline must reproduce stage 0's G N2000 w5 (+12.88 [+1.50, +27.66]) on dev."""
    train_rows, test_rows = (load_sprout(s)[0] for s in ("train", "test"))
    pool, exclude, _ = build_pool(train_rows, test_rows)
    dev = [r for r in test_rows if usable(r.prompt, exclude) and half(r.prompt) == "dev"]
    z = np.load(HERE / "stage0_cache.npz")
    out = compare("dev (stage 0 reproduction)", dev, z["dev"], z["pool"],
                  {"G": np.array([gold(r) for r in pool])})  # fmt: skip
    g = out["G"]
    ok = abs(g["diff_mean"] - 0.1288) < 0.0005 and abs(g["lo"] - 0.0150) < 0.0005
    print(f"\nreproduces stage 0: {ok}")
    if not ok:
        raise SystemExit("the analysis does not match stage 0: do not run stage 2")


def confirm_set(train_rows: list[EvalRow], test_rows: list[EvalRow], pool: list[EvalRow],
                exclude: set[str]) -> list[EvalRow]:  # fmt: skip
    """Amendment 1: the test split's confirm half, plus every train prompt outside the pool,
    minus anything in the base data."""
    base = {normalize(p) for s in ("train", "val", "test") for p in _load(s)[0]}
    taken = {normalize(r.prompt) for r in pool} | base
    rows, seen = [], set()
    for r in test_rows:
        key = normalize(r.prompt)
        if usable(r.prompt, exclude) and half(r.prompt) == "confirm" and key not in base:
            rows.append(r)
            seen.add(key)
    for r in train_rows:
        key = normalize(r.prompt)
        if usable(r.prompt, exclude) and key not in taken and key not in seen:
            seen.add(key)
            rows.append(r)
    return rows


def test(dry_run: bool) -> None:
    result = HERE / "stage1_v2_results.json"
    if not dry_run and not (result.exists() and json.loads(result.read_text())["go"]):
        raise SystemExit("stage 2 runs only after a GO from the pilot (stage1_v2_results.json)")
    train_rows, test_rows = (load_sprout(s)[0] for s in ("train", "test"))
    pool, exclude, _ = build_pool(train_rows, test_rows)
    texts = answers(pool)
    jobs = label_jobs(pool[PILOT:N], texts)
    rows = confirm_set(train_rows, test_rows, pool, exclude)
    if dry_run:
        print(f"confirm set: {len(rows)} rows (outcomes not looked at)")
        return dry(jobs)
    asyncio.run(judge(jobs, "test"))

    got = stored()
    scored = [i for i, r in enumerate(pool[:N]) if f"label:{pid(r.prompt)}" in got]
    unscored = N - len(scored)
    if unscored > MAX_UNSCORED * N:
        print(f"INCOMPLETE: {unscored}/{N} pairs unscored; no conclusion")
        return
    haiku = np.array([needs_strong(*got[f"label:{pid(pool[i].prompt)}"]) for i in scored], int)
    g = np.array([gold(pool[i]) for i in scored])
    cache = HERE / "samepair2_confirm.npz"
    if cache.exists():
        x_eval = np.load(cache)["x"]
        assert len(x_eval) == len(rows), "stale samepair2_confirm.npz"
    else:
        embed = retrain.default_embedder(SHIPPED)
        x_eval = embed([_request(r.prompt) for r in rows])
        np.savez(cache, x=x_eval)
    x_pool = np.load(HERE / "stage0_cache.npz")["pool"][scored]
    print(f"labels: {len(scored)} scored ({unscored} unscored); Haiku rate {haiku.mean():.1%}, "
          f"G rate {g.mean():.1%}, agreement {np.mean(haiku == g):.1%}")  # fmt: skip
    out = compare("confirm set (the primary test)", rows, x_eval, x_pool,
                  {"haiku (primary)": haiku, "G (ceiling)": g})  # fmt: skip
    verdict = "SUCCESS" if out["haiku (primary)"]["pass"] else "NOT SUPPORTED"
    print(f"\nprimary test: {verdict}")
    agreement = float(np.mean(haiku == g))
    out_json = {"scored": len(scored), "unscored": unscored, "agreement": agreement,
                "results": out, "verdict": verdict}  # fmt: skip
    (HERE / "stage2_v2_results.json").write_text(json.dumps(out_json, indent=2) + "\n",
                                                 encoding="utf-8")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["check", "pilot", "test"])
    parser.add_argument("--dry-run", action="store_true", help="build the calls, make none")
    args = parser.parse_args()
    if args.stage == "check":
        check()
    elif args.stage == "pilot":
        pilot(args.dry_run)
    else:
        test(args.dry_run)


if __name__ == "__main__":
    main()
