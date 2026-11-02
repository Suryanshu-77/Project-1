"""Collect and label: do labels from an independent local judge improve routing?

SPROUT prompts (none in the base data, ~20% held out by prompt hash) go through the gateway
(answers up to 1024 tokens); mistral:7b, which wrote neither answer, judges them. Phase 3 is the
first, saturated evaluation at 95% quality, kept to reproduce it; analyze.py is the real one.

Resumable: phase 1 is skipped once samples exist; phase 2 labels whatever is pending.
Run from the repo root: uv run --group retrain --group train python <this file> [N]
"""

import asyncio
import json
import random
import shutil
import sys
import time
from pathlib import Path

import httpx
import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent

from evaluation.datasets import load_sprout  # noqa: E402
from evaluation.gate import evaluate_classifier  # noqa: E402
from evaluation.metrics import EvalRow, cost_saved_at, sweep  # noqa: E402
from tierwise.api.app import build_providers, create_app  # noqa: E402
from tierwise.config import Settings  # noqa: E402
from tierwise.flywheel import label_pending  # noqa: E402
from tierwise.keys import generate_key, hash_key  # noqa: E402
from tierwise.registry import ModelSpec, load_registry  # noqa: E402
from tierwise.router.classifier import Classifier  # noqa: E402
from tierwise.store import RequestLog  # noqa: E402
from training import retrain  # noqa: E402
from training.prepare_data import DATA_DIR, normalize, split_of  # noqa: E402
from training.train import _request, export_weights, fit  # noqa: E402

REGISTRY = load_registry(HERE / "registry.yaml")
JUDGE = ModelSpec.model_validate({  # not in the registry: the router must never route to it
    "id": "judge", "provider": "ollama", "upstream_id": "mistral:7b", "price_in": 1.0,
    "price_out": 4.0, "context": 32768, "caps": ["json"], "quality": 0.5, "tier": "moderate",
})  # fmt: skip
DB = f"sqlite+aiosqlite:///{(HERE / 'flywheel.db').as_posix()}"
ROUTER = HERE / "router"
SHIPPED = REPO / "models" / "router"
LONG, BOOTSTRAPS = 300.0, 500


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sprout(n: int = 800) -> list[EvalRow]:
    rows, _ = load_sprout("test")
    critical = {normalize(p) for p in retrain._critical(retrain.CRITICAL)}
    usable = [
        r for r in rows if 20 <= len(r.prompt) <= 1500 and normalize(r.prompt) not in critical
    ]
    return random.Random(0).sample(usable, min(n, len(usable)))


def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        db_url=DB,
        router_dir=str(ROUTER),
        registry_path=str(HERE / "registry.yaml"),
        ollama_base_url="http://localhost:11434/v1",
        require_api_key=True,
        store_payloads=True,
        shadow_rate=1.0,
        retries=1,
    )


async def send(chosen: list[EvalRow]) -> None:
    app = create_app(settings(), registry=REGISTRY)
    async with app.router.lifespan_context(app):
        log_ = app.state.log
        existing = len(await log_.pending_samples(100_000)) + len(await log_.labeled_samples())
        if existing:
            log(f"phase 1 skipped: {existing} samples already collected")
            return
        key = generate_key()
        await log_.create_key("exp", hash_key(key), key[:10], None, None, True)
        app.state.keys.replace(await log_.list_keys())
        auth = {"Authorization": f"Bearer {key}"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gw", timeout=LONG
        ) as client:
            routed: dict[str, int] = {}
            for i, row in enumerate(chosen, 1):
                payload = {
                    "model": "auto",
                    "max_tokens": 1024,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": row.prompt}],
                }
                resp = await client.post("/v1/chat/completions", json=payload, headers=auth)
                model = resp.headers.get("X-Tierwise-Model", f"error {resp.status_code}")
                routed[model] = routed.get(model, 0) + 1
                if i % 25 == 0:
                    log(f"sent {i}/{len(chosen)} (routed {routed})")
    log(f"traffic done: routed {routed}")


async def label() -> None:
    clients: list[httpx.AsyncClient] = []

    def client_for(_: str) -> httpx.AsyncClient:
        clients.append(httpx.AsyncClient(timeout=LONG))
        return clients[-1]

    log_ = RequestLog(DB)
    await log_.start()
    try:
        providers = build_providers(settings(), client_for)
        total = 0
        while True:
            run = await label_pending(log_, REGISTRY, providers, JUDGE, 1e9, 25)
            total += run.labeled + run.failed
            log(f"labeled {run.labeled}, failed {run.failed} (total {total})")
            if run.labeled + run.failed == 0:
                break
    finally:
        await log_.stop()
        for client in clients:
            await client.aclose()


def gold(row: EvalRow) -> int:
    """SPROUT's own verdict: gpt-4o-mini fell short where gpt-4o didn't."""
    return int(row.q_cheap < 0.5 <= row.q_strong)


def as_classifier(model: object, path: Path, dim: int) -> Classifier:
    np.savez(path, **export_weights(model))  # type: ignore[arg-type]
    return Classifier.load(path, expected_dim=dim)


async def ablation(chosen: list[EvalRow]) -> None:
    by_prompt = {normalize(r.prompt): r for r in chosen}
    log_ = RequestLog(DB)
    await log_.start()
    try:
        samples = await log_.labeled_samples()
    finally:
        await log_.stop()
    critical = retrain._critical(retrain.CRITICAL)
    splits = retrain.shadow_rows(samples, REGISTRY, {normalize(p) for p in critical})
    log("shadow rows: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))

    labeled = [r for rows in splits.values() for r in rows if normalize(r.prompt) in by_prompt]
    agree = sum(r.needs_strong == gold(by_prompt[normalize(r.prompt)]) for r in labeled)
    local = sum(r.needs_strong for r in labeled) / max(1, len(labeled))
    sprout_rate = sum(gold(by_prompt[normalize(r.prompt)]) for r in labeled) / max(1, len(labeled))
    log(
        f"mistral judge vs SPROUT's gpt-4o scores: agree {agree}/{len(labeled)} "
        f"= {agree / max(1, len(labeled)):.0%}; needs-strong rate {local:.0%} vs {sprout_rate:.0%}"
    )

    # held out = every SPROUT prompt we sent whose hash isn't train: never trained on, scored
    # with the REAL gpt-4o-mini / gpt-4o qualities, independent of our judge
    held = [r for r in chosen if split_of(r.prompt) != "train"]
    log(f"held-out SPROUT prompts (gold-scored): {len(held)}")

    config = json.loads((SHIPPED / "router.json").read_text(encoding="utf-8"))
    dim = int(config["dim"])
    embed = retrain.default_embedder(SHIPPED)
    base_train, y_train = retrain._base(DATA_DIR, "train")
    base_val, y_val = retrain._base(DATA_DIR, "val")
    log(f"embedding {len(base_train) + len(base_val)} base prompts")
    x_train, x_val = embed(base_train), embed(base_val)
    sh = splits["train"]
    x_sh = embed([r.request for r in sh]) if sh else np.zeros((0, dim), dtype=np.float32)
    y_sh = np.array([r.needs_strong for r in sh], dtype=int)
    x_held = embed([_request(r.prompt) for r in held])
    x_crit = embed([_request(p) for p in critical])

    log(f"fitting WITHOUT the {len(sh)} shadow labels, then WITH them")
    without = fit(x_train, np.array(y_train), x_val, np.array(y_val))
    with_ = fit(np.vstack([x_train, x_sh]), np.concatenate([y_train, y_sh]), x_val, np.array(y_val))
    models = {
        "champion": (Classifier.load(SHIPPED / "classifier.npz", dim), float(config["threshold"])),
        "without": (
            as_classifier(without.model, HERE / "ablation_without.npz", dim),
            without.threshold,
        ),
        "with": (as_classifier(with_.model, HERE / "ablation_with.npz", dim), with_.threshold),
    }
    p_held = {}
    print(f"\non {len(held)} held-out SPROUT prompts, real gpt-4o-mini/gpt-4o quality:")
    print(f"{'model':<10}{'cost_saved_95':>15}{'auc':>8}{'crit(hybrid)':>14}{'crit(clf)':>11}")
    for name, (clf, threshold) in models.items():
        p = clf.predict_proba(x_held)
        p_held[name] = p
        p_crit = clf.predict_proba(x_crit)
        hybrid = await retrain.critical_misroutes(critical, p_crit, threshold, REGISTRY)
        res = evaluate_classifier(p, held, hybrid, int((p_crit < threshold).sum()))
        saved = "None" if res.cost_saved_95 is None else f"{res.cost_saved_95:.4f}"
        auc = "None" if res.auc is None else f"{res.auc:.4f}"
        print(
            f"{name:<10}{saved:>15}{auc:>8}{res.critical_misroutes:>14}"
            f"{res.classifier_misroutes:>11}"
        )

    rng = np.random.default_rng(0)
    n, diffs, wins = len(held), [], 0
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        rows = [held[i] for i in idx]
        a = cost_saved_at(sweep(rows, p_held["with"][idx]), 0.95) or 0.0
        b = cost_saved_at(sweep(rows, p_held["without"][idx]), 0.95) or 0.0
        diffs.append(a - b)
        wins += a > b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"\nwith - without, cost saved @95% (gold), {BOOTSTRAPS} bootstraps: "
        f"mean {np.mean(diffs) * 100:+.2f} pts, 95% interval [{lo * 100:+.2f}, {hi * 100:+.2f}], "
        f"'with' wins {wins}/{BOOTSTRAPS}",
        flush=True,
    )


async def main() -> None:
    if not ROUTER.exists():
        shutil.copytree(SHIPPED, ROUTER, ignore=shutil.ignore_patterns("versions"))
    chosen = sprout(int(sys.argv[1]) if len(sys.argv) > 1 else 800)
    log(f"phase 1: {len(chosen)} SPROUT prompts through the gateway")
    await send(chosen)
    log("phase 2: contrast (qwen) + independent judge (mistral:7b)")
    await label()
    log("phase 3: ablation on held-out SPROUT prompts with real quality scores")
    await ablation(chosen)
    log("done")


if __name__ == "__main__":
    asyncio.run(main())
