"""Retrain the learned router from flywheel labels, gate it, and promote it (spec §5.2).

Usage: uv run --group retrain python -m training.retrain [--db-url URL] [--router-dir DIR]
           [--registry PATH] [--dry-run]

Labeled shadow samples (from `tierwise flywheel label`) are split by prompt hash like the base
dataset, merged with training/data when it exists, and a challenger is fitted. Champion and
challenger score the same held-out rows; `evaluation.gate.should_promote` decides. Every
challenger is kept in <router-dir>/versions/<UTC timestamp>/; promoting copies it over the
router dir (the replaced champion is kept too). The gateway loads it on its next restart.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.datasets import load_gpt4_test
from evaluation.gate import GateResult, evaluate_classifier, should_promote
from evaluation.metrics import EvalRow
from tierwise.config import Settings, registry_file, router_directory
from tierwise.registry import ModelSpec, Registry, load_registry
from tierwise.router.classifier import Classifier
from tierwise.router.encoder import OnnxEncoder, load_tokenizer
from tierwise.router.learned import model_path
from tierwise.schema import ChatRequest
from tierwise.store import RequestLog, ShadowSample
from tierwise.tokens import text_of
from training.prepare_data import DATA_DIR, Split, normalize, split_of
from training.train import _embed, _request, export_weights, fit

MIN_LABELS = 20
CRITICAL = Path(__file__).parents[1] / "tests" / "router" / "labeled_prompts.jsonl"
FILES = ("classifier.npz", "metrics.json", "router.json")  # router.json last: it names the rest
Embed = Callable[[list[ChatRequest]], np.ndarray]


class NotEnoughLabels(Exception):
    pass


@dataclass(frozen=True)
class ShadowRow:
    prompt: str  # the last user turn: what the split and the critical check key on
    request: ChatRequest  # the whole stored request: what the router embeds
    needs_strong: int
    row: EvalRow


@dataclass(frozen=True)
class RetrainReport:
    decision: bool
    reason: str
    champion: GateResult
    challenger: GateResult
    version: str
    promoted: bool
    n_labels: int


def prompt_of(request: ChatRequest) -> str:
    users = [m for m in request.messages if m.role == "user"]
    return text_of(users[-1].content) if users else ""


def _cost(spec: ModelSpec, prompt: str, answer: str) -> float:
    return (len(prompt) / 4 * spec.price_in + len(answer) / 4 * spec.price_out) / 1e6


def shadow_rows(
    samples: list[ShadowSample], registry: Registry, critical: set[str]
) -> dict[Split, list[ShadowRow]]:
    """Labeled samples as training and eval rows, split by prompt hash. Critical prompts are
    left out, so the gate's critical-misroute check never scores a prompt it trained on."""
    splits: dict[Split, list[ShadowRow]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        routed = registry.get(sample.routed_model)
        contrast = registry.get(sample.contrast_model or "")
        if routed is None or contrast is None or sample.needs_strong is None:
            continue  # a model left the registry: its prices are gone
        request = ChatRequest.model_validate_json(sample.request_json)
        prompt = prompt_of(request)
        if normalize(prompt) in critical:
            continue
        cheap_is_routed = contrast.tier == "flagship" or contrast.id == registry.baseline_model
        by_routed = ((routed, sample.routed_answer), (contrast, sample.contrast_answer or ""))
        (cheap, cheap_answer), (strong, strong_answer) = (
            by_routed if cheap_is_routed else by_routed[::-1]
        )
        label = int(sample.needs_strong)
        cost_cheap = _cost(cheap, prompt, cheap_answer)
        row = EvalRow(prompt, 0.0 if label else 1.0, 1.0, cost_cheap,
                      _cost(strong, prompt, strong_answer))  # fmt: skip
        splits[split_of(prompt)].append(ShadowRow(prompt, request, label, row))
    return splits


def default_embedder(router_dir: Path) -> Embed:
    """The gateway's own tokenizer, truncation and encoder: no train/serve skew."""
    tokenizer = load_tokenizer(router_dir / "tokenizer.json")
    encoder = OnnxEncoder(model_path(router_dir), threads=4)
    return lambda requests: _embed(encoder, tokenizer, requests)


def _critical(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r["prompt"] for r in rows if r["label"] == "hard"]


def _base(data_dir: Path, split: str) -> tuple[list[ChatRequest], list[int]]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        return [], []
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [_request(r["prompt"]) for r in rows], [int(r["label"]) for r in rows]


def _promote(router_dir: Path, new: Path, champion_version: str | None) -> None:
    """Keep the champion in versions/, then swap each file in with an atomic replace."""
    old = router_dir / "versions" / (champion_version or "initial")
    if not old.exists():
        old.mkdir(parents=True)
        for name in FILES:
            if (router_dir / name).exists():
                shutil.copy2(router_dir / name, old / name)
    for name in FILES:
        tmp = router_dir / f"{name}.tmp"
        shutil.copy2(new / name, tmp)
        os.replace(tmp, router_dir / name)


async def run(
    db_url: str,
    router_dir: Path,
    registry: Registry,
    promote: bool,
    data_dir: Path = DATA_DIR,
    embed: Embed | None = None,
    now: Callable[[], float] = time.time,
    critical_path: Path = CRITICAL,
) -> RetrainReport:
    log = RequestLog(db_url)
    await log.start()
    try:
        samples = await log.labeled_samples()
    finally:
        await log.stop()
    critical_prompts = _critical(critical_path)
    splits = shadow_rows(samples, registry, {normalize(p) for p in critical_prompts})
    n_labels = sum(len(rows) for rows in splits.values())
    if n_labels < MIN_LABELS:
        raise NotEnoughLabels(f"{n_labels} usable labeled samples; need at least {MIN_LABELS}")

    train_req, train_y = _base(data_dir, "train")
    val_req, val_y = _base(data_dir, "val")
    train_req += [r.request for r in splits["train"]]
    train_y += [r.needs_strong for r in splits["train"]]
    val_req += [r.request for r in splits["val"]]
    val_y += [r.needs_strong for r in splits["val"]]
    if min(sum(train_y), len(train_y) - sum(train_y)) < 2:
        raise NotEnoughLabels("training needs at least 2 samples of each label")
    if not val_req:  # tiny shadow sets: validate on train rather than not at all
        val_req, val_y = train_req, train_y

    eval_rows: list[EvalRow] = []
    eval_req: list[ChatRequest] = []
    if (data_dir / "test.jsonl").exists():
        eval_rows = load_gpt4_test(data_dir / "test.jsonl")
        eval_req = [_request(r.prompt) for r in eval_rows]
    held_out = splits["test"] or splits["val"]  # some held-out shadow rows to score on
    eval_rows += [r.row for r in held_out]
    eval_req += [r.request for r in held_out]
    if not eval_rows:
        raise NotEnoughLabels("no held-out labels to score champion and challenger on")

    embed = embed or default_embedder(router_dir)
    fitted = fit(embed(train_req), np.array(train_y), embed(val_req), np.array(val_y))
    x_eval, x_critical = embed(eval_req), embed([_request(p) for p in critical_prompts])

    config: dict[str, Any] = json.loads((router_dir / "router.json").read_text(encoding="utf-8"))
    dim = int(config["dim"])
    champion = Classifier.load(router_dir / "classifier.npz", expected_dim=dim)
    champion_result = evaluate_classifier(
        champion.predict_proba(x_eval), eval_rows, champion.predict_proba(x_critical),
        float(config["threshold"]),
    )  # fmt: skip

    version = datetime.fromtimestamp(now(), UTC).strftime("%Y%m%dT%H%M%SZ")
    new = router_dir / "versions" / version
    new.mkdir(parents=True, exist_ok=True)
    np.savez(new / "classifier.npz", **export_weights(fitted.model))
    challenger = Classifier.load(new / "classifier.npz", expected_dim=dim)  # score what ships
    challenger_result = evaluate_classifier(
        challenger.predict_proba(x_eval), eval_rows, challenger.predict_proba(x_critical),
        fitted.threshold,
    )  # fmt: skip
    decision, reason = should_promote(champion_result, challenger_result)

    metrics = {
        "version": version, "kind": fitted.kind, "threshold": fitted.threshold,
        "val_auc": fitted.val_auc, "n_labels": n_labels, "n_train": len(train_y),
        "n_eval": len(eval_rows), "decision": decision, "reason": reason,
        "champion": asdict(champion_result), "challenger": asdict(challenger_result),
    }  # fmt: skip
    (new / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    router_json = {**config, "threshold": fitted.threshold, "kind": fitted.kind, "version": version}
    (new / "router.json").write_text(json.dumps(router_json, indent=2) + "\n", encoding="utf-8")

    promoted = decision and promote
    if promoted:
        _promote(router_dir, new, config.get("version"))
    return RetrainReport(decision, reason, champion_result, challenger_result, version,
                         promoted, n_labels)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="python -m training.retrain")
    parser.add_argument("--db-url", default=settings.db_url)
    parser.add_argument("--router-dir", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="gate and version, never promote")
    args = parser.parse_args(argv)
    router_dir = args.router_dir or router_directory(settings)
    try:
        registry = load_registry(args.registry or registry_file(settings))
        report = asyncio.run(run(args.db_url, router_dir, registry, promote=not args.dry_run))
    except NotEnoughLabels as exc:
        print(f"retrain skipped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), indent=2))
    if report.promoted:
        print(f"promoted {report.version}; restart the gateway to load it", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
