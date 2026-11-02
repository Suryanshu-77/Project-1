"""Retrain the learned router from flywheel labels, gate it, and promote it (spec §5.2).

Usage: uv run --group retrain python -m training.retrain [--db-url URL] [--router-dir DIR]
           [--registry PATH] [--dry-run]

Labeled shadow samples (from `tierwise flywheel label`) are split by prompt hash like the base
dataset, merged with training/data when it exists, and a challenger is fitted. Champion and
challenger score the same held-out rows; `evaluation.gate.decide` decides. Every challenger
is kept, write-once, in <router-dir>/versions/<UTC timestamp>/. Promoting is one atomic replace
of router.json, which names its version's weights; an unversioned champion is first archived
under its content hash. The gateway loads the new router on its next restart.
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from evaluation.datasets import load_gpt4_test
from evaluation.gate import GateResult, decide, evaluate_classifier
from evaluation.metrics import EvalRow
from tierwise.affinity import AffinityTable
from tierwise.config import Settings, registry_file, router_directory
from tierwise.registry import ModelSpec, Registry, load_registry
from tierwise.router import RoutingDecision
from tierwise.router.classifier import Classifier
from tierwise.router.encoder import OnnxEncoder, load_tokenizer
from tierwise.router.learned import LearnedRouter, model_path, to_decision, weights_path
from tierwise.router.router import route
from tierwise.schema import ChatRequest
from tierwise.selector import select
from tierwise.store import RequestLog, ShadowSample
from tierwise.tokens import text_of
from training.prepare_data import DATA_DIR, Split, normalize, split_of
from training.train import _embed, _request, export_weights, fit

MIN_LABELS = 20
MIN_HELD_OUT = 20  # rows, with both labels, before a promotion can rest on them
CRITICAL = Path(__file__).parents[1] / "tests" / "router" / "labeled_prompts.jsonl"
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
    config = json.loads((router_dir / "router.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(router_dir / "tokenizer.json")
    encoder = OnnxEncoder(model_path(router_dir), threads=4)
    if encoder.dim != int(config["dim"]):
        raise ValueError(f"the encoder makes {encoder.dim}-dim vectors but router.json says "
                         f"{config['dim']}: fetch the matching model.onnx")  # fmt: skip
    return lambda requests: _embed(encoder, tokenizer, requests)


def _critical(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r["prompt"] for r in rows if r["label"] == "hard"]


class _Scored:
    """Stands in for LearnedRouter.decide with probabilities already computed for each prompt,
    so the gateway's own route() applies rules, the shortcut and escalation unchanged."""

    def __init__(self, p: dict[str, float], threshold: float) -> None:
        self._p, self._threshold = p, threshold

    async def decide(self, req: ChatRequest) -> RoutingDecision:
        difficulty, confidence = to_decision(self._p[prompt_of(req)], self._threshold)
        return RoutingDecision(difficulty=difficulty, confidence=confidence, source="learned")


async def critical_misroutes(
    prompts: list[str], p: np.ndarray, threshold: float, registry: Registry
) -> int:
    """Hard prompts the deployed router would send to a basic model with this classifier:
    the same check as the CI critical gate, through route() and select()."""
    scored = cast(LearnedRouter, _Scored(dict(zip(prompts, map(float, p), strict=True)), threshold))
    providers = {m.provider for m in registry.models}
    picks = 0
    for prompt in prompts:
        result = await route(_request(prompt), scored)
        option = select(result, registry, providers, AffinityTable()).candidates[0]
        picks += option.spec.tier == "basic"
    return picks


def _base(data_dir: Path, split: str) -> tuple[list[ChatRequest], list[int]]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        return [], []
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [_request(r["prompt"]) for r in rows], [int(r["label"]) for r in rows]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _free_version(versions: Path, version: str) -> str:
    """Versions are write-once (router.json files point into them): never reuse a name."""
    for suffix in ("", *(f"-{i}" for i in range(1, 100))):
        name = f"{version}{suffix}"
        if not (versions / name).exists() and not (versions / f"{name}.partial").exists():
            return name
    raise RuntimeError(f"too many versions named {version}")


def _archive_unversioned(router_dir: Path, config: dict[str, Any]) -> None:
    """Keep a champion that doesn't live in versions/ yet (the shipped router, or one from
    training/train.py), keyed by its weights' hash so two different ones never collide."""
    weights = weights_path(router_dir, config)
    name = "initial-" + hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
    target = router_dir / "versions" / name
    if target.exists():
        return
    staging = target.with_name(name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copy2(weights, staging / "classifier.npz")
    if (router_dir / "metrics.json").exists():
        shutil.copy2(router_dir / "metrics.json", staging / "metrics.json")
    archived = {**config, "version": name, "weights": f"versions/{name}/classifier.npz"}
    _write_json(staging / "router.json", archived)
    staging.rename(target)


def _promote(router_dir: Path, new: Path, champion: dict[str, Any]) -> None:
    """One os.replace of router.json is the commit point: before it the champion routes,
    after it the challenger does, never a mix of the two."""
    if "weights" not in champion:
        _archive_unversioned(router_dir, champion)
    tmp = router_dir / "router.json.tmp"
    shutil.copy2(new / "router.json", tmp)
    try:
        os.replace(tmp, router_dir / "router.json")
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    shutil.copy2(new / "metrics.json", router_dir / "metrics.json")  # informational only


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
    champion = Classifier.load(weights_path(router_dir, config), expected_dim=dim)
    champion_misroutes = await critical_misroutes(
        critical_prompts, champion.predict_proba(x_critical), float(config["threshold"]), registry
    )
    champion_p_critical = champion.predict_proba(x_critical)
    champion_result = evaluate_classifier(
        champion.predict_proba(x_eval), eval_rows, champion_misroutes,
        int((champion_p_critical < float(config["threshold"])).sum()),
    )  # fmt: skip

    versions = router_dir / "versions"
    version = _free_version(versions, datetime.fromtimestamp(now(), UTC).strftime("%Y%m%dT%H%M%SZ"))
    staging = versions / f"{version}.partial"  # renamed into place only once complete
    staging.mkdir(parents=True)
    try:
        np.savez(staging / "classifier.npz", **export_weights(fitted.model))
        challenger = Classifier.load(staging / "classifier.npz", expected_dim=dim)  # what ships
        challenger_misroutes = await critical_misroutes(
            critical_prompts, challenger.predict_proba(x_critical), fitted.threshold, registry
        )
        challenger_p_critical = challenger.predict_proba(x_critical)
        challenger_result = evaluate_classifier(
            challenger.predict_proba(x_eval), eval_rows, challenger_misroutes,
            int((challenger_p_critical < fitted.threshold).sum()),
        )  # fmt: skip
        labels_held_out = {row.q_cheap < 0.5 for row in eval_rows}
        if len(eval_rows) < MIN_HELD_OUT or len(labels_held_out) < 2:
            decision = False
            reason = (f"rejected: only {len(eval_rows)} held-out rows; promotion needs "
                      f"{MIN_HELD_OUT} with both labels")  # fmt: skip
        else:
            decision, reason = decide(champion_result, challenger_result)

        metrics = {
            "version": version, "kind": fitted.kind, "threshold": fitted.threshold,
            "val_auc": fitted.val_auc, "n_labels": n_labels, "n_train": len(train_y),
            "n_eval": len(eval_rows), "decision": decision, "reason": reason,
            "champion": asdict(champion_result), "challenger": asdict(challenger_result),
        }  # fmt: skip
        _write_json(staging / "metrics.json", metrics)
        _write_json(staging / "router.json", {
            **config, "threshold": fitted.threshold, "kind": fitted.kind, "version": version,
            "weights": f"versions/{version}/classifier.npz",
        })  # fmt: skip
        new = staging.rename(versions / version)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    promoted = decision and promote
    if promoted:
        _promote(router_dir, new, config)
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
