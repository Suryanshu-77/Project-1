"""Train the learned router's classifier head and compare it with the rules router.

Usage: uv run --group train python -m training.train
Reads training/data/*.jsonl (training.prepare_data), embeds prompts through the gateway's own
truncation and ONNX encoder (no train/serve skew), fits logistic regression and an MLP, keeps the
better one on validation ROC-AUC, picks the threshold on validation, and writes
models/router/{classifier.npz, router.json, metrics.json}.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

from tierwise.router.encoder import OnnxEncoder, build_inputs, load_tokenizer
from tierwise.router.features import extract_features
from tierwise.router.rules import score
from tierwise.schema import ChatRequest
from training.export_onnx import ENCODER
from training.prepare_data import DATA_DIR

ROUTER_DIR = Path("models/router")
BATCH = 64
MIN_MLP_TRAIN = 50  # an MLP with early stopping needs a validation slice of its own


@dataclass
class Fitted:
    model: LogisticRegression | MLPClassifier
    kind: str
    threshold: float
    val_auc: dict[str, float] | None  # None when validation has a single class


def fit(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Fitted:
    """Fit logistic regression and (with enough data) an MLP; keep the better on validation
    ROC-AUC and pick the threshold there. With one validation class neither can be scored:
    keep logistic regression and a neutral 0.5 threshold."""
    candidates: dict[str, LogisticRegression | MLPClassifier] = {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000),
    }
    if len(x_train) >= MIN_MLP_TRAIN:
        candidates["mlp"] = MLPClassifier(
            hidden_layer_sizes=(128,), early_stopping=True, random_state=0
        )
    for model in candidates.values():
        model.fit(x_train, y_train)
    if len(np.unique(y_val)) < 2:
        return Fitted(candidates["logreg"], "logreg", 0.5, None)
    val_auc = {
        kind: float(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))
        for kind, model in candidates.items()
    }
    kind = max(val_auc, key=lambda k: val_auc[k])
    model = candidates[kind]
    return Fitted(model, kind, _threshold(y_val, model.predict_proba(x_val)[:, 1]), val_auc)


def export_weights(model: LogisticRegression | MLPClassifier) -> dict[str, np.ndarray]:
    """sklearn model -> the npz layout tierwise.router.classifier reads (W0, b0, W1, b1, ...)."""
    if isinstance(model, LogisticRegression):
        layers = [(model.coef_.T, model.intercept_)]
    else:
        layers = list(zip(model.coefs_, model.intercepts_, strict=True))
    weights: dict[str, np.ndarray] = {}
    for i, (w, b) in enumerate(layers):
        weights[f"W{i}"] = np.asarray(w, dtype=np.float32)
        weights[f"b{i}"] = np.asarray(b, dtype=np.float32).reshape(-1)
    return weights


def _request(prompt: str) -> ChatRequest:
    return ChatRequest.model_validate({"messages": [{"role": "user", "content": prompt}]})


def _load(split: str) -> tuple[list[str], np.ndarray]:
    rows = [json.loads(line) for line in (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8")]
    return [r["prompt"] for r in rows], np.array([r["label"] for r in rows])


def _embed(encoder: OnnxEncoder, tokenizer: Any, prompts: list[str]) -> np.ndarray:
    inputs = [build_inputs(tokenizer, _request(p)) for p in prompts]
    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i].input_ids))  # less padding
    out = np.zeros((len(inputs), encoder.dim), dtype=np.float32)
    for start in range(0, len(order), BATCH):
        idx = order[start : start + BATCH]
        out[idx] = encoder.embed([inputs[i] for i in idx])
        if start % (BATCH * 50) == 0:
            print(f"  embedded {start}/{len(order)}", flush=True)
    return out


def _threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Youden's J: the cut that best separates the classes on validation."""
    fpr, tpr, thresholds = roc_curve(y, p)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thresholds[best], 1e-3, 1 - 1e-3))


def main() -> None:
    tokenizer = load_tokenizer(ROUTER_DIR / "tokenizer.json")
    encoder = OnnxEncoder(ROUTER_DIR / "model.onnx", threads=4)
    data = {}
    for split in ("train", "val", "test"):
        prompts, y = _load(split)
        print(f"embedding {split}: {len(prompts)} prompts", flush=True)
        data[split] = (prompts, _embed(encoder, tokenizer, prompts), y)

    _, x_train, y_train = data["train"]
    _, x_val, y_val = data["val"]
    fitted = fit(x_train, y_train, x_val, y_val)
    model, kind, threshold = fitted.model, fitted.kind, fitted.threshold
    val_auc = fitted.val_auc or {}
    for name, auc in val_auc.items():
        print(f"{name}: val ROC-AUC {auc:.4f}", flush=True)

    test_prompts, x_test, y_test = data["test"]
    rules = [score(extract_features(_request(p))) for p in test_prompts]
    test_auc = {
        "learned": float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])),
        "rules": float(roc_auc_score(y_test, rules)),
    }

    np.savez(ROUTER_DIR / "classifier.npz", **export_weights(model))
    config = {"encoder": ENCODER, "threshold": threshold, "dim": encoder.dim, "kind": kind}
    (ROUTER_DIR / "router.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics = {
        "kind": kind,
        "threshold": threshold,
        "n": {s: len(d[2]) for s, d in data.items()},
        "positive_rate": {s: round(float(d[2].mean()), 4) for s, d in data.items()},
        "val_auc": {k: round(v, 4) for k, v in val_auc.items()},
        "test_auc": {k: round(v, 4) for k, v in test_auc.items()},
    }
    (ROUTER_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
