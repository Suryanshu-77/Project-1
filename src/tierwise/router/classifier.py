"""The learned router's classifier head: a numpy forward pass over weights exported from sklearn.

Weights live in an .npz as W0, b0, W1, b1, ... with W_i shaped (in, out). One layer is logistic
regression; more layers are an MLP with ReLU between them. No pickles, no sklearn at runtime.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ArtifactError(Exception):
    """A router artifact is missing or doesn't fit the encoder."""


@dataclass(frozen=True)
class Classifier:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    input_dim: int

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """P(strong model needed) for each row of x, shape (batch,)."""
        h = np.asarray(x, dtype=np.float32)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = h @ w + b
            if i < last:
                h = np.maximum(h, 0.0)
        logits = h[:, 0].astype(np.float64)
        proba: np.ndarray = 1.0 / (1.0 + np.exp(-logits))
        return proba

    @classmethod
    def load(cls, path: Path, expected_dim: int) -> "Classifier":
        try:
            with np.load(path) as data:
                arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"cannot read classifier weights {path}: {exc}") from exc
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        while f"W{len(weights)}" in arrays:
            i = len(weights)
            if f"b{i}" not in arrays:
                raise ArtifactError(f"{path}: W{i} has no matching b{i}")
            weights.append(arrays[f"W{i}"])
            biases.append(arrays[f"b{i}"].reshape(-1))
        if not weights:
            raise ArtifactError(f"{path}: no W0 layer")
        if weights[0].shape[0] != expected_dim:
            raise ArtifactError(
                f"{path}: classifier expects {weights[0].shape[0]}-dim input, "
                f"encoder produces {expected_dim}"
            )
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            if w.ndim != 2 or b.shape != (w.shape[1],):
                raise ArtifactError(f"{path}: layer {i} has W {w.shape} and b {b.shape}")
            if i and weights[i - 1].shape[1] != w.shape[0]:
                raise ArtifactError(f"{path}: layer {i} input doesn't match layer {i - 1} output")
        if weights[-1].shape[1] != 1:
            raise ArtifactError(f"{path}: last layer must have one output unit")
        return cls(weights=weights, biases=biases, input_dim=expected_dim)
