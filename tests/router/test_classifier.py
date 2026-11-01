import math
from pathlib import Path

import numpy as np
import pytest

from tierwise.router.classifier import ArtifactError, Classifier


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _clf(*layers: tuple[list[list[float]], list[float]]) -> Classifier:
    weights = [np.array(w, dtype=np.float32) for w, _ in layers]
    biases = [np.array(b, dtype=np.float32) for _, b in layers]
    return Classifier(weights=weights, biases=biases, input_dim=weights[0].shape[0])


def test_logistic_regression_case() -> None:
    clf = _clf(([[2.0], [0.0]], [-1.0]))
    assert clf.predict_proba(np.array([[1.0, 5.0]])) == pytest.approx([_sigmoid(1.0)])


def test_mlp_relu_between_layers() -> None:
    # hidden = relu([1,-1] @ [[1,2],[3,-1]] + [0,0]) = relu([-2, 3]) = [0, 3]
    # out = sigmoid(0*1 + 3*0.5 - 1) = sigmoid(0.5)
    clf = _clf(([[1.0, 2.0], [3.0, -1.0]], [0.0, 0.0]), ([[1.0], [0.5]], [-1.0]))
    assert clf.predict_proba(np.array([[1.0, -1.0]])) == pytest.approx([_sigmoid(0.5)])


def test_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "classifier.npz"
    rng = np.random.default_rng(0)
    np.savez(path, W0=rng.normal(size=(4, 3)), b0=np.zeros(3), W1=rng.normal(size=(3, 1)),
             b1=np.zeros(1))  # fmt: skip
    clf = Classifier.load(path, expected_dim=4)
    x = rng.normal(size=(2, 4))
    expected = _clf(
        (np.load(path)["W0"].tolist(), [0.0] * 3), (np.load(path)["W1"].tolist(), [0.0])
    ).predict_proba(x)
    np.testing.assert_allclose(clf.predict_proba(x), expected, rtol=1e-6)


def test_load_rejects_dim_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "classifier.npz"
    np.savez(path, W0=np.zeros((10, 1)), b0=np.zeros(1))
    with pytest.raises(ArtifactError, match="384"):
        Classifier.load(path, expected_dim=384)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        Classifier.load(tmp_path / "nope.npz", expected_dim=384)


def test_load_rejects_layers_that_do_not_chain(tmp_path: Path) -> None:
    path = tmp_path / "classifier.npz"
    np.savez(path, W0=np.zeros((4, 3)), b0=np.zeros(3), W1=np.zeros((2, 1)), b1=np.zeros(1))
    with pytest.raises(ArtifactError):
        Classifier.load(path, expected_dim=4)
