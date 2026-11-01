import numpy as np
import pytest

pytest.importorskip("sklearn")

from training.train import fit  # noqa: E402


def _data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(int)
    x = rng.normal(size=(n, 8)) + y[:, None] * 2.0  # separable along every axis
    return x, y


def test_fit_picks_better_of_logreg_and_mlp() -> None:
    (x_train, y_train), (x_val, y_val) = _data(200, 0), _data(80, 1)
    fitted = fit(x_train, y_train, x_val, y_val)
    assert fitted.kind in {"logreg", "mlp"}
    assert fitted.val_auc is not None
    assert fitted.val_auc[fitted.kind] == max(fitted.val_auc.values())
    assert fitted.val_auc[fitted.kind] > 0.95
    assert 0.0 < fitted.threshold < 1.0
    assert fitted.model.predict_proba(x_val).shape == (80, 2)


def test_fit_one_class_val_falls_back() -> None:
    x_train, y_train = _data(100, 0)
    x_val = np.random.default_rng(2).normal(size=(10, 8))
    fitted = fit(x_train, y_train, x_val, np.zeros(10, dtype=int))
    assert fitted.threshold == 0.5 and fitted.val_auc is None
