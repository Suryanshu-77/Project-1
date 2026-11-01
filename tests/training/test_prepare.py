import numpy as np
import pytest

from tierwise.router.classifier import Classifier
from training.prepare_data import label, prepare, split_of


def test_label_threshold() -> None:
    assert label(3) == 1
    assert label(4) == 0
    assert label(5) == 0


def test_split_is_deterministic_and_dedup_safe() -> None:
    assert split_of("Hello  World") == split_of("hello world")
    assert split_of("hello world") == split_of("hello world")


def test_split_proportions() -> None:
    splits = [split_of(f"prompt number {i}") for i in range(10_000)]
    assert 0.78 <= splits.count("train") / len(splits) <= 0.82
    assert 0.08 <= splits.count("val") / len(splits) <= 0.12


def test_prepare_dedups_and_drops_empty() -> None:
    rows = [
        {"prompt": "What is 2+2?", "mixtral_score": 5},
        {"prompt": "what is  2+2?", "mixtral_score": 1},
        {"prompt": "   ", "mixtral_score": 3},
        {"prompt": "no score here", "mixtral_score": None},
        {"prompt": "Prove Fermat's last theorem", "mixtral_score": 2},
    ]
    out = prepare(rows, limit=None)
    everything = [r["prompt"] for split in out.values() for r in split]
    assert sorted(everything) == ["Prove Fermat's last theorem", "What is 2+2?"]
    labels = {r["prompt"]: r["label"] for split in out.values() for r in split}
    assert labels == {"What is 2+2?": 0, "Prove Fermat's last theorem": 1}


def test_prepare_limit_caps_total_rows() -> None:
    rows = [{"prompt": f"p{i}", "mixtral_score": 5} for i in range(100)]
    assert sum(len(v) for v in prepare(rows, limit=30).values()) == 30


def test_exported_weights_match_sklearn() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    neural_network = pytest.importorskip("sklearn.neural_network")
    from training.train import export_weights

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8))
    y = (x[:, 0] + x[:, 1] ** 2 > 0.5).astype(int)
    models = [
        linear_model.LogisticRegression(max_iter=500).fit(x, y),
        neural_network.MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0).fit(
            x, y
        ),
    ]
    for model in models:
        weights = export_weights(model)
        n = len(weights) // 2
        clf = Classifier(
            weights=[weights[f"W{i}"] for i in range(n)],
            biases=[weights[f"b{i}"] for i in range(n)],
            input_dim=8,
        )
        np.testing.assert_allclose(clf.predict_proba(x), model.predict_proba(x)[:, 1], atol=1e-5)
