import os
from pathlib import Path

import numpy as np
import pytest

from tierwise.router.encoder import (
    EncodedInput,
    OnnxEncoder,
    build_inputs,
    load_tokenizer,
    pad_batch,
    pool,
)
from tierwise.schema import ChatRequest

ROUTER_DIR = Path(__file__).parents[2] / "models" / "router"
MODEL = ROUTER_DIR / "model.onnx"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run training.export_onnx first")


def _user(text: str) -> ChatRequest:
    return ChatRequest.model_validate({"messages": [{"role": "user", "content": text}]})


def test_pool_ignores_padding() -> None:
    tokens = np.array([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])
    np.testing.assert_allclose(pool(tokens, mask), [[1.0, 0.0]], atol=1e-6)


def test_pool_is_unit_norm() -> None:
    rng = np.random.default_rng(0)
    out = pool(rng.normal(size=(4, 7, 384)).astype(np.float32), np.ones((4, 7), dtype=np.int64))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert out.dtype == np.float32


def test_pad_batch_shapes() -> None:
    batch = pad_batch(
        [EncodedInput([1, 2], [1, 1], [0, 0]), EncodedInput([1, 2, 3, 4, 5], [1] * 5, [0] * 5)]
    )
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        assert batch[name].shape == (2, 5)
        assert batch[name].dtype == np.int64
    assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0, 0]


TEXTS = [
    "hi",
    "Write a Python function that merges two sorted lists.",
    "東京で一番おいしいラーメン屋はどこですか？",
    "Prove that there are infinitely many primes.",
    " ".join(["distributed consensus under network partitions"] * 60),
]


@needs_model
def test_onnx_matches_sentence_transformers() -> None:
    st = pytest.importorskip("sentence_transformers")
    reference = st.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    reference.max_seq_length = 256
    expected = reference.encode(TEXTS, normalize_embeddings=True)
    tokenizer = load_tokenizer(ROUTER_DIR / "tokenizer.json")
    got = OnnxEncoder(MODEL).embed([build_inputs(tokenizer, _user(t)) for t in TEXTS])
    cosines = (got * expected).sum(axis=1)
    assert cosines.min() >= 0.999, cosines


@needs_model
def test_batch_equals_single() -> None:
    tokenizer = load_tokenizer(ROUTER_DIR / "tokenizer.json")
    encoder = OnnxEncoder(MODEL)
    a, b = (build_inputs(tokenizer, _user(t)) for t in (TEXTS[0], TEXTS[4]))
    together = encoder.embed([a, b])
    np.testing.assert_allclose(together[0], encoder.embed([a])[0], atol=1e-5)
    np.testing.assert_allclose(together[1], encoder.embed([b])[0], atol=1e-5)
    assert encoder.dim == 384


@needs_model
def test_hugging_face_onnx_matches_ours() -> None:
    """`tierwise fetch-model` serves HF's own ONNX export: it must embed like ours."""
    hf = os.environ.get("TIERWISE_HF_ONNX")
    if not hf or not Path(hf).exists():
        pytest.skip("set TIERWISE_HF_ONNX to a file from `tierwise fetch-model`")
    tokenizer = load_tokenizer(ROUTER_DIR / "tokenizer.json")
    inputs = [build_inputs(tokenizer, _user(t)) for t in TEXTS]
    ours, theirs = OnnxEncoder(MODEL).embed(inputs), OnnxEncoder(Path(hf)).embed(inputs)
    cosines = (ours * theirs).sum(axis=1)
    assert cosines.min() > 0.999, cosines
