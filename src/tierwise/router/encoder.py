"""Learned-router encoder: token-space truncation, then MiniLM on ONNX Runtime, pooled in numpy."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from tierwise.schema import ChatRequest
from tierwise.tokens import text_of

MAX_TOKENS = 256  # including [CLS] and [SEP]


@dataclass(frozen=True)
class EncodedInput:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    # tokenizer.json ships with truncation and fixed padding at 128; we budget tokens ourselves
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def special_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer has no {token} token")
    return token_id


def segments(req: ChatRequest) -> list[str]:
    """What the classifier reads, most important first: last user turn, previous one, system."""
    users = [text_of(m.content) for m in req.messages if m.role == "user"]
    system = "\n".join(text_of(m.content) for m in req.messages if m.role == "system")
    ordered = [*reversed(users[-2:]), system]
    return [s for s in ordered if s.strip()]


def build_inputs(
    tokenizer: Tokenizer, req: ChatRequest, max_tokens: int = MAX_TOKENS
) -> EncodedInput:
    cls_id, sep_id = special_id(tokenizer, "[CLS]"), special_id(tokenizer, "[SEP]")
    budget = max_tokens - 2  # [CLS] ... [SEP]
    body: list[int] = []
    for text in segments(req):
        if body:
            if budget < 2:  # a separator plus at least one token
                break
            body.append(sep_id)
            budget -= 1
        if budget <= 0:
            break
        # cut from the tail: the start of a turn carries the instruction
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:budget]
        body.extend(ids)
        budget -= len(ids)
    input_ids = [cls_id, *body, sep_id]
    return EncodedInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        token_type_ids=[0] * len(input_ids),
    )


def pad_batch(inputs: list[EncodedInput]) -> dict[str, np.ndarray]:
    """Right-pad to the longest sequence; a single request runs unpadded."""
    length = max(len(x.input_ids) for x in inputs)
    batch = {
        name: np.zeros((len(inputs), length), dtype=np.int64)
        for name in ("input_ids", "attention_mask", "token_type_ids")
    }
    for row, x in enumerate(inputs):
        n = len(x.input_ids)
        batch["input_ids"][row, :n] = x.input_ids
        batch["attention_mask"][row, :n] = x.attention_mask
        batch["token_type_ids"][row, :n] = x.token_type_ids
    return batch


def pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean over real tokens, then L2-normalize: the same as sentence-transformers' MiniLM."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    mean = summed / np.clip(mask.sum(axis=1), 1e-9, None)
    norms = np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)
    pooled: np.ndarray = (mean / norms).astype(np.float32)
    return pooled


class OnnxEncoder:
    """One InferenceSession for the app's life. `embed` blocks: call it off the event loop."""

    def __init__(
        self, model_path: Path, threads: int = 2, provider: str = "CPUExecutionProvider"
    ) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(model_path), options, providers=[provider])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self.dim = int(self.embed([EncodedInput([101, 102], [1, 1], [0, 0])]).shape[1])  # warm-up

    def embed(self, inputs: list[EncodedInput]) -> np.ndarray:
        feed = {k: v for k, v in pad_batch(inputs).items() if k in self._input_names}
        (tokens,) = self._session.run(["last_hidden_state"], feed)
        return pool(np.asarray(tokens), feed["attention_mask"])
