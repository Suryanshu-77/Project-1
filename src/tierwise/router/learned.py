"""Learned router: MiniLM embedding + classifier head -> P(strong model needed) -> a decision."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer

from tierwise.config import CACHE_ROUTER_DIR, Settings, router_directory
from tierwise.router import RoutingDecision
from tierwise.router.classifier import ArtifactError, Classifier
from tierwise.router.encoder import (
    MAX_CHARS_PER_TOKEN,
    MAX_TOKENS,
    EncodedInput,
    OnnxEncoder,
    build_inputs,
    load_tokenizer,
    special_id,
)
from tierwise.schema import ChatRequest

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, inputs: list[EncodedInput]) -> np.ndarray: ...


def to_decision(p: float, threshold: float) -> tuple[float, float]:
    """Turn P(strong model needed) and the trained threshold into (difficulty, confidence).

    The selector reads them as: quality bar = 0.30 + 0.65 * difficulty (+/- the mode shift), and
    confidence below 0.3 raises the bar one step. `threshold` is where the classifier best
    separates "the cheap answer was good enough" from "it wasn't" on the validation split; it
    is generally not 0.5.
    """
    # Clamp bounds to prevent division by zero at edge cases
    p = max(0.0, min(1.0, p))
    threshold = max(1e-5, min(1.0 - 1e-5, threshold))

    if p <= threshold:
        # Map [0, threshold] -> [0, 0.5]
        difficulty = 0.5 * (p / threshold)
        # 1 at p=0, 0 at p=threshold
        confidence = (threshold - p) / threshold
    else:
        # Map [threshold, 1] -> [0.5, 1]
        difficulty = 0.5 + 0.5 * ((p - threshold) / (1.0 - threshold))
        # 0 at p=threshold, 1 at p=1
        confidence = (p - threshold) / (1.0 - threshold)

    return float(difficulty), float(confidence)


class LearnedRouter:
    def __init__(
        self,
        tokenizer: Tokenizer,
        embedder: Embedder,
        classifier: Classifier,
        threshold: float,
        executor: ThreadPoolExecutor,
        version: str | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._embedder = embedder
        self._classifier = classifier
        self.threshold = threshold
        self._executor = executor
        self.version = version  # router.json "version"; None for the shipped classifier

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def score_batch(self, reqs: list[ChatRequest]) -> np.ndarray:
        """Blocking: P(strong model needed) for each request, in one padded ONNX batch."""
        embeddings = self._embedder.embed([build_inputs(self._tokenizer, r) for r in reqs])
        return self._classifier.predict_proba(embeddings)

    def _score_sync(self, req: ChatRequest) -> float:
        """Blocking: tokenization and ONNX inference. Runs in the router's thread pool."""
        return float(self.score_batch([req])[0])

    def _embed_sync(self, text: str) -> np.ndarray | None:
        if len(text) > (MAX_TOKENS - 2) * MAX_CHARS_PER_TOKEN:
            return None  # build_inputs would cut the text before tokenizing it
        req = ChatRequest.model_validate({"messages": [{"role": "user", "content": text}]})
        inputs = build_inputs(self._tokenizer, req)
        if len(inputs.input_ids) >= MAX_TOKENS:
            return None  # truncated: the vector wouldn't represent the whole text
        if special_id(self._tokenizer, "[UNK]") in inputs.input_ids:
            return None  # emoji, rare symbols, 100+ char words: different texts, same vector
        vector: np.ndarray = self._embedder.embed([inputs])[0]
        return vector

    async def embed_text(self, text: str) -> np.ndarray | None:
        """A unit vector for the semantic cache, or None when the text is too long to embed
        without truncation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._embed_sync, text)

    async def decide(self, req: ChatRequest) -> RoutingDecision:
        loop = asyncio.get_running_loop()
        p = await loop.run_in_executor(self._executor, self._score_sync, req)
        difficulty, confidence = to_decision(p, self.threshold)
        return RoutingDecision(
            difficulty=difficulty,
            confidence=confidence,
            source="learned",
            signals={"p_strong": p, "threshold": self.threshold},
        )


def model_path(directory: Path) -> Path:
    """The encoder: next to the router artifacts, else where `tierwise fetch-model` put it."""
    local = directory / "model.onnx"
    return local if local.exists() else CACHE_ROUTER_DIR / "model.onnx"


def _load(directory: Path, settings: Settings) -> LearnedRouter:
    config_path = directory / "router.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        threshold, dim = float(config["threshold"]), int(config["dim"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"cannot read {config_path}: {exc}") from exc
    if not 0.0 < threshold < 1.0:
        raise ArtifactError(f"{config_path}: threshold {threshold} is outside (0, 1)")
    encoder_path = model_path(directory)
    if not encoder_path.exists():
        raise ArtifactError(f"model.onnx is in neither {directory} nor {CACHE_ROUTER_DIR}; "
                            "run `tierwise fetch-model`")  # fmt: skip
    classifier = Classifier.load(directory / "classifier.npz", expected_dim=dim)
    try:
        tokenizer = load_tokenizer(directory / "tokenizer.json")
        encoder = OnnxEncoder(encoder_path, settings.router_threads, settings.router_provider)
    except Exception as exc:  # tokenizers and onnxruntime raise their own exception types
        raise ArtifactError(f"cannot load the encoder from {directory}: {exc}") from exc
    if encoder.dim != dim:
        raise ArtifactError(f"encoder produces {encoder.dim}-dim vectors, router.json says {dim}")
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tierwise-router")
    version = config.get("version")
    return LearnedRouter(tokenizer, encoder, classifier, threshold, executor,
                         str(version) if version is not None else None)  # fmt: skip


def load_learned(settings: Settings) -> LearnedRouter | None:
    """The learned router, or None (rules-only) when disabled or its artifacts are unusable."""
    if not settings.learned_router:
        return None
    try:
        return _load(router_directory(settings), settings)
    except ArtifactError as exc:
        logger.warning("learned router disabled, routing with rules only: %s", exc)
        return None
