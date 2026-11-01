"""Learned router: MiniLM embedding + classifier head -> P(strong model needed) -> a decision."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer

from tierwise.config import Settings
from tierwise.router import RoutingDecision
from tierwise.router.classifier import ArtifactError, Classifier
from tierwise.router.encoder import (
    MAX_TOKENS,
    EncodedInput,
    OnnxEncoder,
    build_inputs,
    load_tokenizer,
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
    ) -> None:
        self._tokenizer = tokenizer
        self._embedder = embedder
        self._classifier = classifier
        self.threshold = threshold
        self._executor = executor

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
        req = ChatRequest.model_validate({"messages": [{"role": "user", "content": text}]})
        inputs = build_inputs(self._tokenizer, req)
        if len(inputs.input_ids) >= MAX_TOKENS:
            return None  # truncated: the vector wouldn't represent the whole text
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


def _load(directory: Path, settings: Settings) -> LearnedRouter:
    config_path = directory / "router.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        threshold, dim = float(config["threshold"]), int(config["dim"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"cannot read {config_path}: {exc}") from exc
    if not 0.0 < threshold < 1.0:
        raise ArtifactError(f"{config_path}: threshold {threshold} is outside (0, 1)")
    model_path = directory / "model.onnx"
    if not model_path.exists():
        raise ArtifactError(f"{model_path} is missing; run training.export_onnx")
    classifier = Classifier.load(directory / "classifier.npz", expected_dim=dim)
    try:
        tokenizer = load_tokenizer(directory / "tokenizer.json")
        encoder = OnnxEncoder(model_path, settings.router_threads, settings.router_provider)
    except Exception as exc:  # tokenizers and onnxruntime raise their own exception types
        raise ArtifactError(f"cannot load the encoder from {directory}: {exc}") from exc
    if encoder.dim != dim:
        raise ArtifactError(f"encoder produces {encoder.dim}-dim vectors, router.json says {dim}")
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tierwise-router")
    return LearnedRouter(tokenizer, encoder, classifier, threshold, executor)


def load_learned(settings: Settings) -> LearnedRouter | None:
    """The learned router, or None (rules-only) when disabled or its artifacts are unusable."""
    if not settings.learned_router:
        return None
    try:
        return _load(Path(settings.router_dir), settings)
    except ArtifactError as exc:
        logger.warning("learned router disabled, routing with rules only: %s", exc)
        return None
