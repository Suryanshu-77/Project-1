"""Learned router: MiniLM embedding + classifier head -> P(strong model needed) -> a decision."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer

from tierwise.router import RoutingDecision
from tierwise.router.classifier import Classifier
from tierwise.router.encoder import EncodedInput, build_inputs
from tierwise.schema import ChatRequest


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

    def _score_sync(self, req: ChatRequest) -> float:
        """Blocking: tokenization and ONNX inference. Runs in the router's thread pool."""
        embedding = self._embedder.embed([build_inputs(self._tokenizer, req)])
        return float(self._classifier.predict_proba(embedding)[0])

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
