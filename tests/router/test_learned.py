import asyncio
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from tierwise.router.classifier import Classifier
from tierwise.router.encoder import EncodedInput, load_tokenizer
from tierwise.router.learned import LearnedRouter, to_decision
from tierwise.schema import ChatRequest

TOKENIZER = Path(__file__).parents[2] / "models" / "router" / "tokenizer.json"


class FakeEmbedder:
    def __init__(self, value: float = 1.0, delay: float = 0.0) -> None:
        self.value, self.delay = value, delay

    def embed(self, inputs: list[EncodedInput]) -> np.ndarray:
        if self.delay:
            time.sleep(self.delay)
        return np.full((len(inputs), 2), self.value, dtype=np.float32)


# logit = x0 + x1 - 1 -> with x = [1, 1]: sigmoid(1) ≈ 0.731
CLASSIFIER = Classifier(
    weights=[np.array([[1.0], [1.0]], dtype=np.float32)],
    biases=[np.array([-1.0], dtype=np.float32)],
    input_dim=2,
)


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=2)
    yield pool
    pool.shutdown(wait=True)


def _router(executor: ThreadPoolExecutor, embedder: FakeEmbedder | None = None) -> LearnedRouter:
    return LearnedRouter(
        load_tokenizer(TOKENIZER), embedder or FakeEmbedder(), CLASSIFIER, 0.5, executor
    )


def _req(content: object = "explain the CAP theorem") -> ChatRequest:
    return ChatRequest.model_validate({"messages": [{"role": "user", "content": content}]})


async def test_decide_returns_learned_source_and_signals(executor: ThreadPoolExecutor) -> None:
    decision = await _router(executor).decide(_req())
    assert decision.source == "learned"
    assert decision.signals["p_strong"] == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert decision.signals["threshold"] == 0.5
    assert 0.0 <= decision.difficulty <= 1.0
    assert 0.0 <= decision.confidence <= 1.0


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_to_decision_bounds(p: float, threshold: float) -> None:
    for value in (p, threshold):
        difficulty, confidence = to_decision(value, threshold)
        assert 0.0 <= difficulty <= 1.0
        assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_to_decision_monotonic(threshold: float) -> None:
    difficulties = [to_decision(p / 100, threshold)[0] for p in range(101)]
    assert all(a <= b for a, b in zip(difficulties, difficulties[1:], strict=False))


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_to_decision_low_confidence_near_threshold(threshold: float) -> None:
    assert to_decision(threshold, threshold)[1] < 0.3


def test_to_decision_confident_far_from_threshold() -> None:
    assert to_decision(0.99, 0.5)[1] > 0.6
    assert to_decision(0.01, 0.5)[1] > 0.6


async def test_inference_does_not_block_event_loop(executor: ThreadPoolExecutor) -> None:
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    await _router(executor, FakeEmbedder(delay=0.2)).decide(_req())
    stop.set()
    await task
    assert ticks >= 10


async def test_twenty_concurrent_decisions_complete(executor: ThreadPoolExecutor) -> None:
    router = _router(executor, FakeEmbedder(delay=0.01))
    decisions = await asyncio.gather(*(router.decide(_req(f"q{i}")) for i in range(20)))
    assert len(decisions) == 20
    assert {d.source for d in decisions} == {"learned"}


async def test_image_only_request_still_decides(executor: ThreadPoolExecutor) -> None:
    image = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    decision = await _router(executor).decide(_req(image))
    assert decision.source == "learned"


def test_score_batch_matches_single(executor: ThreadPoolExecutor) -> None:
    router = _router(executor)
    a, b = _req("explain the CAP theorem"), _req("hi")
    np.testing.assert_allclose(
        router.score_batch([a, b]), [router._score_sync(a), router._score_sync(b)]
    )
