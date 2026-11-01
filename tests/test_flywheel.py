import json
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from tierwise import flywheel
from tierwise.__main__ import main
from tierwise.flywheel import JUDGE_MARKER, contrast_for, label_pending, parse_score
from tierwise.providers.base import Provider
from tierwise.providers.openai import OpenAIProvider
from tierwise.registry import ModelSpec, Registry
from tierwise.reliability.retry import RetryPolicy
from tierwise.store import RequestLog, ShadowSample

NOV = 1793577600.0  # 2026-11-03 UTC


def _spec(model_id: str, tier: str, price: float) -> ModelSpec:
    return ModelSpec.model_validate(
        {"id": model_id, "provider": "openai", "price_in": price, "price_out": price,
         "context": 100_000, "caps": ["tools", "json"], "quality": 0.5, "tier": tier}  # fmt: skip
    )


REGISTRY = Registry(
    baseline_model="large",
    models=[_spec("small", "basic", 1.0), _spec("medium", "moderate", 5.0),
            _spec("large", "flagship", 20.0)],  # fmt: skip
)


def _upstream(judge_reply: str = '{"score": 2}', calls: list[dict[str, Any]] | None = None):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        first = body["messages"][0]
        judging = first["role"] == "system" and JUDGE_MARKER in first["content"]
        content = judge_reply if judging else f"answer from {body['model']}"
        return httpx.Response(200, json={
            "id": "x", "model": body["model"], "created": 1,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
        })  # fmt: skip

    return handler


def _providers(handler: Callable[[httpx.Request], httpx.Response]) -> dict[str, Provider]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return {"openai": OpenAIProvider(client, "sk-test")}


def _sample(i: int, routed: str = "small", prompt: str = "capital of france?") -> ShadowSample:
    request = {"model": "auto", "messages": [{"role": "user", "content": prompt}]}
    return ShadowSample(id=f"s{i}", ts=1000.0 + i, key_id="k", request_json=json.dumps(request),
                        routed_model=routed, routed_answer=f"answer from {routed}",
                        difficulty=0.3, confidence=0.4, source="rules")  # fmt: skip


@pytest.fixture
async def log(tmp_path: Path) -> AsyncIterator[RequestLog]:
    request_log = RequestLog(f"sqlite+aiosqlite:///{(tmp_path / 'f.db').as_posix()}")
    await request_log.start()
    yield request_log
    await request_log.stop()


async def _seed(log: RequestLog, *samples: ShadowSample) -> None:
    for sample in samples:
        log.submit_sample(sample)
    await log.stop()
    await log.start()


def _judge() -> ModelSpec:
    spec = REGISTRY.get("large")
    assert spec is not None
    return spec


async def test_cheap_routed_gets_flagship_contrast_and_label(log: RequestLog) -> None:
    await _seed(log, _sample(0, routed="small"))
    run = await label_pending(log, REGISTRY, _providers(_upstream()), _judge(), 100.0, 10,
                              clock=lambda: NOV)  # fmt: skip
    [labeled] = await log.labeled_samples()
    assert (run.labeled, run.failed, run.left) == (1, 0, 0)
    assert labeled.status == "labeled" and labeled.contrast_model == "large"
    assert labeled.contrast_answer == "answer from large"
    assert labeled.judge_score == 2 and labeled.needs_strong is True
    # contrast on large (2000 tokens at $20/M) plus the judge on large: $0.04 + $0.04
    assert labeled.label_cost_usd == pytest.approx(0.08) and run.spent_usd == pytest.approx(0.08)
    assert labeled.labeled_ts == NOV


async def test_flagship_routed_contrasts_cheapest_basic(log: RequestLog) -> None:
    await _seed(log, _sample(0, routed="large"))
    await label_pending(log, REGISTRY, _providers(_upstream('{"score": 5}')), _judge(), 100.0, 10)
    [labeled] = await log.labeled_samples()
    assert labeled.contrast_model == "small" and labeled.needs_strong is False


def test_contrast_for_skips_unconfigured_providers() -> None:
    small, large = REGISTRY.get("small"), REGISTRY.get("large")
    assert small is not None and large is not None
    assert contrast_for(small, REGISTRY, {"openai"}) == large
    assert contrast_for(small, REGISTRY, {"anthropic"}) is None


@pytest.mark.parametrize("bad", ["not json", '{"score": 9}', '{"score": "high"}', "{}"])
def test_parse_score_rejects_bad_replies(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_score(bad)


def test_parse_score_accepts_fenced_json() -> None:
    assert parse_score('```json\n{"score": 4}\n```') == 4


async def test_bad_judge_json_marks_failed_and_continues(log: RequestLog) -> None:
    await _seed(log, _sample(0), _sample(1))
    run = await label_pending(log, REGISTRY, _providers(_upstream("no idea")), _judge(), 100.0, 10)
    assert (run.labeled, run.failed) == (0, 2)
    assert await log.labeled_samples() == [] and await log.pending_samples(10) == []
    assert run.spent_usd > 0  # both calls ran and are paid for, though the label failed


async def test_budget_stops_before_next_sample(log: RequestLog) -> None:
    await _seed(log, _sample(0), _sample(1), _sample(2))
    run = await label_pending(log, REGISTRY, _providers(_upstream()), _judge(), 0.10, 10,
                              clock=lambda: NOV)  # fmt: skip
    assert run.stopped_by_budget and (run.labeled, run.left) == (2, 1)  # $0.08 each
    assert [s.id for s in await log.pending_samples(10)] == ["s2"]


async def test_budget_counts_earlier_spend_this_month(log: RequestLog) -> None:
    await _seed(log, _sample(0), _sample(1))
    first = await label_pending(log, REGISTRY, _providers(_upstream()), _judge(), 0.10, 1,
                                clock=lambda: NOV)  # fmt: skip
    second = await label_pending(log, REGISTRY, _providers(_upstream()), _judge(), 0.10, 10,
                                 clock=lambda: NOV)  # fmt: skip
    assert first.labeled == 1 and second.labeled == 1 and second.stopped_by_budget is False
    assert await log.label_spend("2026-11") == pytest.approx(0.16)


def test_judge_order_is_seeded() -> None:
    def order(seed: str) -> bool:  # True when the cheaper answer is shown first
        text = flywheel.judge_messages("q", "CHEAP", "STRONG", seed=seed)[-1].content
        assert isinstance(text, str)
        return text.index("CHEAP") < text.index("STRONG")

    assert order("s7") == order("s7")  # a re-run judges in the same order
    assert {order(f"s{i}") for i in range(20)} == {True, False}  # both positions get used


async def test_judge_sees_both_answers_and_the_marker(log: RequestLog) -> None:
    calls: list[dict[str, Any]] = []
    await _seed(log, _sample(0, prompt="why is the sky blue?"))
    await label_pending(log, REGISTRY, _providers(_upstream(calls=calls)), _judge(), 100.0, 10)
    judge = calls[-1]["messages"]
    assert JUDGE_MARKER in judge[0]["content"]
    assert "why is the sky blue?" in judge[-1]["content"]
    assert "answer from small" in judge[-1]["content"]
    assert "answer from large" in judge[-1]["content"]


def test_cli_flywheel_label_prints_summary_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake(*args: Any, **kwargs: Any) -> flywheel.LabelRun:
        return flywheel.LabelRun(labeled=3, failed=1, left=2, spent_usd=0.25,
                                 stopped_by_budget=True)  # fmt: skip

    monkeypatch.setattr(flywheel, "label_pending", fake)
    monkeypatch.setenv("TIERWISE_DB_URL", f"sqlite+aiosqlite:///{(tmp_path / 'c.db').as_posix()}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert main(["flywheel", "label", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"labeled 3, failed 1, left 2", out) and "$0.25" in out
    assert "budget" in out


@pytest.mark.parametrize(("content", "finish"), [("", "stop"), ("half an answ", "length")])
async def test_unusable_contrast_answer_fails_instead_of_labeling(
    log: RequestLog, content: str, finish: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if JUDGE_MARKER in str(body["messages"][0]["content"]):
            return _upstream()(request)
        return httpx.Response(200, json={
            "id": "x", "model": body["model"], "created": 1,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        })  # fmt: skip

    await _seed(log, _sample(0))
    run = await label_pending(log, REGISTRY, _providers(handler), _judge(), 100.0, 10)
    assert (run.labeled, run.failed) == (0, 1)  # a wrong label would poison retraining
    assert await log.labeled_samples() == []


async def test_transient_judge_error_is_retried(log: RequestLog) -> None:
    judged: list[int] = []
    ok = _upstream()

    def handler(request: httpx.Request) -> httpx.Response:
        if JUDGE_MARKER in str(json.loads(request.content)["messages"][0]["content"]):
            judged.append(1)
            if len(judged) == 1:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
        return ok(request)

    await _seed(log, _sample(0))
    run = await label_pending(log, REGISTRY, _providers(handler), _judge(), 100.0, 10,
                              policy=RetryPolicy(retries=2, base_s=0.0, max_s=0.0))  # fmt: skip
    assert (run.labeled, run.failed, len(judged)) == (1, 0, 2)
