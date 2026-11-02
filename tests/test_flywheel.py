import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from tierwise import flywheel
from tierwise.__main__ import main
from tierwise.flywheel import (
    JUDGE_MARKER,
    contrast_for,
    label_pending,
    needs_strong,
    parse_scores,
)  # fmt: skip
from tierwise.providers.base import Provider
from tierwise.providers.openai import OpenAIProvider
from tierwise.registry import ModelSpec, Registry
from tierwise.reliability.retry import RetryPolicy
from tierwise.schema import ChatRequest
from tierwise.store import RequestLog, ShadowSample

NOV = 1793577600.0  # 2026-11-02T00:00:00Z


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


CHEAP_SHORT = {"small": 2, "medium": 5, "large": 5}  # the small model's answer falls short


def _verdict(body: dict[str, Any], scores: dict[str, int]) -> str:
    """A judge that scores each answer by the model that wrote it, wherever it is shown."""
    text = body["messages"][-1]["content"]
    a, b = re.search(r"Answer A:\n(.*?)\n\nAnswer B:\n(.*?)\n\n", text, re.DOTALL).groups()  # type: ignore[union-attr]
    by_model = {f"answer from {m}": s for m, s in scores.items()}
    return json.dumps({"a": by_model[a], "b": by_model[b]})


def _upstream(judge_reply: str | None = None, calls: list[dict[str, Any]] | None = None,  # type: ignore[no-untyped-def]
              scores: dict[str, int] = CHEAP_SHORT):  # fmt: skip
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        first = body["messages"][0]
        judging = first["role"] == "system" and JUDGE_MARKER in first["content"]
        if not judging:
            content = f"answer from {body['model']}"
        else:
            content = judge_reply if judge_reply is not None else _verdict(body, scores)
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
    assert (labeled.judge_score, labeled.judge_score_strong) == (2, 5)  # cheap, strong
    assert labeled.needs_strong is True
    # contrast on large (2000 tokens at $20/M) plus the judge on large: $0.04 + $0.04
    assert labeled.label_cost_usd == pytest.approx(0.08) and run.spent_usd == pytest.approx(0.08)
    assert labeled.labeled_ts == NOV


async def test_flagship_routed_contrasts_cheapest_basic(log: RequestLog) -> None:
    await _seed(log, _sample(0, routed="large"))
    upstream = _upstream(scores={"small": 5, "large": 5})
    await label_pending(log, REGISTRY, _providers(upstream), _judge(), 100.0, 10)
    [labeled] = await log.labeled_samples()
    assert labeled.contrast_model == "small" and labeled.needs_strong is False


@pytest.mark.parametrize("routed", ["small", "large"])
async def test_both_answers_wrong_does_not_need_the_strong_model(
    log: RequestLog, routed: str
) -> None:
    """The flagship can't fix a prompt it fails too: paying for it would buy nothing. A judge
    scoring only the cheaper answer labeled these 'needs strong' (stage 1 of the flywheel
    experiment: 5 of 20 identical pairs where gpt-4o itself was wrong)."""
    await _seed(log, _sample(0, routed=routed))
    upstream = _upstream(scores={"small": 1, "large": 1})
    await label_pending(log, REGISTRY, _providers(upstream), _judge(), 100.0, 10)
    [labeled] = await log.labeled_samples()
    assert (labeled.judge_score, labeled.judge_score_strong) == (1, 1)
    assert labeled.needs_strong is False


def test_needs_strong_never_fires_on_equal_scores() -> None:
    assert not any(needs_strong(s, s) for s in range(1, 6))  # identical answers, any quality


def test_needs_strong_when_only_the_cheap_answer_falls_short() -> None:
    assert needs_strong(2, 5) and needs_strong(1, 5)
    assert not needs_strong(5, 3) and not needs_strong(4, 2)  # the cheaper answer was better


def test_needs_strong_is_monotonic() -> None:
    """A worse cheap answer, or a better strong one, can only make the label more 'strong'."""
    for cheap in range(1, 6):
        for strong in range(1, 6):
            if needs_strong(cheap, strong):
                assert cheap == 1 or needs_strong(cheap - 1, strong)
                assert strong == 5 or needs_strong(cheap, strong + 1)


TEXT = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
PICTURE = ChatRequest.model_validate({"messages": [{"role": "user", "content": [
    {"type": "text", "text": "what is this?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})  # fmt: skip


def _vision_registry(basic_vision: bool) -> Registry:
    def spec(model_id: str, tier: str, price: float, caps: list[str]) -> ModelSpec:
        return ModelSpec.model_validate({"id": model_id, "provider": "openai", "price_in": price,
                                         "price_out": price, "context": 100_000, "caps": caps,
                                         "quality": 0.5, "tier": tier})  # fmt: skip

    models = [spec("small", "basic", 1.0, ["tools"]), spec("large", "flagship", 20.0,
                                                            ["tools", "vision"])]  # fmt: skip
    if basic_vision:
        models.append(spec("eyes", "basic", 2.0, ["tools", "vision"]))
    return Registry(baseline_model="large", models=models)


def test_contrast_for_skips_unconfigured_providers() -> None:
    small, large = REGISTRY.get("small"), REGISTRY.get("large")
    assert small is not None and large is not None
    assert contrast_for(small, TEXT, REGISTRY, {"openai"}) == large
    assert contrast_for(small, TEXT, REGISTRY, {"anthropic"}) is None


def test_contrast_is_baseline_when_another_model_answered() -> None:
    medium, large = REGISTRY.get("medium"), REGISTRY.get("large")
    assert medium is not None
    assert contrast_for(medium, TEXT, REGISTRY, {"openai"}) == large
    assert contrast_for(large, TEXT, REGISTRY, {"openai"}) == REGISTRY.get("small")


def test_vision_request_skips_basic_models_without_vision() -> None:
    registry = _vision_registry(basic_vision=True)
    large = registry.get("large")
    assert large is not None
    eyes = contrast_for(large, PICTURE, registry, {"openai"})
    assert eyes is not None and eyes.id == "eyes"  # small is cheaper but can't see
    assert contrast_for(large, PICTURE, _vision_registry(False), {"openai"}) is None


async def test_no_fitting_contrast_fails_before_any_call(log: RequestLog) -> None:
    calls: list[dict[str, Any]] = []
    registry = _vision_registry(basic_vision=False)
    sample = replace(_sample(0, routed="large"), request_json=PICTURE.model_dump_json())
    await _seed(log, sample)
    run = await label_pending(log, registry, _providers(_upstream(calls=calls)),
                              registry.get("large") or _judge(), 100.0, 10)  # fmt: skip
    assert (run.labeled, run.failed, run.spent_usd) == (0, 1, 0.0)
    assert calls == []  # nothing was paid for


@pytest.mark.parametrize(
    "bad", ["not json", '{"a": 9, "b": 5}', '{"a": "high", "b": 5}', "{}", '{"a": 5}',
            '{"score": 5}', '{"a": Infinity, "b": 5}', '{"a": 5, "b": NaN}', '{"a": 1e400, "b": 5}']
)
def test_parse_scores_rejects_bad_replies(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_scores(bad)


def test_parse_scores_accepts_fenced_json() -> None:
    assert parse_scores('```json\n{"a": 4, "b": 5}\n```') == (4, 5)


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
        messages, cheap_key = flywheel.judge_messages("q", "CHEAP", "STRONG", seed=seed)
        text = messages[-1].content
        assert isinstance(text, str)
        cheap_first = text.index("CHEAP") < text.index("STRONG")
        assert cheap_key == ("a" if cheap_first else "b")  # the caller maps scores back right
        return cheap_first

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
    monkeypatch.chdir(tmp_path)  # no developer .env: it could name another judge model
    monkeypatch.setenv("TIERWISE_REGISTRY", str(Path(__file__).parents[1] / "config/registry.yaml"))
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


def test_parse_scores_rejects_bool_accepts_integral_float() -> None:
    with pytest.raises(ValueError):
        parse_scores('{"a": true, "b": 5}')  # bool is an int subclass in Python
    assert parse_scores('{"a": 4.0, "b": 5}') == (4, 5)
    with pytest.raises(ValueError):
        parse_scores('{"a": 3.5, "b": 5}')


@pytest.mark.parametrize("reply", ['{"a": "5", "b": 5}', '{"a": " 5 ", "b": 5}',
                                   '{"a": "5.0", "b": 5}'])  # fmt: skip
def test_parse_scores_accepts_a_quoted_number(reply: str) -> None:
    assert parse_scores(reply) == (5, 5)  # mistral:7b quoted 216 of 744 scores in a real run


@pytest.mark.parametrize("bad", ['"3.5"', '"Infinity"', '"nan"', '"9"', '""', '"true"'])
def test_parse_scores_holds_quoted_numbers_to_the_same_rules(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_scores(f'{{"a": 5, "b": {bad}}}')


def test_non_baseline_flagship_gets_a_basic_contrast() -> None:
    """A cheaper flagship is still the strong side: comparing it with the baseline would label
    hard prompts 'easy', teaching the router the dangerous direction."""
    other = ModelSpec.model_validate({
        "id": "other-flagship", "provider": "openai", "price_in": 10.0, "price_out": 10.0,
        "context": 100_000, "caps": ["tools", "json"], "quality": 0.9, "tier": "flagship",
    })  # fmt: skip
    registry = Registry(baseline_model="large", models=[*REGISTRY.models, other])
    assert contrast_for(other, TEXT, registry, {"openai"}) == registry.get("small")


def test_parse_scores_error_shows_what_the_judge_wrote() -> None:
    with pytest.raises(ValueError, match="'5_0'"):
        parse_scores('{"a": "5_0", "b": 5}')  # float() reads it as 50.0
