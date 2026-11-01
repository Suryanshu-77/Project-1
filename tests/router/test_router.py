import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tierwise.config import Settings
from tierwise.router import RoutingDecision
from tierwise.router import router as router_module
from tierwise.router.learned import load_learned
from tierwise.router.router import required_caps, route
from tierwise.schema import ChatRequest


def _req(content: Any = "hi", model: str = "auto", **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    )


async def test_auto_uses_rules() -> None:
    result = await route(_req("hi, thanks"))
    assert result.target.kind == "auto"
    assert result.decision is not None
    assert result.decision.source == "rules"
    assert result.decision.difficulty < 0.4


async def test_critical_tag_overrides_rules() -> None:
    result = await route(_req("hi [critical]"))
    assert result.decision is not None
    assert (result.decision.difficulty, result.decision.confidence) == (1.0, 1.0)
    assert result.decision.source == "override"
    assert result.request.messages[0].content == "hi"


async def test_easy_tag_overrides_rules() -> None:
    result = await route(_req("[easy] design a distributed database for production"))
    assert result.decision is not None
    assert (result.decision.difficulty, result.decision.source) == (0.0, "override")


async def test_scorer_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(req: ChatRequest) -> None:
        raise RuntimeError("bad rule")

    monkeypatch.setattr(router_module, "rules_decision", broken)
    result = await route(_req("hi"))
    assert result.decision is not None
    assert (result.decision.difficulty, result.decision.confidence) == (0.7, 0.0)
    assert result.decision.source == "fallback"


@pytest.mark.parametrize("model", ["tier:basic", "claude-haiku-5-5"])
async def test_tier_and_model_have_no_decision(model: str) -> None:
    assert (await route(_req("hi", model=model))).decision is None


async def test_tag_stripped_for_concrete_model_too() -> None:
    result = await route(_req("[critical] hi", model="claude-haiku-5-5"))
    assert result.request.messages[0].content == "hi"
    assert result.decision is None


def test_required_caps() -> None:
    image = [
        {"type": "text", "text": "what"},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
    ]
    tools = [{"type": "function", "function": {"name": "f"}}]
    assert required_caps(_req("hi")) == frozenset()
    assert required_caps(_req(image)) == {"vision"}
    assert required_caps(_req("hi", tools=tools)) == {"tools"}
    assert required_caps(_req("hi", response_format={"type": "json_object"})) == {"json"}


class FakeLearned:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    async def decide(self, req: ChatRequest) -> RoutingDecision:
        self.calls += 1
        if self.error:
            raise self.error
        return RoutingDecision(difficulty=0.8, confidence=0.9, source="learned")


MID = "explain the CAP theorem"  # rules confidence well under RULES_SHORTCUT


async def test_confident_rules_skip_learned() -> None:
    learned = FakeLearned()
    result = await route(_req("hi, thanks"), learned)  # type: ignore[arg-type]
    assert result.decision is not None and result.decision.source == "rules"
    assert learned.calls == 0


async def test_unsure_rules_use_learned() -> None:
    learned = FakeLearned()
    result = await route(_req(MID), learned)  # type: ignore[arg-type]
    assert result.decision is not None and result.decision.source == "learned"
    assert learned.calls == 1


async def test_no_learned_router_keeps_rules_decision() -> None:
    result = await route(_req(MID), None)
    assert result.decision is not None and result.decision.source == "rules"


async def test_learned_error_falls_back() -> None:
    result = await route(_req(MID), FakeLearned(RuntimeError("onnx")))  # type: ignore[arg-type]
    assert result.decision == router_module.FALLBACK


async def test_override_tag_skips_learned() -> None:
    learned = FakeLearned()
    result = await route(_req(f"{MID} [critical]"), learned)  # type: ignore[arg-type]
    assert result.decision is not None and result.decision.source == "override"
    assert learned.calls == 0


def _settings(router_dir: Path, **kw: Any) -> Settings:
    return Settings(_env_file=None, router_dir=str(router_dir), **kw)  # type: ignore[call-arg]


def test_load_learned_missing_dir_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert load_learned(_settings(tmp_path)) is None
    assert "router.json" in caplog.text or "model.onnx" in caplog.text


def test_load_learned_disabled_returns_none() -> None:
    assert load_learned(_settings(Path("models/router"), learned_router=False)) is None


def test_load_learned_bad_dim_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    real = Path(__file__).parents[2] / "models" / "router"
    if not (real / "model.onnx").exists():
        pytest.skip("run training.export_onnx first")
    for name in ("model.onnx", "tokenizer.json"):
        (tmp_path / name).write_bytes((real / name).read_bytes())
    (tmp_path / "router.json").write_text(
        json.dumps({"encoder": "x", "threshold": 0.5, "dim": 384, "kind": "logreg"})
    )
    np.savez(tmp_path / "classifier.npz", W0=np.zeros((10, 1)), b0=np.zeros(1))
    assert load_learned(_settings(tmp_path)) is None
    assert "384" in caplog.text


def test_load_learned_corrupt_config_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "router.json").write_text("{not json")
    assert load_learned(_settings(tmp_path)) is None
    assert "router.json" in caplog.text


def test_load_learned_truncated_classifier_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    real = Path(__file__).parents[2] / "models" / "router"
    (tmp_path / "router.json").write_text(
        json.dumps({"encoder": "x", "threshold": 0.5, "dim": 384, "kind": "mlp"})
    )
    (tmp_path / "model.onnx").write_bytes(b"")  # never reached: the classifier fails first
    (tmp_path / "classifier.npz").write_bytes((real / "classifier.npz").read_bytes()[:5000])
    assert load_learned(_settings(tmp_path)) is None
    assert "classifier.npz" in caplog.text
