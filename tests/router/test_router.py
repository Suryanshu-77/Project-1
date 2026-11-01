from typing import Any

import pytest

from tierwise.router import router as router_module
from tierwise.router.router import required_caps, route
from tierwise.schema import ChatRequest


def _req(content: Any = "hi", model: str = "auto", **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    )


def test_auto_uses_rules() -> None:
    result = route(_req("hi"))
    assert result.target.kind == "auto"
    assert result.decision is not None
    assert result.decision.source == "rules"
    assert result.decision.difficulty < 0.4


def test_critical_tag_overrides_rules() -> None:
    result = route(_req("hi [critical]"))
    assert result.decision is not None
    assert (result.decision.difficulty, result.decision.confidence) == (1.0, 1.0)
    assert result.decision.source == "override"
    assert result.request.messages[0].content == "hi"


def test_easy_tag_overrides_rules() -> None:
    result = route(_req("[easy] design a distributed database for production"))
    assert result.decision is not None
    assert (result.decision.difficulty, result.decision.source) == (0.0, "override")


def test_scorer_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(req: ChatRequest) -> None:
        raise RuntimeError("bad rule")

    monkeypatch.setattr(router_module, "rules_decision", broken)
    result = route(_req("hi"))
    assert result.decision is not None
    assert (result.decision.difficulty, result.decision.confidence) == (0.7, 0.0)
    assert result.decision.source == "fallback"


@pytest.mark.parametrize("model", ["tier:basic", "claude-haiku-5-5"])
def test_tier_and_model_have_no_decision(model: str) -> None:
    assert route(_req("hi", model=model)).decision is None


def test_tag_stripped_for_concrete_model_too() -> None:
    result = route(_req("[critical] hi", model="claude-haiku-5-5"))
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
