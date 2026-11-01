from typing import Any

import pytest

from tierwise.router.modes import (
    EmptyAfterTag,
    UnknownRoutingValue,
    extract_tag,
    parse_target,
)
from tierwise.schema import ChatRequest, TextPart


def _req(messages: list[dict[str, Any]], model: str = "auto") -> ChatRequest:
    return ChatRequest.model_validate({"model": model, "messages": messages})


def test_parse_auto_variants() -> None:
    assert parse_target("auto").mode == "balanced"
    assert parse_target("auto:balanced").mode == "balanced"
    assert parse_target("auto:cheap").mode == "cheap"
    assert parse_target("auto:quality").mode == "quality"
    assert parse_target("auto").kind == "auto"


def test_parse_tier() -> None:
    target = parse_target("tier:flagship")
    assert (target.kind, target.tier, target.mode) == ("tier", "flagship", None)


def test_parse_concrete_id() -> None:
    target = parse_target("claude-haiku-5-5")
    assert (target.kind, target.model_id) == ("model", "claude-haiku-5-5")


@pytest.mark.parametrize("value", ["auto:fast", "tier:mega"])
def test_parse_unknown_auto_or_tier_raises(value: str) -> None:
    with pytest.raises(UnknownRoutingValue):
        parse_target(value)


def test_model_defaults_to_auto() -> None:
    req = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    assert req.model == "auto"


def test_critical_tag_stripped_from_last_user_message() -> None:
    req = _req([{"role": "user", "content": "Fix this [CRITICAL] now"}])
    cleaned, tag = extract_tag(req)
    assert tag == "critical"
    assert cleaned.messages[0].content == "Fix this now"
    assert req.messages[0].content == "Fix this [CRITICAL] now"


def test_tag_in_earlier_message_ignored() -> None:
    req = _req(
        [
            {"role": "user", "content": "[critical] old"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "new question"},
        ]
    )
    cleaned, tag = extract_tag(req)
    assert tag is None
    assert cleaned.messages[0].content == "[critical] old"


def test_both_tags_critical_wins() -> None:
    _, tag = extract_tag(_req([{"role": "user", "content": "[easy] but [critical] thing"}]))
    assert tag == "critical"


def test_tag_in_multipart_text_part() -> None:
    content = [
        {"type": "text", "text": "[easy] describe"},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
    ]
    cleaned, tag = extract_tag(_req([{"role": "user", "content": content}]))
    assert tag == "easy"
    parts = cleaned.messages[0].content
    assert isinstance(parts, list)
    assert isinstance(parts[0], TextPart)
    assert parts[0].text == "describe"


def test_tag_only_message_raises_empty() -> None:
    with pytest.raises(EmptyAfterTag):
        extract_tag(_req([{"role": "user", "content": "  [critical]  "}]))
