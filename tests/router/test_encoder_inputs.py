from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer

from tierwise.router.encoder import (
    MAX_TOKENS,
    SYSTEM_BUDGET,
    build_inputs,
    load_tokenizer,
    segments,
)
from tierwise.schema import ChatRequest

TOKENIZER = Path(__file__).parents[2] / "models" / "router" / "tokenizer.json"


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return load_tokenizer(TOKENIZER)


def _req(*messages: tuple[str, Any]) -> ChatRequest:
    return ChatRequest.model_validate(
        {"messages": [{"role": role, "content": content} for role, content in messages]}
    )


def _middle(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids[1:-1])


def test_segments_priority_order() -> None:
    req = _req(("system", "S"), ("user", "U1"), ("assistant", "A"), ("user", "U2"))
    assert segments(req) == ["U2", "U1", "S"]


def test_short_request_fits_whole(tok: Tokenizer) -> None:
    enc = build_inputs(tok, _req(("user", "hello world")))
    assert enc.input_ids[0] == tok.token_to_id("[CLS]")
    assert enc.input_ids[-1] == tok.token_to_id("[SEP]")
    assert _middle(tok, enc.input_ids) == "hello world"


@pytest.mark.parametrize("text", ["漢字" * 25000, "🙂" * 20000, "word " * 10000])
def test_never_exceeds_budget_with_cjk_and_emoji(tok: Tokenizer, text: str) -> None:
    enc = build_inputs(tok, _req(("system", text), ("user", text), ("user", text)))
    assert len(enc.input_ids) <= MAX_TOKENS
    assert enc.input_ids[0] == tok.token_to_id("[CLS]")
    assert enc.input_ids[-1] == tok.token_to_id("[SEP]")
    assert len(enc.input_ids) == len(enc.attention_mask) == len(enc.token_type_ids)


def test_last_user_turn_gets_budget_first(tok: Tokenizer) -> None:
    enc = build_inputs(tok, _req(("user", "PREVIOUS"), ("user", "alpha " * 300)))
    assert len(enc.input_ids) == MAX_TOKENS
    assert "previous" not in _middle(tok, enc.input_ids)


def test_earlier_segments_fill_remaining_budget(tok: Tokenizer) -> None:
    req = _req(("system", "be brief"), ("user", "firstword " * 400), ("user", "second one"))
    enc = build_inputs(tok, req)
    text = _middle(tok, enc.input_ids)
    assert len(enc.input_ids) == MAX_TOKENS
    assert text.startswith("second one")
    assert "firstword" in text
    assert "be brief" not in text  # the previous user turn used up the budget


def test_no_user_text_gives_cls_sep(tok: Tokenizer) -> None:
    image = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    enc = build_inputs(tok, _req(("user", image)))
    assert enc.input_ids == [tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")]


class _SpyTokenizer:
    """Wraps the real tokenizer and records how much text each encode() call was given."""

    def __init__(self, inner: Tokenizer) -> None:
        self.inner, self.lengths = inner, []  # type: list[int]

    def token_to_id(self, token: str) -> int | None:
        return self.inner.token_to_id(token)

    def encode(self, text: str, add_special_tokens: bool = True) -> Any:
        self.lengths.append(len(text))
        return self.inner.encode(text, add_special_tokens=add_special_tokens)


def test_huge_segment_tokenizes_only_a_prefix(tok: Tokenizer) -> None:
    spy = _SpyTokenizer(tok)
    enc = build_inputs(spy, _req(("user", "word " * 400_000)))  # type: ignore[arg-type]
    assert len(enc.input_ids) == MAX_TOKENS
    assert max(spy.lengths) <= MAX_TOKENS * 16


def test_system_prompt_capped(tok: Tokenizer) -> None:
    req = _req(("system", "boilerplate " * 400), ("user", "hi"))
    enc = build_inputs(tok, req)
    # [CLS] hi [SEP] <system head> [SEP]
    assert len(enc.input_ids) == 4 + SYSTEM_BUDGET
    assert "boilerplate" in _middle(tok, enc.input_ids)
