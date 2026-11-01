from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer

from tierwise.router.encoder import MAX_TOKENS, build_inputs, load_tokenizer, segments
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
    req = _req(("system", "systemword " * 400), ("user", "first question"), ("user", "second one"))
    enc = build_inputs(tok, req)
    text = _middle(tok, enc.input_ids)
    assert len(enc.input_ids) == MAX_TOKENS
    assert text.startswith("second one")
    assert "first question" in text
    assert "systemword" in text


def test_no_user_text_gives_cls_sep(tok: Tokenizer) -> None:
    image = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    enc = build_inputs(tok, _req(("user", image)))
    assert enc.input_ids == [tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")]
