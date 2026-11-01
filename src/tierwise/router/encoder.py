"""Learned-router inputs: token-space truncation of a request into one BERT-style sequence."""

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from tierwise.schema import ChatRequest
from tierwise.tokens import text_of

MAX_TOKENS = 256  # including [CLS] and [SEP]


@dataclass(frozen=True)
class EncodedInput:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    # tokenizer.json ships with truncation and fixed padding at 128; we budget tokens ourselves
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def special_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer has no {token} token")
    return token_id


def segments(req: ChatRequest) -> list[str]:
    """What the classifier reads, most important first: last user turn, previous one, system."""
    users = [text_of(m.content) for m in req.messages if m.role == "user"]
    system = "\n".join(text_of(m.content) for m in req.messages if m.role == "system")
    ordered = [*reversed(users[-2:]), system]
    return [s for s in ordered if s.strip()]


def build_inputs(
    tokenizer: Tokenizer, req: ChatRequest, max_tokens: int = MAX_TOKENS
) -> EncodedInput:
    cls_id, sep_id = special_id(tokenizer, "[CLS]"), special_id(tokenizer, "[SEP]")
    budget = max_tokens - 2  # [CLS] ... [SEP]
    body: list[int] = []
    for text in segments(req):
        if body:
            if budget < 2:  # a separator plus at least one token
                break
            body.append(sep_id)
            budget -= 1
        if budget <= 0:
            break
        # cut from the tail: the start of a turn carries the instruction
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:budget]
        body.extend(ids)
        budget -= len(ids)
    input_ids = [cls_id, *body, sep_id]
    return EncodedInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        token_type_ids=[0] * len(input_ids),
    )
