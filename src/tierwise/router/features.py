"""Cheap, deterministic signals about how demanding a request is."""

import re
from dataclasses import dataclass

from tierwise.schema import ChatRequest, ImagePart
from tierwise.tokens import estimate_tokens, text_of

HIGH_STAKES = (
    "production", "security", "legal", "medical", "financial", "migrate", "migration",
    "compliance", "audit", "outage", "incident", "do not get this wrong", "critical",
)  # fmt: skip
REASONING = (
    "prove", "derive", "analyze", "analyse", "design", "architect", "architecture",
    "optimize", "debug", "refactor", "step by step", "trade-off", "tradeoff", "compare",
    "research",
)  # fmt: skip
EASY = (
    "translate", "summarize", "summarise", "what is", "define", "hello", "hi", "thanks",
    "fix typo", "spell", "rephrase", "format",
)  # fmt: skip

_FENCE = re.compile(r"```")
_CODE_LINE = re.compile(
    r"^\s*(def |class |import |from \S+ import |function |#include|public |return\b)"
    r"|[;{]\s*$",
    re.MULTILINE,
)
_MATH = re.compile(r"\$[^$\n]+\$|\\(frac|int|sum|sqrt)|\d\s*[\^*/=]\s*\d|O\(n")


def _pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    words = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"\b({words})\b", re.IGNORECASE)


_HIGH_STAKES, _REASONING, _EASY = _pattern(HIGH_STAKES), _pattern(REASONING), _pattern(EASY)


@dataclass(frozen=True)
class Features:
    est_tokens: int
    user_turns: int
    code_blocks: int
    has_math: bool
    high_stakes_hits: int
    reasoning_hits: int
    easy_hits: int
    has_tools: bool
    has_images: bool


def _hits(pattern: re.Pattern[str], text: str) -> int:
    """Number of distinct keywords from the list that appear."""
    return len({m.lower() for m in pattern.findall(text)})


def extract_features(req: ChatRequest) -> Features:
    users = [m for m in req.messages if m.role == "user"]
    last = text_of(users[-1].content) if users else ""
    code_blocks = len(_FENCE.findall(last)) // 2
    if code_blocks == 0 and _CODE_LINE.search(last):
        code_blocks = 1
    has_images = any(
        isinstance(part, ImagePart)
        for m in req.messages
        if isinstance(m.content, list)
        for part in m.content
    )
    return Features(
        est_tokens=estimate_tokens(req),
        user_turns=len(users),
        code_blocks=code_blocks,
        has_math=bool(_MATH.search(last)),
        high_stakes_hits=_hits(_HIGH_STAKES, last),
        reasoning_hits=_hits(_REASONING, last),
        easy_hits=_hits(_EASY, last),
        has_tools=bool(req.tools),
        has_images=has_images,
    )
