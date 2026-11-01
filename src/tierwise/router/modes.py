"""Parse the ``model`` field into a routing target, and handle ``[critical]``/``[easy]`` tags."""

import re
from dataclasses import dataclass
from typing import Literal, cast, get_args

from tierwise.schema import ChatRequest, TextPart

Mode = Literal["cheap", "balanced", "quality"]
Tier = Literal["basic", "moderate", "flagship"]
Tag = Literal["critical", "easy"]

VIRTUAL_MODELS: tuple[str, ...] = (
    "auto",
    "auto:cheap",
    "auto:balanced",
    "auto:quality",
    "tier:basic",
    "tier:moderate",
    "tier:flagship",
)

# A tag must stand alone (not part of code like flags[critical]); only spaces and tabs
# around it are removed, so the surrounding lines are left intact.
_TAG = re.compile(r"[ \t]*(?<!\S)\[(critical|easy)\](?!\S)[ \t]*", re.IGNORECASE)


class UnknownRoutingValue(ValueError):
    """``auto:<x>`` or ``tier:<x>`` with an unknown ``x``."""


class EmptyAfterTag(ValueError):
    """The last user message contained nothing but a routing tag."""


@dataclass(frozen=True)
class RouteTarget:
    kind: Literal["auto", "tier", "model"]
    mode: Mode | None = None
    tier: Tier | None = None
    model_id: str | None = None


def parse_target(model: str) -> RouteTarget:
    if model == "auto":
        return RouteTarget(kind="auto", mode="balanced")
    prefix, sep, value = model.partition(":")
    if sep and prefix == "auto":
        if value not in get_args(Mode):
            raise UnknownRoutingValue(model)
        return RouteTarget(kind="auto", mode=cast(Mode, value))
    if sep and prefix == "tier":
        if value not in get_args(Tier):
            raise UnknownRoutingValue(model)
        return RouteTarget(kind="tier", tier=cast(Tier, value))
    return RouteTarget(kind="model", model_id=model)


def _strip(text: str, found: set[str]) -> str:
    def remove(match: re.Match[str]) -> str:
        found.add(match.group(1).lower())
        return " "

    return _TAG.sub(remove, text).strip()


def extract_tag(req: ChatRequest) -> tuple[ChatRequest, Tag | None]:
    """Return a copy of ``req`` with tags removed from its last user message, and the tag."""
    last_user = next(
        (i for i in range(len(req.messages) - 1, -1, -1) if req.messages[i].role == "user"),
        None,
    )
    if last_user is None:
        return req, None
    cleaned = req.model_copy(deep=True)
    message = cleaned.messages[last_user]
    found: set[str] = set()
    if isinstance(message.content, str):
        message.content = _strip(message.content, found)
        empty = not message.content
    elif message.content:
        for part in message.content:
            if isinstance(part, TextPart):
                part.text = _strip(part.text, found)
        # An empty text block is rejected upstream, so drop parts the tag emptied.
        message.content = [
            p for p in message.content if not (isinstance(p, TextPart) and not p.text)
        ]
        empty = not message.content
    else:
        empty = False
    if not found:
        return req, None
    if empty:
        raise EmptyAfterTag("the message is empty after removing the routing tag")
    return cleaned, "critical" if "critical" in found else "easy"
