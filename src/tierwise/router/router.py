"""The routing pipeline: override tag, rules, learned router, and a fallback that never fails."""

import logging
from dataclasses import dataclass

from tierwise.router import RoutingDecision
from tierwise.router.learned import LearnedRouter
from tierwise.router.modes import RouteTarget, extract_tag, parse_target
from tierwise.router.rules import rules_decision
from tierwise.schema import ChatRequest, ImagePart

logger = logging.getLogger(__name__)

FALLBACK = RoutingDecision(difficulty=0.7, confidence=0.0, source="fallback")
RULES_SHORTCUT = 0.8  # rules this sure of themselves skip the learned router


@dataclass(frozen=True)
class RoutingResult:
    target: RouteTarget
    decision: RoutingDecision | None  # None when the client picked a tier or a model
    request: ChatRequest  # routing tags removed
    required_caps: frozenset[str]


def required_caps(req: ChatRequest) -> frozenset[str]:
    caps: set[str] = set()
    if req.tools:
        caps.add("tools")
    if any(
        isinstance(part, ImagePart)
        for m in req.messages
        if isinstance(m.content, list)
        for part in m.content
    ):
        caps.add("vision")
    if req.response_format:
        caps.add("json")
    return frozenset(caps)


async def route(req: ChatRequest, learned: LearnedRouter | None = None) -> RoutingResult:
    """Raises only for bad client input (UnknownRoutingValue, EmptyAfterTag)."""
    target = parse_target(req.model)
    cleaned, tag = extract_tag(req)  # tags are never forwarded, whatever the target
    decision: RoutingDecision | None = None
    if target.kind == "auto":
        if tag is not None:
            decision = RoutingDecision(
                difficulty=1.0 if tag == "critical" else 0.0, confidence=1.0, source="override"
            )
        else:
            try:
                decision = rules_decision(cleaned)
                if learned is not None and decision.confidence < RULES_SHORTCUT:
                    decision = await learned.decide(cleaned)
            except Exception:
                logger.exception("router failed; using the fallback decision")
                decision = FALLBACK
    return RoutingResult(target, decision, cleaned, required_caps(cleaned))
