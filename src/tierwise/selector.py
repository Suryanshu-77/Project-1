"""Pick the cheapest model that can handle a request and clears its quality bar.

The result is an ordered candidate list: the chosen model first, then the failover chain.
"""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from tierwise.affinity import AffinityTable, prefix_boundaries, split_prefix
from tierwise.options import Option, expand, forced
from tierwise.registry import ModelSpec, Registry
from tierwise.router.modes import Mode
from tierwise.router.router import RoutingResult
from tierwise.tokens import estimate_output, estimate_tokens

BAR_BASE = 0.30
BAR_SPAN = 0.65
MODE_SHIFT: dict[Mode, float] = {"cheap": -0.10, "balanced": 0.0, "quality": 0.10}
LOW_CONFIDENCE = 0.3
BAR_STEP = 0.10  # upward bias when the router is unsure
AFFINITY_SLACK = 1.10  # keep a warm incumbent if it costs at most 10% more
REFERENCE_DIFFICULTY = 0.5  # registry effort shifts are measured at this difficulty

Reason = Literal["override", "rules", "learned", "affinity", "fallback"]


class NoEligibleModel(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Selection:
    candidates: list[Option]  # never empty; candidates[0] serves the request
    reason: Reason
    bar: float | None
    prefix_hash: str | None  # the conversation so far (for the log)
    prompt_hash: str | None  # the whole prompt; record it for the model that serves it


def quality_bar(difficulty: float, confidence: float, mode: Mode) -> float:
    bar = BAR_BASE + BAR_SPAN * difficulty + MODE_SHIFT[mode]
    if confidence < LOW_CONFIDENCE:
        bar += BAR_STEP
    return max(0.0, min(1.0, bar))


def effective_quality(option: Option, difficulty: float) -> float:
    """Quality of ``option`` for a request of this difficulty.

    Lower effort costs little on easy requests and a lot on hard ones, so a level's quality
    penalty scales with difficulty (registry values hold at REFERENCE_DIFFICULTY). Bonuses
    for effort above the provider default are not scaled.
    """
    shift = option.quality - option.spec.quality
    if shift < 0:
        shift *= difficulty / REFERENCE_DIFFICULTY
    return max(0.0, min(1.0, option.spec.quality + shift))


def estimated_cost(
    spec: ModelSpec, est_input: int, est_output: int, warm_prefix_tokens: int
) -> float:
    """Expected USD for this request on ``spec``; warm prefix tokens get the cache-read price."""
    read_price = spec.price_cache_read if spec.price_cache_read is not None else spec.price_in
    warm = min(warm_prefix_tokens, est_input)
    total = warm * read_price + (est_input - warm) * spec.price_in + est_output * spec.price_out
    return total / 1_000_000


def select(
    result: RoutingResult,
    registry: Registry,
    available_providers: Collection[str],
    affinity: AffinityTable,
) -> Selection:
    """Choose for an ``auto`` or ``tier`` target. Raises NoEligibleModel if nothing fits."""
    req, target = result.request, result.target
    est_input, est_output = estimate_tokens(req), estimate_output(req)
    prefix_hash, _ = split_prefix(req)
    boundaries = prefix_boundaries(req)
    prompt_hash = boundaries[-1][0] if boundaries else None
    earlier = boundaries[:-1]  # the new last message can't be cached yet

    fits = [
        m
        for m in registry.models
        if m.provider in available_providers
        and result.required_caps <= m.caps
        and est_input + est_output <= m.context
        and (target.kind != "tier" or m.tier == target.tier)
    ]
    if not fits:
        needs = ", ".join(sorted(result.required_caps)) or "none"
        scope = f" in tier '{target.tier}'" if target.kind == "tier" else ""
        raise NoEligibleModel(
            f"No configured model{scope} supports this request (needs capabilities: {needs}; "
            f"about {est_input + est_output} tokens)."
        )

    if req.reasoning_effort is not None:  # the client's effort always wins
        options = [o for m in fits if (o := forced(m, req.reasoning_effort)) is not None]
        if not options:
            scope = f" in tier '{target.tier}'" if target.kind == "tier" else ""
            raise NoEligibleModel(
                f"No configured model{scope} supports reasoning effort "
                f"'{req.reasoning_effort}' for this request."
            )
    else:
        options = [o for m in fits for o in expand(m)]

    cost: dict[int, float] = {}
    warm_tokens: dict[int, int] = {}
    for option in options:
        warm = affinity.longest_warm(
            earlier,
            option.spec.id,
            option.spec.cache_ttl_s,
            option.wire_effort,
            option.spec.effort_in_cache_key,
        )
        warm_tokens[id(option)] = warm
        thinking = option.thinking_tokens * option.spec.price_out / 1_000_000
        cost[id(option)] = estimated_cost(option.spec, est_input, est_output, warm) + thinking

    def by_cost_key(o: Option) -> tuple[float, int, float, str]:
        return (cost[id(o)], o.rank, -o.quality, o.spec.id)

    by_cost = sorted(options, key=by_cost_key)
    if target.kind == "tier":
        return Selection(by_cost, "override", None, prefix_hash, prompt_hash)

    decision = result.decision
    assert decision is not None and target.mode is not None  # auto targets always have both
    bar = quality_bar(decision.difficulty, decision.confidence, target.mode)
    quality = {id(o): effective_quality(o, decision.difficulty) for o in options}
    clear = [o for o in by_cost if quality[id(o)] >= bar]
    by_quality = sorted(options, key=lambda o: (-quality[id(o)], o.rank, o.spec.id))
    if not clear:
        return Selection(by_quality, "fallback", bar, prefix_hash, prompt_hash)

    below = [o for o in by_quality if quality[id(o)] < bar]
    reason: Reason = decision.source
    incumbent_id = affinity.chain_incumbent(earlier, lambda mid: _ttl(registry, mid))
    # the incumbent's cheapest *warm* option: a cold one (different effort on a model whose
    # cache depends on effort) has no cache to keep
    incumbent = next(
        (o for o in clear if o.spec.id == incumbent_id and warm_tokens[id(o)] > 0), None
    )
    if incumbent is not None and cost[id(incumbent)] <= AFFINITY_SLACK * cost[id(clear[0])]:
        clear = [incumbent, *(o for o in clear if o is not incumbent)]
        reason = "affinity"
    return Selection([*clear, *below], reason, bar, prefix_hash, prompt_hash)


def _ttl(registry: Registry, model_id: str) -> int | None:
    spec = registry.get(model_id)
    return spec.cache_ttl_s if spec is not None else None
