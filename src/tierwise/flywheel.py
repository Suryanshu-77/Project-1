"""Offline flywheel labels (`tierwise flywheel label`, spec §5.2).

For each pending shadow sample: ask a contrasting model the same request, then have a judge
score the cheaper of the two answers. `needs_strong = score < 4`, the mapping the M4 training
labels used, so shadow labels and the base dataset mean the same thing. Every call is paid
for, so the whole run stops at a monthly budget.
"""

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

from tierwise.budgets import month_of
from tierwise.costs import cost_usd
from tierwise.providers.base import Provider, ProviderError
from tierwise.registry import ModelSpec, Registry
from tierwise.reliability.retry import RetryPolicy, with_retries
from tierwise.router.router import required_caps
from tierwise.schema import ChatRequest, ChatResponse, Message
from tierwise.selector import fits
from tierwise.store import RequestLog, ShadowSample
from tierwise.tokens import estimate_output, estimate_tokens, text_of

JUDGE_MARKER = "tierwise-judge-v1"
GOOD_ENOUGH = 4  # a cheaper answer scored below this needs a strong model
RUBRIC = f"""You grade answers for {JUDGE_MARKER}.
You see a conversation and two answers to its last turn. Score ONLY the answer you are told
to score, from 1 to 5, using the other one as a reference:
5 = as good as the reference or better; 4 = minor gaps a user would not mind;
3 = noticeably worse; 2 = misses important parts; 1 = wrong or unhelpful.
Reply with JSON only: {{"score": <1-5>}}"""


@dataclass(frozen=True)
class LabelRun:
    labeled: int
    failed: int
    left: int
    spent_usd: float
    stopped_by_budget: bool


def contrast_for(
    routed: ModelSpec, request: ChatRequest, registry: Registry, providers: set[str]
) -> ModelSpec | None:
    """The model whose answer the routed one is compared with (spec §5.2, §5.3).

    `baseline_model` when a cheaper, non-flagship model answered; the cheapest `basic` model
    when a flagship (the baseline or another) answered: a flagship is the strong side. Either
    way it must be configured (its provider in `providers`) and able to serve `request`
    (`fits`). None when nothing qualifies: the sample then fails before any paid call.
    """
    caps = required_caps(request)
    tokens = estimate_tokens(request) + estimate_output(request)
    
    if routed.id != registry.baseline_model and routed.tier != "flagship":
        baseline = registry.get(registry.baseline_model)
        if baseline and baseline.provider in providers and fits(baseline, caps, tokens):
            return baseline
        return None
        
    candidates = [
        m for m in registry.models 
        if m.tier == "basic" and m.provider in providers and fits(m, caps, tokens)
    ]
    return min(candidates, key=lambda m: (m.price_in + m.price_out, m.id), default=None)


def _transcript(request: ChatRequest) -> str:
    return "\n\n".join(f"{m.role}: {text_of(m.content)}" for m in request.messages)


def judge_messages(prompt: str, cheap: str, strong: str, seed: str) -> list[Message]:
    """The rubric plus both answers, in an order seeded by the sample id: judges favour one
    position, so a fixed order would bias every label the same way."""
    cheap_first = random.Random(seed).random() < 0.5
    first, second = (cheap, strong) if cheap_first else (strong, cheap)
    target = "A" if cheap_first else "B"
    body = (f"Conversation:\n{prompt}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}\n\n"
            f"Score answer {target}.")  # fmt: skip
    return [Message(role="system", content=RUBRIC), Message(role="user", content=body)]


def parse_score(text: str) -> int:
    match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate ```json fences and chatter
    if match is None:
        raise ValueError(f"judge reply has no JSON object: {text[:80]!r}")
    score = json.loads(match.group(0)).get("score")
    if isinstance(score, str):  # small judges often quote it: {"score": "5"}
        try:
            score = float(score)
        except ValueError:
            pass  # left a string: rejected below
    # bool is an int subclass, so `true` would pass as 1; 4.0 is a fine way to write 4. The
    # range check comes first: int() of inf (JSON Infinity, 1e400) raises OverflowError.
    number = isinstance(score, int | float) and not isinstance(score, bool)
    if not (number and 1 <= score <= 5 and score == int(score)):
        raise ValueError(f"judge score {score!r} is not an integer from 1 to 5")
    return int(score)


async def _ask(
    provider: Provider, spec: ModelSpec, request: ChatRequest, policy: RetryPolicy
) -> tuple[ChatResponse, float]:
    """One paid call, retried on transient errors (429, 5xx, timeouts) like the gateway does."""
    response = await with_retries(partial(provider.chat, request, spec.wire_id), policy)
    return response, cost_usd(response.usage, spec)


async def _label(
    sample: ShadowSample,
    registry: Registry,
    providers: dict[str, Provider],
    judge: ModelSpec,
    now: float,
    policy: RetryPolicy,
) -> ShadowSample:
    """One sample, labeled or failed; its cost counts either way."""
    spent = 0.0
    try:
        routed = registry.get(sample.routed_model)
        if routed is None:
            raise ValueError(f"model {sample.routed_model!r} is no longer in the registry")
        request = ChatRequest.model_validate_json(sample.request_json)
        contrast = contrast_for(routed, request, registry, set(providers))
        if contrast is None:
            raise ValueError(f"no configured contrast model can serve this {routed.id!r} sample")
        reply, cost = await _ask(providers[contrast.provider], contrast, request, policy)
        spent += cost
        contrast_answer = reply.content or ""
        if reply.tool_calls or not contrast_answer or reply.finish_reason != "stop":
            # an empty, cut-off or tool-call answer can't be compared: judging it would
            # store a confident, wrong label
            raise ValueError(f"contrast answer unusable (finish_reason={reply.finish_reason})")
        cheap_is_routed = contrast.tier == "flagship" or contrast.id == registry.baseline_model
        cheap, strong = (
            (sample.routed_answer, contrast_answer)
            if cheap_is_routed
            else (contrast_answer, sample.routed_answer)
        )
        grading = ChatRequest(
            model=judge.id,
            messages=judge_messages(_transcript(request), cheap, strong, seed=sample.id),
            temperature=0,
        )
        verdict, cost = await _ask(providers[judge.provider], judge, grading, policy)
        spent += cost
        score = parse_score(verdict.content or "")
    except (ProviderError, ValueError) as exc:
        return replace(sample, status="failed", error=str(exc)[:500], label_cost_usd=spent,
                       labeled_ts=now)  # fmt: skip
    return replace(
        sample,
        status="labeled",
        contrast_model=contrast.id,
        contrast_answer=contrast_answer,
        judge_score=score,
        needs_strong=score < GOOD_ENOUGH,
        label_cost_usd=spent,
        labeled_ts=now,
    )


async def label_pending(
    log: RequestLog,
    registry: Registry,
    providers: dict[str, Provider],
    judge: ModelSpec,
    budget_usd: float,
    limit: int,
    clock: Callable[[], float] = time.time,
    policy: RetryPolicy = RetryPolicy(),  # noqa: B008 (frozen dataclass)
) -> LabelRun:
    """Label pending samples, oldest first, until `limit`, or until this month's label spend
    (earlier runs included) reaches `budget_usd`."""
    if judge.provider not in providers:
        raise ValueError(f"judge model {judge.id!r}: provider {judge.provider!r} not configured")
    samples = await log.pending_samples(limit)
    month_spend = await log.label_spend(month_of(clock()))
    labeled = failed = 0
    run_spent = 0.0
    for done, sample in enumerate(samples):
        if month_spend + run_spent >= budget_usd:
            return LabelRun(labeled, failed, len(samples) - done, run_spent, True)
        result = await _label(sample, registry, providers, judge, clock(), policy)
        await log.save_label(result)
        run_spent += result.label_cost_usd
        if result.status == "labeled":
            labeled += 1
        else:
            failed += 1
    return LabelRun(labeled, failed, 0, run_spent, False)
