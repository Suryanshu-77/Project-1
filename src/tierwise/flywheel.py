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

from tierwise.budgets import month_of
from tierwise.costs import cost_usd
from tierwise.providers.base import Provider, ProviderError
from tierwise.registry import ModelSpec, Registry
from tierwise.schema import ChatRequest, Message
from tierwise.store import RequestLog, ShadowSample
from tierwise.tokens import text_of

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


def contrast_for(routed: ModelSpec, registry: Registry, providers: set[str]) -> ModelSpec | None:
    """The flagship (baseline) for an answer from a cheaper model; the cheapest basic model for
    an answer from the flagship. None when that model's provider isn't configured."""
    if routed.id != registry.baseline_model and routed.tier != "flagship":
        baseline = registry.get(registry.baseline_model)
        return baseline if baseline is not None and baseline.provider in providers else None
    basic = [m for m in registry.models if m.tier == "basic" and m.provider in providers]
    return min(basic, key=lambda m: (m.price_in + m.price_out, m.id), default=None)


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
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"judge score {score!r} is not an integer from 1 to 5")
    return score


async def _ask(provider: Provider, spec: ModelSpec, request: ChatRequest) -> tuple[str, float]:
    response = await provider.chat(request, spec.wire_id)
    return response.content or "", cost_usd(response.usage, spec)


async def _label(
    sample: ShadowSample,
    registry: Registry,
    providers: dict[str, Provider],
    judge: ModelSpec,
    now: float,
) -> ShadowSample:
    """One sample, labeled or failed; its cost counts either way."""
    spent = 0.0
    try:
        routed = registry.get(sample.routed_model)
        if routed is None:
            raise ValueError(f"model {sample.routed_model!r} is no longer in the registry")
        contrast = contrast_for(routed, registry, set(providers))
        if contrast is None:
            raise ValueError(f"no configured contrast model for {routed.id!r}")
        request = ChatRequest.model_validate_json(sample.request_json)
        contrast_answer, cost = await _ask(providers[contrast.provider], contrast, request)
        spent += cost
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
        verdict, cost = await _ask(providers[judge.provider], judge, grading)
        spent += cost
        score = parse_score(verdict)
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
        result = await _label(sample, registry, providers, judge, clock())
        await log.save_label(result)
        run_spent += result.label_cost_usd
        if result.status == "labeled":
            labeled += 1
        else:
            failed += 1
    return LabelRun(labeled, failed, 0, run_spent, False)
