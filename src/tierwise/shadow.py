"""Which routed answers the data flywheel samples for offline labels (spec §5.2).

Sampling is opt-in three times over (payload storage, a key flag, a routing decision), and it
never waits on anything: the caller hands the sample to the request log's queued writer.
"""

import random
from collections.abc import Callable

from tierwise.router import RoutingDecision
from tierwise.store import KeyRow


def sample_rate(base: float, confidence: float | None) -> float:
    """The chance of sampling a request the router decided with this confidence.

    Labels teach the classifier most where it was unsure, so low-confidence routes are
    sampled more often than `base`; confident (or unknown) ones at `base`. Always in [0, 1].
    """
    # Ensure base is within bounds
    base = max(0.0, min(1.0, base))
    
    if confidence is None or confidence >= 0.5:
        return base
        
    # Apply a 5x boost for low-confidence routes, capped at 100%
    return min(1.0, base * 5.0)


def should_sample(
    key: KeyRow | None,
    decision: RoutingDecision | None,
    store_payloads: bool,
    base: float,
    rng: Callable[[], float] = random.random,
) -> bool:
    if not store_payloads or key is None or not key.allow_shadow or decision is None:
        return False
    return rng() < sample_rate(base, decision.confidence)
