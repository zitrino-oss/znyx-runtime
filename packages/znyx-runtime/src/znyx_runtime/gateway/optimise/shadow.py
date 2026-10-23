"""Shadow tests: a technique that can change answers (stage C, and right-sizing) is tried
on a sample of a key's calls before it may be switched on.

The client always gets the normal answer. Afterwards the gateway sends the variant (the
technique applied to the request as it was sent, or the smaller model) and compares the
two answers. Only the verdict reaches the control plane: agreed or not, how close, by
which method, what the technique would have saved, and what the comparison cost. The
variant's answer is discarded.

Comparison: NLI both ways when the policy configures an NLI endpoint (the hallucination
check's ``nli`` block): the answers agree when each entails the other. Otherwise a lexical
comparison: JSON answers compared as data, text by the share of content words in common.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from typing import Any, Callable, Optional, Tuple

DEFAULT_PERCENT = 5
DEFAULT_CALLS_PER_DAY = 200
NLI_AGREES_AT = 0.5
LEXICAL_AGREES_AT = 0.6
_WORD = re.compile(r"[a-z0-9]{2,}")
_STOP = frozenset("""the a an and or of to in on for is are was were be been it this that with as by at from
your you we our they their he she his her its can will would should could do does did have has
had so if then than but there here which who what when where how about into over under more most less""".split())


def sampled(percent: Optional[int]) -> bool:
    return random.random() * 100 < (percent or DEFAULT_PERCENT)


def within_daily_allowance(store, caller: str, limit: Optional[int]) -> bool:
    """Count one shadow call for today; False when the key's allowance is used up (or the
    count cannot be kept: a shadow call is never worth a call that cannot be counted)."""
    # "pin:": never evicted by other traffic (one key per gateway key and day).
    key = f"pin:gw:shadow:{caller}:{time.strftime('%Y%m%d', time.gmtime())}"
    try:
        used = store.incr(key, {"calls": 1}, ttl_seconds=2 * 86400)
    except Exception:  # noqa: BLE001
        return False
    return used.get("calls", 0) <= (limit or DEFAULT_CALLS_PER_DAY)


def _words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def lexical_agreement(a: str, b: str) -> float:
    try:
        ja, jb = json.loads(a), json.loads(b)
    except (ValueError, RecursionError):
        ja = jb = None
    if ja is not None and jb is not None:
        if ja == jb:
            return 1.0
        if isinstance(ja, dict) and isinstance(jb, dict) and (ja or jb):
            keys = set(ja) | set(jb)
            return sum(1 for k in keys if ja.get(k) == jb.get(k)) / len(keys)
        return 0.0
    wa, wb = _words(a), _words(b)
    # Lexical scores are diagnostic only, but must not label negated answers as
    # identical. Keep polarity even when the rest of the wording is the same.
    polarity = {"not", "no", "never", "neither", "without", "yes"}
    if wa & polarity != wb & polarity:
        return 0.0
    if not wa and not wb:
        return 1.0 if a.strip() == b.strip() else 0.0
    return len(wa & wb) / len(wa | wb)


def compare(served: str, variant: str, nli: Optional[Callable] = None) -> Tuple[bool, float, str]:
    """(agreed, score 0..1, method)."""
    if nli is not None:
        try:
            forward = float(nli(served, [variant])[0])
            backward = float(nli(variant, [served])[0])
            if not all(math.isfinite(s) and 0 <= s <= 1 for s in (forward, backward)):
                raise ValueError("NLI scores must be finite probabilities")
            score = min(forward, backward)
            return score >= NLI_AGREES_AT, round(score, 4), "nli"
        except Exception:  # noqa: BLE001 - the endpoint failed: fall back, and say so
            pass
    score = lexical_agreement(served, variant)
    return score >= LEXICAL_AGREES_AT, round(score, 4), "lexical"


def nli_from_policy(policy: dict, egress_sink, request) -> Optional[Callable]:
    """The policy's NLI scorer (the hallucination check's ``nli`` block), or None."""
    config = policy.get("hallucination") if isinstance(policy.get("hallucination"), dict) else {}
    if not (isinstance(config.get("nli"), dict) and config["nli"].get("enabled")):
        return None
    from znyx_core.engine.quality.nli_client import nli_scorer_from_config
    try:
        return nli_scorer_from_config(config, runtime_policy=policy.get("runtime_policy"), policy=policy,
                                      egress_sink=egress_sink, request=request)
    except Exception:  # noqa: BLE001
        return None


def pick(candidates, chooser: Any = random) -> Optional[str]:
    """One technique to shadow on this call (one at a time keeps the cost bounded)."""
    candidates = list(candidates)
    return chooser.choice(candidates) if candidates else None
