"""Numerical consistency detector (deferred backlog, deterministic).

Flags EXPLICIT arithmetic equations in text whose stated result is wrong — e.g.
``2 + 2 = 5``, ``$10 + $20 = $40``, ``10% + 20% = 35%``. It only evaluates a literal
``number <op> number = number`` pattern, so it cannot false-positive on prose; the
harder reasoning-heavy numeric claims are deferred to an LLM judge.

Pure-rules, no dependencies — an output-stage detector in the deterministic family.
"""
import re
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# A number: optional sign, optional leading $, digits with thousands-commas, optional
# decimals, optional trailing %. (We strip $/%/commas before evaluating.)
_NUM = r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?"
# "a op b = c" — op ∈ + - * x × / ÷ . Anchored so both operands and the result are numbers.
_EQN_RE = re.compile(
    rf"(?P<a>{_NUM})\s*(?P<op>[-+x*×/÷])\s*(?P<b>{_NUM})\s*=\s*(?P<c>{_NUM})",
    re.IGNORECASE,
)


def _to_float(tok: str) -> Optional[float]:
    t = tok.strip().replace(",", "").replace("$", "").replace("%", "").replace("×", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _apply(a: float, op: str, b: float) -> Optional[float]:
    op = op.lower()
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op in ("x", "*", "×"):
        return a * b
    if op in ("/", "÷"):
        return a / b if b != 0 else None
    return None


def _mismatch(expected: float, stated: float) -> bool:
    # Absolute + relative tolerance so float rounding ("0.1+0.2=0.3") isn't flagged.
    return abs(expected - stated) > max(0.01, 1e-3 * abs(expected))


class NumericalConsistencyDetector:
    """Deterministic arithmetic-equation consistency check (OWASP LLM09-adjacent)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "WARN").upper()
        self.block_threshold = self.config.get("block_threshold", 60)

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        hits: List[RuleHit] = []
        for m in _EQN_RE.finditer(text):
            a, b, c = _to_float(m.group("a")), _to_float(m.group("b")), _to_float(m.group("c"))
            if a is None or b is None or c is None:
                continue
            expected = _apply(a, m.group("op"), b)
            if expected is None:  # e.g. division by zero — can't verify
                continue
            if _mismatch(expected, c):
                hits.append(RuleHit(
                    rule_id="numerical_consistency.bad_equation",
                    message=f"Arithmetic error: '{m.group(0).strip()}' (expected {expected:g})",
                    severity=Severity.HIGH,
                ))

        if not hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(hits)
        if self.action == "BLOCK" and risk_score >= self.block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=hits,
                user_message="This response was blocked: it contains an incorrect calculation.",
                developer_message=f"numerical_consistency: {len(hits)} incorrect equation(s)",
            )
        return DetectorResult(
            decision=Decision.WARN, risk_score=risk_score, rule_hits=hits,
            developer_message=f"numerical_consistency: {len(hits)} incorrect equation(s) (advisory)",
        )
