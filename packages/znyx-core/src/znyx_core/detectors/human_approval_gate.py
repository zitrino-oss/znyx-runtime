"""Human-approval gate (OWASP LLM03 - Excessive Agency).

OWASP's LLM03 mitigation #6 is "require user approval": a human confirms high-impact,
irreversible, or externally-visible actions before they happen. ``excessive_agency``
already risk-scores what an action DOES; this detector asks the orthogonal question of
whether anyone agreed to it, and blocks when the answer is no.

Approval is proved by the CALLER, in request metadata:

    metadata = {"approval": {"approved_by": "alice@acme.com", "approval_id": "CHG-4471"}}
    metadata = {"human_approved": True, "approved_by": "alice@acme.com"}

Two deliberate design choices:

* **The action taxonomy is imported, not re-declared.** It lives in ``excessive_agency``
  and is shared, so the two detectors can never drift into disagreeing about what counts
  as destructive — which would be worse than either being wrong alone.

* **A bare boolean is not approval.** ``human_approved: true`` with no approver identity
  is accepted only when ``require_approver_identity`` is off. An audit trail that cannot
  name who approved a payout is not an audit trail, and the model itself can emit a
  boolean; it cannot emit a signed-in human's identity.

Runs in the ``agent_plan`` and ``agent_loop`` stages, matching ``excessive_agency`` —
a plan is gated before it starts, and each live step is gated as it is taken.
"""
import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score
from znyx_core.detectors.excessive_agency import (
    _COMPILED_ACTIONS,
    build_action_scan_text,
)

# Categories that need a human by default. Mutating actions are excluded: they are
# reversible often enough that gating every one of them would train operators to
# click through approvals, which is how approval fatigue destroys the control.
DEFAULT_GATED_CATEGORIES = (
    "destructive_action",
    "external_side_effect",
    "privilege_escalation",
    "code_execution",
)

# Metadata keys a caller may use to carry approval evidence.
_APPROVAL_BLOCKS = ("approval", "human_approval")
_APPROVED_FLAGS = ("human_approved", "approved", "is_approved")
_APPROVER_KEYS = ("approved_by", "approver", "approver_id", "approver_email")
_APPROVAL_ID_KEYS = ("approval_id", "ticket", "change_id", "request_id")
_TOKEN_KEYS = ("token", "signature", "approval_token", "hmac")
_ISSUED_AT_KEYS = ("issued_at", "approved_at", "timestamp", "iat")


def approval_action_hash(text: str) -> str:
    """The action digest an approval must be bound to.

    Exported so the approving side computes it exactly the way the gate does. Whitespace
    is collapsed first: a plan re-serialised with different indentation is the same
    action, and an approval that broke on pretty-printing would be unusable."""
    normalised = " ".join((text or "").split())
    return hashlib.sha256(normalised.encode("utf-8", "ignore")).hexdigest()


def issue_approval_token(action_hash: str, issued_at: int, approver: str,
                         secret: str) -> str:
    """Mint the token the gate will accept. The approving service calls this; ZNYX only
    ever verifies. Binding the approver into the MAC stops a token issued for one person
    being replayed under another's name."""
    msg = f"{action_hash}.{int(issued_at)}.{approver}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _truthy(v: Any) -> bool:
    """Accept the shapes a JSON caller realistically sends for a yes."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "1", "approved")
    if isinstance(v, (int, float)):
        return v == 1
    return False


def _first_str(source: Dict[str, Any], keys) -> Optional[str]:
    for k in keys:
        v = source.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


class HumanApprovalGateDetector:
    """Blocks irreversible agent actions that carry no human approval (LLM03)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        # BLOCK by default: an unapproved irreversible action is the thing this exists to
        # stop. WARN suits a rollout where approvals are not yet plumbed through.
        self.action = (self.config.get("action") or "BLOCK").upper()
        gated = self.config.get("gated_categories") or DEFAULT_GATED_CATEGORIES
        self.gated_categories = {str(c) for c in gated}
        # An approval with no named approver is not auditable. On by default.
        self.require_approver_identity = bool(
            self.config.get("require_approver_identity", True))
        # Signed approval. Without it the gate can only take the caller's word that a
        # human agreed, and the caller is the agent's own request path. With it, the
        # approval is a MAC bound to THIS action, issued by a service holding a secret
        # ZNYX never mints tokens with, and it expires.
        self.require_signed_approval = bool(
            self.config.get("require_signed_approval", False))
        self.signing_secret = self.config.get("signing_secret") or ""
        self.max_approval_age_seconds = max(
            0, int(self.config.get("max_approval_age_seconds", 900)))

    @staticmethod
    def _approval_evidence(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalise however the caller expressed approval into one shape."""
        if not isinstance(metadata, dict):
            return {}
        block: Dict[str, Any] = {}
        for k in _APPROVAL_BLOCKS:
            v = metadata.get(k)
            if isinstance(v, dict):
                block = v
                break
        approver = _first_str(block, _APPROVER_KEYS) or _first_str(metadata, _APPROVER_KEYS)
        approval_id = _first_str(block, _APPROVAL_ID_KEYS) or _first_str(metadata, _APPROVAL_ID_KEYS)
        flagged = any(_truthy(block.get(f)) for f in _APPROVED_FLAGS) \
            or any(_truthy(metadata.get(f)) for f in _APPROVED_FLAGS)
        token = _first_str(block, _TOKEN_KEYS) or _first_str(metadata, _TOKEN_KEYS)
        issued_at = None
        for src in (block, metadata):
            for k in _ISSUED_AT_KEYS:
                v = src.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    issued_at = int(v)
                    break
            if issued_at is not None:
                break
        # Approval requires an AFFIRMATIVE flag. A name on its own used to count, which
        # meant any caller-controlled string under `approved_by` opened the gate: the
        # agent asking permission was also the thing granting it. A name is now identity
        # evidence for an approval, never the approval itself.
        return {"approved": bool(flagged), "approver": approver,
                "approval_id": approval_id, "token": token, "issued_at": issued_at}

    def _gated_categories_in(self, text: str) -> List[str]:
        """Which gated action categories this plan/step contains.

        Uses excessive_agency's scan-text builder, so `delete_customer_records` is
        classified the same way in both detectors — including the separator
        normalisation that a bare \b pattern needs to see a verb inside an identifier."""
        scan_text = build_action_scan_text(text)
        return [name for pattern, _severity, name, _msg in _COMPILED_ACTIONS
                if name in self.gated_categories and pattern.search(scan_text)]

    def _verify_signature(self, text: str,
                          evidence: Dict[str, Any]) -> Optional[RuleHit]:
        """Return the finding that invalidates this approval, or None if it verifies.

        Four ways a signed approval fails, each reported distinctly so an operator can
        tell a misconfiguration from an attack."""
        if not self.signing_secret:
            # Fail closed. Requiring signatures with no key to check them against would
            # otherwise wave every approval through while appearing to be the strict mode.
            return RuleHit(
                rule_id="human_approval_gate.approval_signing_unconfigured",
                severity=Severity.HIGH,
                message=("Signed approval is required but no signing secret is configured; "
                         "no approval can be verified"),
            )
        token = evidence.get("token")
        issued_at = evidence.get("issued_at")
        if not token or issued_at is None:
            return RuleHit(
                rule_id="human_approval_gate.unsigned_approval",
                severity=Severity.HIGH,
                message="Approval carries no signed token bound to this action",
            )
        if self.max_approval_age_seconds:
            age = int(time.time()) - int(issued_at)
            # Reject future-dated tokens too: a clock the caller controls is not a clock.
            if age > self.max_approval_age_seconds or age < -60:
                return RuleHit(
                    rule_id="human_approval_gate.expired_approval",
                    severity=Severity.HIGH,
                    message=(f"Approval is {age}s old, outside the "
                             f"{self.max_approval_age_seconds}s window"),
                )
        expected = issue_approval_token(
            approval_action_hash(text), int(issued_at),
            evidence.get("approver") or "", self.signing_secret)
        if not hmac.compare_digest(expected, str(token)):
            # Covers both a forged MAC and a valid token replayed against a DIFFERENT
            # action, since the action digest is inside the MAC.
            return RuleHit(
                rule_id="human_approval_gate.invalid_approval_signature",
                severity=Severity.HIGH,
                message=("Approval signature does not verify for this action and approver; "
                         "it is forged, or issued for a different action"),
            )
        return None

    def detect(self, text: str,
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        categories = self._gated_categories_in(text)
        if not categories:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        evidence = self._approval_evidence(metadata)
        rule_hits: List[RuleHit] = []

        signature_hit = (self._verify_signature(text, evidence)
                         if self.require_signed_approval else None)

        # An approval that fails verification is not an approval. Folding the signature
        # result into the approved flag keeps one decision path: whatever the reason, an
        # unapproved gated action reports the same unapproved_<category> findings.
        approved = bool(evidence.get("approved")) and signature_hit is None
        if signature_hit is not None:
            rule_hits.append(signature_hit)

        if not approved:
            for name in categories:
                rule_hits.append(RuleHit(
                    rule_id=f"human_approval_gate.unapproved_{name}",
                    severity=Severity.HIGH,
                    message=f"{name.replace('_', ' ')} requires human approval and none was supplied",
                ))
        elif self.require_approver_identity and not evidence.get("approver"):
            # Approved, but by nobody nameable: accepted as a flag, rejected as evidence.
            rule_hits.append(RuleHit(
                rule_id="human_approval_gate.anonymous_approval",
                severity=Severity.MEDIUM,
                message=("Approval carries no approver identity; an unattributable approval "
                         "cannot be audited"),
            ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = (f"human_approval_gate: {', '.join(h.rule_id for h in rule_hits)}"
               f" (categories: {', '.join(categories)})")
        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="This action needs to be approved by a person before it can run.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
