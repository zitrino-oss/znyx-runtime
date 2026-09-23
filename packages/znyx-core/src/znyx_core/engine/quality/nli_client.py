"""Inference-backed NLI scorer for groundedness.

Builds the ``nli_scorer`` callable ``(premise, hypotheses) -> list[float]`` that
``score_groundedness``/``QualityScorer`` use for entailment-based grounding. It talks to
the inference sidecar's ``POST /v1/infer/{task}`` (default task ``nli``) using the
same JSON ``{"premise", "hypothesis"}`` pair the ``NliRunner`` expects, and converts the
contract's ``risk_score`` back into an entailment probability.

**Egress:** the premise (caller-provided sources) and hypotheses (model output
claims) are content that may leave the trust boundary, so EVERY call is routed through
the shared ``prepare_and_audit_egress`` gate — the same one the model-backed escalation
and custom-webhook paths use. A co-located sidecar (``in_boundary=True``, the default) is
not a boundary crossing → called directly at full fidelity. A hosted sidecar
(``in_boundary=False``) is gated: ``no_external_calls`` / allowlist / residency can deny
it, the payload is strict-redacted before it leaves, and a fail-closed audit event is
written first — any of those failing makes the scorer raise, so groundedness degrades to
token overlap rather than leaking unaudited content.

The scorer raises on transport/contract/egress errors (it never returns a wrong-length
list), so the caller can fall back cleanly.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx

from znyx_core.engine.egress import prepare_and_audit_egress, redact_for_egress
from znyx_core.net_guard import assert_safe_egress_url

logger = logging.getLogger(__name__)


class EgressBlocked(RuntimeError):
    """The egress gate denied / could not audit the NLI call → fall back to overlap."""


@dataclass
class NliEgressPolicy:
    """The boundary controls applied to NLI sidecar calls. Defaults describe a
    co-located in-boundary sidecar (no crossing, no redaction/audit needed)."""
    in_boundary: Optional[bool] = True       # None/True → co-located; False → hosted (gated)
    no_external_calls: bool = False
    egress_allowlist: Optional[List[str]] = None
    allowed_regions: Optional[List[str]] = None
    region: Optional[str] = None
    redact_pii: bool = True                   # privacy-first when the call DOES cross
    redact_secrets: bool = True
    pii_config: Optional[Dict[str, Any]] = None
    secrets_config: Optional[Dict[str, Any]] = None
    detector_key: str = "quality:groundedness_nli"
    model_version: Optional[str] = None


class ClaimScore(float):
    """Entailment probability that also carries the full NLI label distribution.

    Subclasses ``float`` so the published ``(premise, hypotheses) -> list[float]``
    contract is unchanged - every existing consumer keeps comparing, ordering and
    ``max()``-ing these exactly as before, and a plain float from an injected test
    scorer stays valid.

    The extra labels exist because entailment alone cannot separate the two ways a
    claim fails to be entailed, and they need opposite handling:

    * ``neutral``       - the source does not speak to this claim. Unsupported, NOT false.
    * ``contradiction`` - the source states the opposite. This is the real hallucination.

    Scoring only on ``1 - entailment`` collapses those into one number, which ranks a
    truthful paraphrase (high neutral) below an outright false claim (high
    contradiction). See ``HallucinationDetector`` for the banding that uses this.

    ``labelled`` is False when the sidecar returned only ``risk_score`` and the label
    distribution had to be derived: contradiction is then unrecoverable, so a consumer
    must not read ``contradiction == 0.0`` as evidence of non-contradiction.
    """

    __slots__ = ("entailment", "contradiction", "neutral", "labelled")

    def __new__(cls, entailment: float, contradiction: float = 0.0,
                neutral: float = 0.0, labelled: bool = False) -> "ClaimScore":
        obj = super().__new__(cls, entailment)
        obj.entailment = float(entailment)
        obj.contradiction = float(contradiction)
        obj.neutral = float(neutral)
        obj.labelled = bool(labelled)
        return obj


def _clamp01(value: Any) -> Optional[float]:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def _claim_score_from_result(result: Dict[str, Any]) -> ClaimScore:
    """Full label distribution from one ``InferResult``.

    Prefers explicit ``label_scores`` (the NLI runner emits contradiction/neutral/
    entailment); otherwise derives entailment from the contract's ``risk_score``
    (``risk = 1 - entailment``) and marks the result unlabelled, since contradiction
    cannot be recovered from a single scalar."""
    label_scores = result.get("label_scores")
    if isinstance(label_scores, dict):
        # Models spell these "entailment" / "ENTAILMENT" / "Entailment".
        lower = {str(k).lower(): v for k, v in label_scores.items()}
        entail = _clamp01(lower.get("entailment"))
        if entail is not None:
            return ClaimScore(
                entail,
                contradiction=_clamp01(lower.get("contradiction")) or 0.0,
                neutral=_clamp01(lower.get("neutral")) or 0.0,
                labelled=True,
            )
    risk = _clamp01(result.get("risk_score") if result.get("risk_score") is None
                    else float(result.get("risk_score")) / 100.0)
    entail = 0.0 if risk is None else 1.0 - risk
    return ClaimScore(entail, labelled=False)


def _entailment_from_result(result: Dict[str, Any]) -> float:
    """Entailment probability from one ``InferResult``. Retained as the narrow
    entailment-only view of :func:`_claim_score_from_result`."""
    return float(_claim_score_from_result(result))


def make_inference_nli_scorer(
    endpoint_url: str,
    *,
    task: str = "nli",
    auth_type: str = "none",
    auth_header: str = "X-API-Key",
    auth_value: str = "",
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    timeout_seconds: float = 5.0,
    egress: Optional[NliEgressPolicy] = None,
    egress_sink: Optional[Callable] = None,
    request: Any = None,
) -> Callable[[str, List[str]], List[float]]:
    """Return a synchronous ``(premise, hypotheses) -> list[float]`` entailment scorer
    backed by the inference sidecar, with egress controls applied per call.

    One entailment probability is returned per hypothesis. The callable raises on
    transport / contract / egress errors (including a result-count mismatch) so
    ``score_groundedness`` degrades to token overlap rather than scoring on partial data
    or leaking unaudited content."""
    base = endpoint_url.rstrip("/")
    url = f"{base}/v1/infer/{task}"
    egress = egress or NliEgressPolicy()
    model_version = egress.model_version or (
        f"{model_id}@{revision}" if model_id and revision else model_id
    )

    headers = {"Content-Type": "application/json"}
    if auth_type == "api_key" and auth_value:
        headers[auth_header] = auth_value
    elif auth_type == "bearer" and auth_value:
        headers["Authorization"] = f"Bearer {auth_value}"

    def scorer(premise: str, hypotheses: List[str]) -> List[float]:
        if not hypotheses:
            return []
        # SSRF guard (defense in depth — the sidecar is operator infra, so private /
        # loopback is permitted, but the cloud-metadata / link-local range is blocked).
        assert_safe_egress_url(url, allow_private=True)

        # egress gate (single fail-closed authority): decision + strict redaction + audit on
        # the full payload content. In-boundary sidecar → not egress, proceeds untouched.
        joined = premise + "\n" + "\n".join(hypotheses)
        prep = prepare_and_audit_egress(
            "local_ml", joined,
            endpoint_url=url, region=egress.region, in_boundary=egress.in_boundary,
            no_external_calls=egress.no_external_calls,
            egress_allowlist=egress.egress_allowlist, allowed_regions=egress.allowed_regions,
            redact_pii=egress.redact_pii, redact_secrets=egress.redact_secrets,
            detector_key=egress.detector_key, request=request, egress_sink=egress_sink,
            model_version=model_version,
            pii_config=egress.pii_config, secrets_config=egress.secrets_config,
        )
        if not prep.proceed:
            raise EgressBlocked(prep.reason or "egress_denied")

        send_premise, send_hyps = premise, hypotheses
        # When the call actually crosses the boundary, scrub each piece individually so
        # the JSON payload that leaves is redacted (the gate above already confirmed the
        # redactors run + audited the crossing).
        if prep.decision.is_egress and (egress.redact_pii or egress.redact_secrets):
            send_premise, _ = redact_for_egress(
                premise, egress.redact_pii, egress.redact_secrets, strict=True,
                pii_config=egress.pii_config, secrets_config=egress.secrets_config)
            send_hyps = [
                redact_for_egress(h, egress.redact_pii, egress.redact_secrets, strict=True,
                                  pii_config=egress.pii_config, secrets_config=egress.secrets_config)[0]
                for h in hypotheses
            ]

        payload: Dict[str, Any] = {
            "texts": [json.dumps({"premise": send_premise, "hypothesis": h}) for h in send_hyps],
        }
        if model_id:
            payload["model_id"] = model_id
            if revision:
                payload["revision"] = revision

        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(hypotheses):
            got = len(results) if isinstance(results, list) else "n/a"
            raise ValueError(
                f"inference NLI returned {got} results for {len(hypotheses)} hypotheses")
        return [_claim_score_from_result(r) for r in results]

    return scorer


def nli_scorer_from_config(
    quality_config: Dict[str, Any],
    *,
    runtime_policy: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
    egress_sink: Optional[Callable] = None,
    request: Any = None,
) -> Optional[Callable[[str, List[str]], List[float]]]:
    """Build an NLI scorer from a policy ``quality_scoring.nli`` block, or None.

    Expected shape (all optional except ``endpoint_url`` when enabled)::

        quality_scoring:
          enabled: true
          nli:
            enabled: true
            endpoint_url: http://inference:8088
            task: nli                 # inference task name (default "nli")
            model_id: <pin>           # optional model pin asserted by the sidecar
            revision: <pin>
            in_boundary: true         # co-located sidecar (default). false → hosted/gated
            region: us                # destination region (residency check when hosted)
            egress_allowlist: [...]   # hosts allowed when hosted
            redact_before_egress: {pii: true, secrets: true}
            auth_type: none|api_key|bearer
            auth_header: X-API-Key
            auth_value: <secret>
            timeout_ms: 5000

    The boundary controls (``no_external_calls`` / ``allowed_regions``) come from the
    ``runtime_policy``; PII/secrets redaction configs from the full ``policy`` so egress
    redaction matches the org's in-pipeline redaction. Returns None (→ token-overlap
    fallback) when the block is missing, disabled, or has no ``endpoint_url``."""
    nli = quality_config.get("nli")
    if not isinstance(nli, dict) or not nli.get("enabled"):
        return None
    endpoint_url = nli.get("endpoint_url")
    if not endpoint_url:
        return None

    runtime_policy = runtime_policy or {}
    policy = policy or {}
    rbe = nli.get("redact_before_egress")
    redact_pii = rbe.get("pii", True) if isinstance(rbe, dict) else True
    redact_secrets = rbe.get("secrets", True) if isinstance(rbe, dict) else True

    model_id = nli.get("model_id")
    revision = nli.get("revision")
    egress = NliEgressPolicy(
        in_boundary=nli.get("in_boundary", True),
        no_external_calls=bool(runtime_policy.get("no_external_calls")),
        egress_allowlist=nli.get("egress_allowlist"),
        allowed_regions=runtime_policy.get("allowed_regions"),
        region=nli.get("region"),
        redact_pii=redact_pii,
        redact_secrets=redact_secrets,
        pii_config=policy.get("pii") if isinstance(policy.get("pii"), dict) else None,
        secrets_config=policy.get("secrets") if isinstance(policy.get("secrets"), dict) else None,
        model_version=(f"{model_id}@{revision}" if model_id and revision else model_id),
    )

    timeout_ms = nli.get("timeout_ms")
    timeout_seconds = (timeout_ms / 1000.0) if isinstance(timeout_ms, (int, float)) else 5.0
    return make_inference_nli_scorer(
        endpoint_url,
        task=nli.get("task", "nli"),
        auth_type=nli.get("auth_type", "none"),
        auth_header=nli.get("auth_header", "X-API-Key"),
        auth_value=nli.get("auth_value", ""),
        model_id=model_id,
        revision=revision,
        timeout_seconds=timeout_seconds,
        egress=egress,
        egress_sink=egress_sink,
        request=request,
    )
