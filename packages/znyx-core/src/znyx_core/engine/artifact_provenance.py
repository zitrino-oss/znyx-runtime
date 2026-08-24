"""Model-artifact provenance and integrity (OWASP LLM04 + LLM05).

Two lifecycle controls that share one artifact and one moment, so they share a module.

**LLM04 provenance, at the promotion boundary.** Digest pinning already proves an
artifact did not change between pinning and loading. It says nothing about where the
artifact came from. 2026's supply-chain entry asks for the other half: a signature that
attests origin, and an AIBOM that records what the thing is made of. The promotion
boundary is the right place to demand it, because it is the last moment at which
declining is cheap.

Signing proves ORIGIN, not safety. A correctly signed backdoored model verifies. That is
why this pairs with the scorecard gate rather than replacing it: provenance answers "do I
know who made this", the scorecard answers "does it behave".

**LLM05 artifact integrity, at load.** 2026 calls out inference-time artifacts
explicitly, and they are the ones nobody watches: the chat template, the tokenizer
config, and any LoRA adapter travel with a model and change its behaviour without
changing a single weight. A template edited to append an instruction to every system
turn is a persistent, invisible prompt injection that survives every request-time
control, because by the time the detectors see the request the template has already been
applied. Diffing a reported artifact manifest against the recorded baseline is what
catches it.

Both are pure functions over data the control plane supplies. Nothing here touches a
network or a filesystem: the caller reports what it has, and this decides.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Artifacts that change model BEHAVIOUR without changing the weights. Anything matching
# one of these is high severity when it drifts; everything else in the manifest is
# reported at medium, because a changed README is noise and a changed template is not.
_BEHAVIOURAL_ARTIFACTS = (
    "chat_template",       # chat_template.jinja / .json - rewrites every turn
    "tokenizer_config",    # special tokens, added_tokens, template fallbacks
    "tokenizer.json",
    "special_tokens_map",
    "generation_config",   # sampling defaults, stop tokens, forced ids
    "adapter_config",      # LoRA: which layers, what rank
    "adapter_model",       # LoRA: the delta weights themselves
)

# AIBOM: an SBOM for a model. No single standard has won, so this validates the shape
# every candidate agrees on rather than pinning one format.
_AIBOM_REQUIRED = ("model_id", "components")


def is_behavioural_artifact(path: str) -> bool:
    """True when this file changes what the model does, not merely what it says about
    itself. Matched on the basename stem so a nested path still resolves."""
    name = (path or "").rsplit("/", 1)[-1].lower()
    return any(marker in name for marker in _BEHAVIOURAL_ARTIFACTS)


# ── provenance (LLM04) ───────────────────────────────────────────────────────

def canonical_artifact_payload(model_id: str, revision: str, sha256: str) -> bytes:
    """The bytes a provenance signature covers.

    Binds the signature to the exact artifact IDENTITY, so a valid signature cannot be
    lifted onto a different model, a different revision, or a different digest. Sorted
    compact JSON so signer and verifier agree byte for byte, matching
    ``scorecard_stamp._canonical_payload``."""
    body = {"model_id": model_id, "revision": revision, "sha256": (sha256 or "").lower()}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign_artifact(model_id: str, revision: str, sha256: str,
                  private_key_pem: str) -> str:
    """Sign an artifact identity. Raises on a bad key: the caller asked to sign, so a
    silent unsigned result would be the worst outcome."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    key = load_pem_private_key(
        private_key_pem.replace("\\n", "\n").strip().encode(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("artifact signing key must be Ed25519")
    return base64.b64encode(
        key.sign(canonical_artifact_payload(model_id, revision, sha256))).decode()


def verify_artifact_signature(model_id: str, revision: str, sha256: str,
                              signature: Optional[str],
                              public_key_pem: Optional[str]) -> bool:
    """True iff ``signature`` verifies for this exact artifact. Fails closed on anything
    unexpected and never raises: a verification helper that throws becomes a helper the
    caller wraps in a bare except, which is how fail-open happens."""
    if not signature or not public_key_pem:
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        key = load_pem_public_key(
            public_key_pem.replace("\\n", "\n").strip().encode())
        if not isinstance(key, Ed25519PublicKey):
            return False
        key.verify(base64.b64decode(signature),
                   canonical_artifact_payload(model_id, revision, sha256))
        return True
    except Exception:                                    # noqa: BLE001 - fail closed
        logger.debug("artifact signature verification failed", exc_info=True)
        return False


def aibom_digest(aibom: Any) -> Optional[str]:
    """Stable digest of an AIBOM document, so a recorded bill of materials can be shown
    not to have been edited after the fact."""
    if not isinstance(aibom, dict) or not aibom:
        return None
    return hashlib.sha256(
        json.dumps(aibom, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


def validate_aibom(aibom: Any) -> List[str]:
    """Problems with an AIBOM document, empty when it is usable.

    Deliberately shallow. The point is to reject a placeholder that exists only to
    satisfy the gate, not to referee a format war between the competing specs."""
    if aibom in (None, {}, ""):
        return ["no AIBOM recorded for this artifact"]
    if not isinstance(aibom, dict):
        return ["AIBOM is not a document"]
    problems = [f"AIBOM is missing '{k}'" for k in _AIBOM_REQUIRED if not aibom.get(k)]
    components = aibom.get("components")
    if isinstance(components, list) and not components:
        problems.append("AIBOM lists no components")
    return problems


@dataclass
class ProvenanceVerdict:
    """Whether an artifact may be promoted, and why not when it may not."""
    ok: bool
    signature_verified: bool = False
    aibom_present: bool = False
    findings: List[dict] = field(default_factory=list)

    def blockers(self) -> List[dict]:
        return [f for f in self.findings if f.get("severity") == "HIGH"]


def evaluate_provenance(*, model_id: str, revision: str, sha256: Optional[str],
                        signature: Optional[str] = None,
                        public_key_pem: Optional[str] = None,
                        aibom: Any = None,
                        require_signature: bool = True,
                        require_aibom: bool = True) -> ProvenanceVerdict:
    """Decide whether an artifact's provenance is good enough to promote (LLM04)."""
    findings: List[dict] = []

    if not sha256:
        # Everything else here is bound to the digest, so without one there is nothing
        # a signature could even be about.
        findings.append({
            "rule_id": "model_provenance.unpinned_artifact", "severity": "HIGH",
            "message": (f"{model_id}@{revision} has no artifact digest, so its origin "
                        f"cannot be attested"),
        })

    verified = verify_artifact_signature(model_id, revision, sha256 or "",
                                         signature, public_key_pem)
    if require_signature and not verified:
        if not public_key_pem:
            findings.append({
                "rule_id": "model_provenance.no_verification_key", "severity": "HIGH",
                "message": ("Signature is required but no verification key is configured; "
                            "nothing can be verified"),
            })
        elif not signature:
            findings.append({
                "rule_id": "model_provenance.unsigned_artifact", "severity": "HIGH",
                "message": f"{model_id}@{revision} carries no provenance signature",
            })
        else:
            findings.append({
                "rule_id": "model_provenance.invalid_signature", "severity": "HIGH",
                "message": (f"Signature does not verify for {model_id}@{revision}; it is "
                            f"forged, or issued for a different artifact"),
            })

    aibom_problems = validate_aibom(aibom)
    aibom_present = not aibom_problems
    if require_aibom and aibom_problems:
        findings.append({
            "rule_id": "model_provenance.missing_aibom", "severity": "HIGH",
            "message": "; ".join(aibom_problems),
        })
    elif aibom_problems and aibom not in (None, {}, ""):
        # An AIBOM was supplied and is malformed. Not a blocker when it is not required,
        # but silently accepting a broken one would make the record worthless.
        findings.append({
            "rule_id": "model_provenance.malformed_aibom", "severity": "MEDIUM",
            "message": "; ".join(aibom_problems),
        })

    return ProvenanceVerdict(
        ok=not any(f["severity"] == "HIGH" for f in findings),
        signature_verified=verified, aibom_present=aibom_present, findings=findings)


# ── artifact integrity (LLM05) ───────────────────────────────────────────────

def manifest_digest(manifest: Dict[str, str]) -> Optional[str]:
    """One digest over a whole artifact manifest, for cheap unchanged/changed checks."""
    if not isinstance(manifest, dict) or not manifest:
        return None
    normalised = {str(k): str(v).lower() for k, v in manifest.items()}
    return hashlib.sha256(
        json.dumps(normalised, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def diff_artifacts(baseline: Optional[Dict[str, str]],
                   reported: Optional[Dict[str, str]]) -> List[dict]:
    """Findings for how a reported artifact set differs from its recorded baseline.

    A changed chat template, tokenizer config, or LoRA adapter is HIGH: those change what
    the model does without changing the weights, so digest pinning on the weights alone
    misses them entirely. Everything else is MEDIUM, because a changed README is drift
    worth seeing and not worth paging anyone about.

    A REMOVED behavioural artifact is treated as seriously as a changed one: dropping a
    chat template silently falls back to a default, which is a behaviour change achieved
    by deletion."""
    if not baseline:
        return [{
            "rule_id": "inference_artifact_integrity.no_baseline", "severity": "MEDIUM",
            "message": ("No artifact baseline recorded, so drift in templates, tokenizer "
                        "config, or adapters cannot be detected"),
        }]
    reported = reported or {}
    findings: List[dict] = []

    for path, expected in sorted(baseline.items()):
        actual = reported.get(path)
        if actual is None:
            severity = "HIGH" if is_behavioural_artifact(path) else "MEDIUM"
            findings.append({
                "rule_id": "inference_artifact_integrity.artifact_removed",
                "severity": severity,
                "message": (f"{path} is in the recorded baseline and absent from the "
                            f"loaded artifacts; a dropped template falls back to a default"),
            })
        elif str(actual).lower() != str(expected).lower():
            severity = "HIGH" if is_behavioural_artifact(path) else "MEDIUM"
            findings.append({
                "rule_id": "inference_artifact_integrity.artifact_modified",
                "severity": severity,
                "message": (f"{path} does not match its baseline digest "
                            f"({str(expected)[:12]} vs {str(actual)[:12]})"),
            })

    for path in sorted(set(reported) - set(baseline)):
        severity = "HIGH" if is_behavioural_artifact(path) else "MEDIUM"
        findings.append({
            "rule_id": "inference_artifact_integrity.artifact_added",
            "severity": severity,
            "message": (f"{path} is loaded and not in the recorded baseline; an adapter "
                        f"nobody recorded is an unreviewed behaviour change"),
        })
    return findings


def artifacts_intact(baseline: Optional[Dict[str, str]],
                     reported: Optional[Dict[str, str]]) -> bool:
    """True when nothing BEHAVIOURAL has drifted. This is the condition the
    ``inference_artifact_integrity`` lifecycle credit is granted on: a changed README is
    drift worth reporting, not grounds for withdrawing the control's credit."""
    return not any(f["severity"] == "HIGH"
                   for f in diff_artifacts(baseline, reported))
