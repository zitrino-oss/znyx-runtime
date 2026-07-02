"""Tamper-evident signing for the `_scorecard_gate` enforcement stamp (console-less tier).

In the managed/console tier the stamp is already protected: `stamp_policy_gates` mutates
`_scorecard_gate` into the policy BEFORE the bundle is hashed + Ed25519-signed, so any edit
invalidates the bundle signature. The console-LESS YAML tier has no such signature — the
runtime would trust a hand-written `enforcement_passed: true`. This module lets the offline
scorecard CLI sign the stamp and the runtime verify it, closing that gap.

Design:
  * The signed payload binds the verdict to its identity: detector + model_version +
    enforcement_passed + validated_at (canonical, sorted, compact JSON). So a stamp can't be
    copied onto a different detector/model or have its boolean flipped without detection.
  * Ed25519, reusing the same key handling as bundle signing (znyx_core.policy.bundle).
  * OPT-IN + backward-compatible: verification only bites when a verification key is
    configured (ZNYX_SCORECARD_PUBLIC_KEY). With no key, an unsigned stamp is trusted exactly
    as before — so existing YAML and managed bundles are unaffected.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Fields of the stamp that are bound by the signature (everything but `sig` itself).
_SIGNED_KEYS = ("enforcement_passed", "model_version", "validated_at")


def _canonical_payload(detector_id: str, stamp: Dict[str, Any]) -> bytes:
    """Deterministic bytes the signature covers: detector identity + the signed stamp fields.
    sort_keys + compact separators so the producer and verifier agree byte-for-byte."""
    body = {
        "detector": detector_id,
        "enforcement_passed": bool(stamp.get("enforcement_passed")),
        "model_version": stamp.get("model_version"),
        "validated_at": stamp.get("validated_at"),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _load_private_key(private_key_pem: str):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    key = load_pem_private_key(private_key_pem.replace("\\n", "\n").strip().encode(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("scorecard signing key must be Ed25519")
    return key


def _load_public_key(public_key_pem: str):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    key = load_pem_public_key(public_key_pem.replace("\\n", "\n").strip().encode())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("scorecard verification key must be Ed25519")
    return key


def sign_stamp(detector_id: str, *, enforcement_passed: bool, model_version: str,
               validated_at: str, private_key_pem: str) -> Dict[str, Any]:
    """Build a signed `_scorecard_gate` stamp. Raises if the key can't be loaded (the caller
    asked to sign, so a bad key is a hard error, not a silent unsigned stamp)."""
    stamp = {
        "enforcement_passed": bool(enforcement_passed),
        "model_version": model_version,
        "validated_at": validated_at,
    }
    key = _load_private_key(private_key_pem)
    stamp["sig"] = base64.b64encode(key.sign(_canonical_payload(detector_id, stamp))).decode()
    return stamp


def verify_stamp(detector_id: str, stamp: Any, public_key_pem: str) -> bool:
    """True iff ``stamp`` carries a valid Ed25519 signature for ``detector_id``. Any problem
    (no sig, wrong type, bad key, mismatched payload) → False (fail closed). Never raises."""
    if not isinstance(stamp, dict) or not stamp.get("sig"):
        return False
    try:
        key = _load_public_key(public_key_pem)
        sig = base64.b64decode(stamp["sig"])
        key.verify(sig, _canonical_payload(detector_id, stamp))
        return True
    except Exception:
        logger.warning("scorecard stamp signature verification failed for detector %r", detector_id)
        return False


def resolve_verification_key(explicit: Optional[str] = None) -> Optional[str]:
    """The Ed25519 public key to verify stamps against, or None (→ verification off / trust
    mode). Dedicated env var so enabling stamp verification is an explicit opt-in and does NOT
    piggyback on ZNYX_BUNDLE_PUBLIC_KEY (managed-tier stamps are unsigned by design — they're
    covered by the bundle signature instead — so reusing that key would wrongly reject them)."""
    if explicit:
        return explicit
    import os
    return os.getenv("ZNYX_SCORECARD_PUBLIC_KEY") or None
