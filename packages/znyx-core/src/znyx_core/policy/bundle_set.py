"""A set of policy bundles, for one runtime that serves several projects or environments.

The control plane serves such a runtime one bundle per environment it serves. Each
member bundle is the ordinary bundle, signed on its own exactly as before (v1 and v2
signatures, see ``bundle.py``). The set is signed too, over which bundles it holds and
the project and environment each one is for, so in transit no member can be added,
dropped, swapped or relabelled. A runtime checks both, and checks that each member's
signed identity (org, project, environment) is the one the set names for it.

A runtime whose key serves one environment receives a set of one, so a single code
path serves both. A control plane that predates sets is read through ``BundleSet.single``
(one member, no set signature; the member bundle's own signatures still apply).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.policy.bundle import PolicyBundle, validate_bundle

logger = logging.getLogger(__name__)

BUNDLE_SET_FORMAT = "znyx-bundle-set"
BUNDLE_SET_VERSION = 1
# Domain separation: a set signature can never be mistaken for a bundle signature.
BUNDLE_SET_SIG_DOMAIN = "znyx-bundle-set-sig-v1\x00"


@dataclass
class BundleSetMember:
    """One environment the runtime serves, and its bundle."""
    environment_id: str
    project_id: str
    environment: str
    bundle: PolicyBundle
    project_slug: str = ""

    @property
    def key(self) -> str:
        """Stable identity of the member within its set."""
        return self.environment_id or f"{self.project_id}:{self.environment}"

    def to_dict(self) -> Dict[str, Any]:
        return {"environment_id": self.environment_id, "project_id": self.project_id,
                "project_slug": self.project_slug, "environment": self.environment,
                "bundle": self.bundle.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleSetMember":
        if not isinstance(data, dict) or not isinstance(data.get("bundle"), dict):
            raise ValueError("bundle set member without a bundle")
        return cls(environment_id=str(data.get("environment_id") or ""),
                   project_id=str(data.get("project_id") or ""),
                   project_slug=str(data.get("project_slug") or ""),
                   environment=str(data.get("environment") or ""),
                   bundle=PolicyBundle.from_dict(data["bundle"]))


@dataclass
class BundleSet:
    org_id: str
    members: List[BundleSetMember] = field(default_factory=list)
    # Environments the runtime should serve that have no published bundle yet.
    missing: List[Dict[str, str]] = field(default_factory=list)
    set_id: str = ""
    generated_at: str = ""
    signature: Optional[str] = None
    format: str = BUNDLE_SET_FORMAT
    version: int = BUNDLE_SET_VERSION
    # The key serves several environments (signed): the runtime then never answers a
    # call that names none of its members, even when only one member is left.
    serves_several: bool = False
    # Read from a control plane that serves only single bundles: one member, unsigned
    # set. Only ever set by ``single()``, never read from a response or a cache file.
    legacy: bool = False

    @classmethod
    def single(cls, bundle: PolicyBundle) -> "BundleSet":
        return cls(org_id=bundle.org_id or "", legacy=True,
                   members=[BundleSetMember(environment_id="", project_id=bundle.project_id or "",
                                            environment=bundle.environment or "", bundle=bundle)])

    def envelope(self) -> Dict[str, Any]:
        """Everything the set signature covers."""
        return {
            "format": self.format, "version": self.version, "org_id": self.org_id or "",
            "set_id": self.set_id or "", "generated_at": self.generated_at or "",
            "serves_several": bool(self.serves_several),
            "members": [{"environment_id": m.environment_id, "project_id": m.project_id,
                         "project_slug": m.project_slug, "environment": m.environment,
                         "bundle_id": m.bundle.bundle_id or "", "policy_hash": m.bundle.policy_hash or ""}
                        for m in self.members],
            "missing": [{k: str(v) for k, v in sorted(entry.items())} for entry in self.missing],
        }

    def canonical_payload(self) -> bytes:
        return (BUNDLE_SET_SIG_DOMAIN + json.dumps(self.envelope(), sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=True)).encode("utf-8")

    def content_hash(self) -> str:
        """Changes exactly when what the runtime would serve changes (the ETag): the
        members, their names, bundles and policy hashes, and the environments still
        missing."""
        payload = json.dumps({
            "serves_several": bool(self.serves_several),
            "members": sorted([m.key, m.project_id, m.project_slug, m.environment,
                               m.bundle.bundle_id or "", m.bundle.policy_hash or ""] for m in self.members),
            "missing": sorted(json.dumps(e, sort_keys=True) for e in self.missing),
        }, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "version": self.version, "org_id": self.org_id,
                "set_id": self.set_id, "generated_at": self.generated_at,
                "serves_several": bool(self.serves_several),
                "members": [m.to_dict() for m in self.members], "missing": list(self.missing),
                "content_hash": self.content_hash(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleSet":
        if not isinstance(data, dict) or data.get("format") != BUNDLE_SET_FORMAT:
            raise ValueError("not a bundle set")
        members = data.get("members")
        missing = data.get("missing") or []
        if not isinstance(members, list) or not isinstance(missing, list):
            raise ValueError("bundle set members and missing must be lists")
        return cls(org_id=str(data.get("org_id") or ""), set_id=str(data.get("set_id") or ""),
                   generated_at=str(data.get("generated_at") or ""),
                   members=[BundleSetMember.from_dict(m) for m in members],
                   missing=[{str(k): str(v) for k, v in e.items()} for e in missing if isinstance(e, dict)],
                   signature=data.get("signature"), format=str(data.get("format")),
                   version=int(data.get("version") or 0),
                   serves_several=data.get("serves_several") is True)


def _load_key(pem: str, private: bool):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    normalized = pem.replace("\\n", "\n").strip().encode()
    key = load_pem_private_key(normalized, password=None) if private else load_pem_public_key(normalized)
    expected_type = Ed25519PrivateKey if private else Ed25519PublicKey
    if not isinstance(key, expected_type):
        raise ValueError("Key must be Ed25519")
    return key


def sign_bundle_set(bundle_set: BundleSet, private_key_pem: Optional[str]) -> None:
    """Stamp ``generated_at`` (when empty) and sign the set; without a key the set is
    served unsigned and ``signature`` is cleared."""
    if not bundle_set.generated_at:
        bundle_set.generated_at = datetime.now(timezone.utc).isoformat()
    if not private_key_pem:
        bundle_set.signature = None
        return
    key = _load_key(private_key_pem, private=True)
    bundle_set.signature = base64.b64encode(key.sign(bundle_set.canonical_payload())).decode()


def verify_bundle_set(bundle_set: BundleSet, public_key_pem: str) -> bool:
    if not bundle_set.signature:
        return False
    try:
        key = _load_key(public_key_pem, private=False)
        key.verify(base64.b64decode(bundle_set.signature), bundle_set.canonical_payload())
        return True
    except Exception:  # noqa: BLE001 - any failure is "not verified"
        logger.warning("Bundle set signature verification failed")
        return False


def validate_bundle_set(bundle_set: BundleSet, public_key_pem: Optional[str] = None,
                        require_signature: bool = False,
                        require_signature_v2: bool = False) -> Tuple[bool, str]:
    """``(ok, reason)``. Every member is validated as a bundle on its own, its signed
    identity must match what the set says it is, no environment appears twice, and
    the set signature must verify when signatures are required (or whenever one is
    present and a public key is configured)."""
    if bundle_set.format != BUNDLE_SET_FORMAT or bundle_set.version != BUNDLE_SET_VERSION:
        return False, "unsupported bundle set format"
    if bundle_set.legacy and len(bundle_set.members) != 1:
        return False, "a single-bundle set must hold exactly one bundle"
    seen_keys, seen_scopes = set(), set()
    for member in bundle_set.members:
        scope = (member.project_id, member.environment)
        if member.key in seen_keys or scope in seen_scopes:
            return False, f"environment {member.environment} of project {member.project_id} appears twice"
        seen_keys.add(member.key)
        seen_scopes.add(scope)
        bundle = member.bundle
        if not validate_bundle(bundle, public_key_pem=public_key_pem, require_signature=require_signature,
                               require_signature_v2=require_signature_v2):
            return False, f"bundle for {member.environment} of project {member.project_id} failed validation"
        if not bundle_set.legacy:
            if bundle.org_id and bundle_set.org_id and bundle.org_id != bundle_set.org_id:
                return False, "a member bundle belongs to another organisation"
            if (bundle.project_id or "") != member.project_id or (bundle.environment or "") != member.environment:
                return False, (f"the bundle labelled {member.environment} of project {member.project_id} "
                               "was published for another project or environment")
    if bundle_set.legacy:
        return True, ""
    if require_signature or require_signature_v2:
        if not public_key_pem:
            return False, "a public key is required to verify the bundle set"
        if not verify_bundle_set(bundle_set, public_key_pem):
            return False, "bundle set signature missing or invalid"
    elif bundle_set.signature and public_key_pem and not verify_bundle_set(bundle_set, public_key_pem):
        return False, "bundle set signature invalid"
    return True, ""


def save_bundle_set_to_file(bundle_set: BundleSet, path: str,
                            extra: Optional[Dict[str, Any]] = None) -> None:
    """Write the set atomically (temp file, fsync, rename), as ``save_bundle_to_file``
    does for one bundle: a crash mid-write never leaves a half-written cache. ``extra``
    adds fields beside the set (``from_dict`` ignores them)."""
    import os
    import tempfile
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({**bundle_set.to_dict(), **(extra or {})}, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_bundle_set_from_file(path: str) -> BundleSet:
    with open(path, "r") as f:
        return BundleSet.from_dict(json.load(f))
