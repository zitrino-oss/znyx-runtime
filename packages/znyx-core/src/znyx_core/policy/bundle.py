"""
Policy Bundle format for distributing signed policy configurations.

A bundle is a self-contained JSON document that the runtime caches locally.
The control plane publishes bundles; the runtime fetches and verifies them.
"""
import json
import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Bundle format version
BUNDLE_FORMAT_VERSION = 1

# Domain-separation prefix for the v2 envelope signature. Legacy signatures
# cover the policy_hash string, which always starts with "sha256:", so the two
# message spaces can never collide even under the same key.
BUNDLE_SIG_V2_DOMAIN = "znyx-policy-bundle-sig-v2\x00"


def _scope_key(tenant_id: str = "", app_id: str = "", agent_id: str = "", env: str = "") -> str:
    """Build a canonical scope lookup key."""
    return f"{tenant_id or '*'}:{app_id or '*'}:{agent_id or '*'}:{env or '*'}"


@dataclass
class PolicyBundle:
    """A signed, versioned policy bundle.

    Supports two modes:
    - **Single-scope**: ``policies`` holds one flat policy dict (local YAML mode).
    - **Multi-scope**: ``scope_policies`` maps scope keys to policy dicts,
      allowing a shared runtime to serve many tenant/app combinations from
      one bundle.

    ``scope_policies`` keys use the format ``tenant:app:agent:env`` where
    ``*`` means "any".  Lookup order: exact → wildcard agent → wildcard
    app+agent → default.
    """
    version: int = BUNDLE_FORMAT_VERSION
    bundle_id: str = ""
    org_id: str = ""
    project_id: str = ""
    environment: str = "prod"
    published_at: str = ""
    policies: Dict[str, Any] = field(default_factory=dict)
    policy_hash: str = ""
    signature: Optional[str] = None  # base64-encoded Ed25519 signature
    scope: Dict[str, str] = field(default_factory=lambda: {
        "tenant_id": "",
        "app_id": "",
        "agent_id": "",
        "env": "",
    })
    # Multi-scope policy graph: scope_key -> policy dict
    scope_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # base64-encoded Ed25519 signature over the v2 signing envelope, which
    # also binds identity metadata; None on bundles from older control planes
    signature_v2: Optional[str] = None

    def resolve_scope(self, tenant_id: str = "default", app_id: str = "default",
                      agent_id: str = "default", env: str = "prod") -> Dict[str, Any]:
        """Look up the best-matching policy for the given scope.

        Falls back through progressively broader wildcards, then to the
        single-scope ``policies`` dict.
        """
        if not self.scope_policies:
            return self.policies

        # Try exact → default agent → wildcard agent → wildcard app+agent → wildcard tenant → default
        candidates = [
            _scope_key(tenant_id, app_id, agent_id, env),
            _scope_key(tenant_id, app_id, "default", env),
            _scope_key(tenant_id, app_id, "*", env),
            _scope_key(tenant_id, "*", "*", env),
            _scope_key(tenant_id, "*", "*", "*"),
            _scope_key("*", "*", "*", "*"),
        ]
        for key in candidates:
            if key in self.scope_policies:
                return self.scope_policies[key]

        # Final fallback: single-scope policies (local YAML mode)
        return self.policies

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the policies dict and any per-agent scope overrides."""
        payload = json.dumps(
            {"policies": self.policies, "scope_policies": self.scope_policies},
            sort_keys=True, separators=(',', ':'),
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyBundle":
        return cls(
            version=data.get("version", BUNDLE_FORMAT_VERSION),
            bundle_id=data.get("bundle_id", ""),
            org_id=data.get("org_id", ""),
            project_id=data.get("project_id", ""),
            environment=data.get("environment", "prod"),
            published_at=data.get("published_at", ""),
            policies=data.get("policies", {}),
            policy_hash=data.get("policy_hash", ""),
            signature=data.get("signature"),
            scope=data.get("scope", {}),
            scope_policies=data.get("scope_policies", {}),
            signature_v2=data.get("signature_v2"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PolicyBundle":
        return cls.from_dict(json.loads(json_str))


def build_bundle(
    policies: Dict[str, Any],
    org_id: str,
    project_id: str,
    environment: str,
    bundle_id: str = "",
    private_key_pem: Optional[str] = None,
    scope_policies: Optional[Dict[str, Dict[str, Any]]] = None,
) -> PolicyBundle:
    """
    Build a policy bundle from a resolved policy dict.

    Args:
        policies: The resolved policy configuration (env-level default)
        org_id: Organization identifier
        project_id: Project identifier
        environment: Environment name (dev/staging/prod)
        bundle_id: Unique bundle identifier
        private_key_pem: Optional Ed25519 private key PEM for signing
        scope_policies: Optional map of ``tenant:app:agent:env`` -> policy dict
                        for per-agent overrides.  Included in the hash/signature.
    """
    bundle = PolicyBundle(
        bundle_id=bundle_id or _generate_bundle_id(),
        org_id=org_id,
        project_id=project_id,
        environment=environment,
        published_at=datetime.now(timezone.utc).isoformat(),
        policies=policies,
        scope_policies=scope_policies or {},
    )
    resign_bundle(bundle, private_key_pem)

    return bundle


def sign_bundle(bundle: PolicyBundle, private_key_pem: str) -> str:
    """Sign the bundle's policy_hash with an Ed25519 private key."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        import base64

        normalized_pem = private_key_pem.replace("\\n", "\n").strip()
        key = load_pem_private_key(normalized_pem.encode(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("Key must be Ed25519")

        payload = bundle.policy_hash.encode()
        sig = key.sign(payload)
        return base64.b64encode(sig).decode()
    except ImportError:
        logger.warning("cryptography library not installed; bundle signing unavailable")
        return ""


def verify_bundle(bundle: PolicyBundle, public_key_pem: str) -> bool:
    """Verify the bundle's signature against an Ed25519 public key."""
    if not bundle.signature:
        return False

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        import base64

        normalized_pem = public_key_pem.replace("\\n", "\n").strip()
        key = load_pem_public_key(normalized_pem.encode())
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("Key must be Ed25519")

        sig = base64.b64decode(bundle.signature)
        payload = bundle.policy_hash.encode()
        key.verify(sig, payload)
        return True
    except ImportError:
        logger.warning("cryptography library not installed; bundle verification unavailable")
        return False
    except Exception:
        logger.warning("Bundle signature verification failed")
        return False


def signing_envelope_v2(bundle: PolicyBundle) -> Dict[str, Any]:
    """The full set of fields covered by the v2 signature.

    The legacy scheme signs only the policy_hash, leaving identity and
    resolution metadata (bundle_id, org, project, environment, scope) open to
    tampering. The v2 envelope binds every field the runtime reads when
    validating and resolving policies. policy_hash and the signatures are
    derived values and deliberately excluded.
    """
    return {
        "bundle_format_version": bundle.version,
        "bundle_id": bundle.bundle_id or "",
        "org_id": bundle.org_id or "",
        "project_id": bundle.project_id or "",
        "environment": bundle.environment or "",
        "published_at": bundle.published_at or "",
        "scope": bundle.scope or {},
        "policies": bundle.policies or {},
        "scope_policies": bundle.scope_policies or {},
    }


def canonical_signing_payload_v2(bundle: PolicyBundle) -> bytes:
    """Deterministic byte encoding of the v2 envelope.

    Signer and verifier must produce identical bytes, so the JSON options are
    pinned: recursive key sorting, compact separators, ASCII-only output.
    """
    envelope = signing_envelope_v2(bundle)
    payload = BUNDLE_SIG_V2_DOMAIN + json.dumps(
        envelope, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    )
    return payload.encode("utf-8")


def sign_bundle_v2(bundle: PolicyBundle, private_key_pem: str) -> str:
    """Sign the bundle's v2 envelope with an Ed25519 private key."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        import base64

        normalized_pem = private_key_pem.replace("\\n", "\n").strip()
        key = load_pem_private_key(normalized_pem.encode(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("Key must be Ed25519")

        sig = key.sign(canonical_signing_payload_v2(bundle))
        return base64.b64encode(sig).decode()
    except ImportError:
        logger.warning("cryptography library not installed; bundle signing unavailable")
        return ""


def verify_bundle_v2(bundle: PolicyBundle, public_key_pem: str) -> bool:
    """Verify the bundle's v2 envelope signature against an Ed25519 public key."""
    if not bundle.signature_v2:
        return False

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        import base64

        normalized_pem = public_key_pem.replace("\\n", "\n").strip()
        key = load_pem_public_key(normalized_pem.encode())
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("Key must be Ed25519")

        sig = base64.b64decode(bundle.signature_v2)
        key.verify(sig, canonical_signing_payload_v2(bundle))
        return True
    except ImportError:
        logger.warning("cryptography library not installed; bundle verification unavailable")
        return False
    except Exception:
        logger.warning("Bundle v2 envelope signature verification failed")
        return False


def resign_bundle(bundle: PolicyBundle, private_key_pem: Optional[str]) -> None:
    """Recompute the bundle's hash and signatures after its content changed.

    Call this whenever policies, scope keys, or identity fields are mutated,
    e.g. secret resolution or serve-time identity overrides. Verifiers must
    always check the bundle as served, never an earlier stored form. Without a
    key the v2 signature is cleared - a stale one would be guaranteed to fail
    verification - while the legacy signature is left as-is to match the
    unsigned path's existing behavior.
    """
    bundle.policy_hash = bundle.compute_hash()
    if private_key_pem:
        bundle.signature = sign_bundle(bundle, private_key_pem)
        bundle.signature_v2 = sign_bundle_v2(bundle, private_key_pem)
    else:
        bundle.signature_v2 = None


def validate_bundle(bundle: PolicyBundle, public_key_pem: Optional[str] = None,
                    require_signature: bool = False,
                    require_signature_v2: bool = False) -> bool:
    """
    Validate a bundle's integrity and optionally its signature.

    Args:
        bundle: The bundle to validate
        public_key_pem: Ed25519 public key for signature verification
        require_signature: If True, reject unsigned bundles
        require_signature_v2: If True, reject bundles that lack the v2
            envelope signature. Implies require_signature - asking for the
            stronger check must not silently do nothing when the caller forgot
            to also pass require_signature=True.
    """
    # Check format version
    if bundle.version != BUNDLE_FORMAT_VERSION:
        logger.error(f"Unsupported bundle version: {bundle.version}")
        return False

    # Verify hash integrity
    expected_hash = bundle.compute_hash()
    if bundle.policy_hash and bundle.policy_hash != expected_hash:
        logger.error("Bundle policy_hash mismatch - content may be tampered")
        return False

    # Verify signature if required
    if require_signature or require_signature_v2:
        if not bundle.signature:
            logger.error("Bundle signature required but not present")
            return False
        if not public_key_pem:
            logger.error("Public key required for signature verification")
            return False
        if not verify_bundle(bundle, public_key_pem):
            logger.error("Bundle signature verification failed")
            return False
        if bundle.signature_v2:
            if not verify_bundle_v2(bundle, public_key_pem):
                logger.error("Bundle v2 envelope signature verification failed")
                return False
        elif require_signature_v2:
            logger.error("Bundle v2 envelope signature required but not present")
            return False
        else:
            logger.warning(
                "Bundle carries only the legacy policy_hash signature; identity "
                "metadata is not signature-covered (control plane predates the "
                "v2 envelope)"
            )

    return True


def load_bundle_from_file(path: str) -> PolicyBundle:
    """Load a bundle from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Support both bundle format and plain YAML-exported policy
    if "policies" in data and "version" in data:
        return PolicyBundle.from_dict(data)

    # Treat as a plain policy dict (e.g., from YAML export)
    bundle = PolicyBundle(policies=data)
    bundle.policy_hash = bundle.compute_hash()
    return bundle


def save_bundle_to_file(bundle: PolicyBundle, path: str) -> None:
    """Save a bundle to a JSON file, atomically.

    Writes the new content to a temp file in the same directory, then swaps it into
    place with ``os.replace()`` (atomic on POSIX) -- a process crash mid-write can
    corrupt an in-place write, but can never corrupt ``path`` itself this way, since
    ``path`` is never opened for writing directly. The previous generation, if any, is
    preserved alongside as ``<path>.bak`` before the swap, so a caller (e.g. a boot-time
    fallback that finds ``path`` unreadable/invalid right after it was cached) has a
    known-good prior copy to recover from.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the previous generation before touching `path` at all, so a failure
    # anywhere below (including this copy) never leaves `path` itself in a bad state.
    if target.exists():
        try:
            shutil.copyfile(target, f"{path}.bak")
        except OSError as e:
            logger.warning(f"Could not preserve previous bundle generation at {path}.bak: {e}")

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(bundle.to_json())
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the file 0600 (owner-only); match the permissions a plain
        # open(path, 'w') would have produced so a swap-in doesn't unexpectedly tighten
        # access to the cache file for callers that relied on the old mode.
        try:
            os.chmod(tmp_path, 0o644)
        except OSError:
            pass
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    logger.info(f"Bundle saved to {path}")


def _generate_bundle_id() -> str:
    import uuid
    return str(uuid.uuid4())
