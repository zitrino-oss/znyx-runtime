"""
Policy Bundle format for distributing signed policy configurations.

A bundle is a self-contained JSON document that the runtime caches locally.
The control plane publishes bundles; the runtime fetches and verifies them.
"""
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Bundle format version
BUNDLE_FORMAT_VERSION = 1


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
    bundle.policy_hash = bundle.compute_hash()

    if private_key_pem:
        bundle.signature = sign_bundle(bundle, private_key_pem)

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


def validate_bundle(bundle: PolicyBundle, public_key_pem: Optional[str] = None,
                    require_signature: bool = False) -> bool:
    """
    Validate a bundle's integrity and optionally its signature.

    Args:
        bundle: The bundle to validate
        public_key_pem: Ed25519 public key for signature verification
        require_signature: If True, reject unsigned bundles
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
    if require_signature:
        if not bundle.signature:
            logger.error("Bundle signature required but not present")
            return False
        if not public_key_pem:
            logger.error("Public key required for signature verification")
            return False
        if not verify_bundle(bundle, public_key_pem):
            logger.error("Bundle signature verification failed")
            return False

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
    """Save a bundle to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(bundle.to_json())
    logger.info(f"Bundle saved to {path}")


def _generate_bundle_id() -> str:
    import uuid
    return str(uuid.uuid4())
