"""Policy loading, PolicyBundle round-trip, and legacy signature verification."""
import pytest

from znyx_core.policy.bundle import (
    BUNDLE_FORMAT_VERSION,
    PolicyBundle,
    build_bundle,
    load_bundle_from_file,
    save_bundle_to_file,
    sign_bundle,
    validate_bundle,
    verify_bundle,
)
from znyx_core.policy.loader import PolicyLoader

POLICIES = {"secrets": {"enabled": True}, "pii": {"enabled": True, "action": "REDACT"}}


class TestPolicyLoader:
    def test_loads_default_policy(self, policy_file):
        loader = PolicyLoader(str(policy_file))
        default = loader.get_default_policy()
        assert default["secrets"]["enabled"] is True
        assert default["jailbreak"]["threshold"] == 60

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PolicyLoader(str(tmp_path / "nope.yaml"))

    def test_unknown_tenant_falls_back_to_empty(self, policy_file):
        loader = PolicyLoader(str(policy_file))
        assert loader.get_tenant_policy("no-such-tenant") == {}

    def test_reload_picks_up_changes(self, policy_file):
        loader = PolicyLoader(str(policy_file))
        policy_file.write_text("default:\n  secrets:\n    enabled: false\n")
        loader.reload()
        assert loader.get_default_policy()["secrets"]["enabled"] is False


class TestBundleRoundTrip:
    def test_json_round_trip_preserves_fields(self):
        bundle = build_bundle(POLICIES, org_id="org-1", project_id="proj-1",
                              environment="prod", bundle_id="b-1")
        restored = PolicyBundle.from_json(bundle.to_json())
        assert restored.bundle_id == "b-1"
        assert restored.org_id == "org-1"
        assert restored.project_id == "proj-1"
        assert restored.environment == "prod"
        assert restored.policies == POLICIES
        assert restored.policy_hash == bundle.policy_hash
        assert restored.compute_hash() == restored.policy_hash

    def test_file_round_trip(self, tmp_path):
        bundle = build_bundle(POLICIES, org_id="o", project_id="p", environment="dev")
        path = tmp_path / "bundle.json"
        save_bundle_to_file(bundle, str(path))
        restored = load_bundle_from_file(str(path))
        assert restored.policies == POLICIES
        assert validate_bundle(restored)

    def test_plain_policy_json_is_wrapped(self, tmp_path):
        # A bare policy dict (YAML export) loads as a single-scope bundle.
        path = tmp_path / "plain.json"
        path.write_text('{"secrets": {"enabled": true}}')
        restored = load_bundle_from_file(str(path))
        assert restored.policies == {"secrets": {"enabled": True}}
        assert restored.policy_hash == restored.compute_hash()


class TestScopeResolution:
    def _bundle(self):
        return PolicyBundle(
            policies={"tier": "fallback"},
            scope_policies={
                "t1:a1:agent1:prod": {"tier": "exact"},
                "t1:a1:*:prod": {"tier": "wildcard-agent"},
                "t1:*:*:*": {"tier": "wildcard-app"},
            },
        )

    def test_exact_match_wins(self):
        policy = self._bundle().resolve_scope("t1", "a1", "agent1", "prod")
        assert policy["tier"] == "exact"

    def test_falls_back_through_wildcards(self):
        bundle = self._bundle()
        assert bundle.resolve_scope("t1", "a1", "other", "prod")["tier"] == "wildcard-agent"
        assert bundle.resolve_scope("t1", "a2", "x", "dev")["tier"] == "wildcard-app"

    def test_no_scope_match_uses_single_scope_policies(self):
        assert self._bundle().resolve_scope("t9", "a9", "x", "prod")["tier"] == "fallback"

    def test_empty_scope_policies_returns_policies(self):
        bundle = PolicyBundle(policies={"tier": "only"})
        assert bundle.resolve_scope()["tier"] == "only"


class TestLegacySignature:
    def _signed(self, private_pem):
        return build_bundle(POLICIES, org_id="org-1", project_id="proj-1",
                            environment="prod", private_key_pem=private_pem)

    def test_signed_bundle_verifies(self, ed25519_keys):
        private_pem, public_pem = ed25519_keys
        bundle = self._signed(private_pem)
        assert bundle.signature
        assert verify_bundle(bundle, public_pem)
        assert validate_bundle(bundle, public_key_pem=public_pem, require_signature=True)

    def test_unsigned_bundle_does_not_verify(self, ed25519_keys):
        _, public_pem = ed25519_keys
        bundle = build_bundle(POLICIES, org_id="o", project_id="p", environment="prod")
        assert bundle.signature is None
        assert not verify_bundle(bundle, public_pem)
        assert not validate_bundle(bundle, public_key_pem=public_pem, require_signature=True)

    def test_wrong_key_fails(self, ed25519_keys):
        private_pem, _ = ed25519_keys
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        other_pub = Ed25519PrivateKey.generate().public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        assert not verify_bundle(self._signed(private_pem), other_pub)

    def test_tampered_policies_fail_hash_check(self, ed25519_keys):
        private_pem, public_pem = ed25519_keys
        bundle = self._signed(private_pem)
        bundle.policies["secrets"]["enabled"] = False
        assert not validate_bundle(bundle, public_key_pem=public_pem, require_signature=True)

    def test_tampered_signature_fails(self, ed25519_keys):
        private_pem, public_pem = ed25519_keys
        bundle = self._signed(private_pem)
        bundle.signature = "AAAA" + bundle.signature[4:]
        assert not verify_bundle(bundle, public_pem)

    def test_require_signature_needs_a_public_key(self, ed25519_keys):
        private_pem, _ = ed25519_keys
        bundle = self._signed(private_pem)
        assert not validate_bundle(bundle, public_key_pem=None, require_signature=True)

    def test_unsupported_format_version_rejected(self):
        bundle = build_bundle(POLICIES, org_id="o", project_id="p", environment="prod")
        bundle.version = BUNDLE_FORMAT_VERSION + 1
        assert not validate_bundle(bundle)

    def test_sign_covers_only_the_policy_hash(self, ed25519_keys):
        # Documents the legacy contract: the signature is over policy_hash, so
        # re-signing an identical hash yields a verifiable signature.
        private_pem, public_pem = ed25519_keys
        bundle = self._signed(private_pem)
        again = sign_bundle(bundle, private_pem)
        bundle.signature = again
        assert verify_bundle(bundle, public_pem)
