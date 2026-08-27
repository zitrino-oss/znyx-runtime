"""v2 envelope signatures: the signature must bind every field the runtime
reads, not just the policy_hash. Skips cleanly on builds that predate the
v2 envelope."""
import pytest

bundle_mod = pytest.importorskip("znyx_core.policy.bundle")

if not hasattr(bundle_mod, "sign_bundle_v2"):
    pytest.skip("v2 envelope signatures not present in this build",
                allow_module_level=True)

from znyx_core.policy.bundle import (  # noqa: E402
    PolicyBundle,
    build_bundle,
    canonical_signing_payload_v2,
    resign_bundle,
    sign_bundle_v2,
    signing_envelope_v2,
    validate_bundle,
    verify_bundle,
    verify_bundle_v2,
)

POLICIES = {"secrets": {"enabled": True}, "toxicity": {"enabled": True, "action": "WARN"}}


def _signed(private_pem):
    return build_bundle(
        POLICIES,
        org_id="org-1",
        project_id="proj-1",
        environment="prod",
        bundle_id="b-1",
        private_key_pem=private_pem,
        scope_policies={"t1:a1:*:prod": {"secrets": {"enabled": True}}},
    )


def _copy(bundle):
    return PolicyBundle.from_dict(bundle.to_dict())


class TestV2Signing:
    def test_signing_sets_both_signatures(self, ed25519_keys):
        private_pem, public_pem = ed25519_keys
        bundle = _signed(private_pem)
        assert bundle.signature and bundle.signature_v2
        assert verify_bundle(bundle, public_pem)
        assert verify_bundle_v2(bundle, public_pem)
        assert validate_bundle(bundle, public_key_pem=public_pem,
                               require_signature=True, require_signature_v2=True)

    def test_round_trip_preserves_v2_signature(self, ed25519_keys):
        private_pem, public_pem = ed25519_keys
        restored = PolicyBundle.from_json(_signed(private_pem).to_json())
        assert verify_bundle_v2(restored, public_pem)

    def test_envelope_excludes_derived_values(self, ed25519_keys):
        private_pem, _ = ed25519_keys
        envelope = signing_envelope_v2(_signed(private_pem))
        assert "policy_hash" not in envelope
        assert "signature" not in envelope
        assert "signature_v2" not in envelope

    def test_canonical_payload_is_deterministic(self, ed25519_keys):
        private_pem, _ = ed25519_keys
        bundle = _signed(private_pem)
        assert canonical_signing_payload_v2(bundle) == canonical_signing_payload_v2(_copy(bundle))


# One entry per field the v2 envelope covers: mutating any of them after
# signing must invalidate the v2 signature.
MUTATIONS = [
    ("version", 0),
    ("bundle_id", "spoofed-bundle"),
    ("org_id", "attacker-org"),
    ("project_id", "other-project"),
    ("environment", "dev"),
    ("published_at", "2020-01-01T00:00:00+00:00"),
    ("scope", {"tenant_id": "victim", "app_id": "", "agent_id": "", "env": ""}),
    ("policies", {"secrets": {"enabled": False}}),
    ("scope_policies", {"t1:a1:*:prod": {"secrets": {"enabled": False}}}),
]


class TestV2MutationMatrix:
    @pytest.mark.parametrize("field,value", MUTATIONS, ids=[f for f, _ in MUTATIONS])
    def test_mutating_any_signed_field_breaks_verification(self, ed25519_keys, field, value):
        private_pem, public_pem = ed25519_keys
        bundle = _copy(_signed(private_pem))
        assert getattr(bundle, field) != value
        setattr(bundle, field, value)
        assert not verify_bundle_v2(bundle, public_pem)
        assert not validate_bundle(bundle, public_key_pem=public_pem,
                                   require_signature=True, require_signature_v2=True)

    def test_legacy_signature_misses_identity_tampering(self, ed25519_keys):
        # The motivation for v2: the legacy signature covers only policy_hash,
        # so identity metadata can be swapped without breaking it.
        private_pem, public_pem = ed25519_keys
        bundle = _copy(_signed(private_pem))
        bundle.org_id = "attacker-org"
        assert verify_bundle(bundle, public_pem)
        assert not verify_bundle_v2(bundle, public_pem)


class TestV2Validation:
    def test_domain_separation_from_legacy_signature(self, ed25519_keys):
        # A legacy signature can never be replayed as a v2 signature.
        private_pem, public_pem = ed25519_keys
        bundle = _signed(private_pem)
        assert bundle.signature != bundle.signature_v2
        forged = _copy(bundle)
        forged.signature_v2 = forged.signature
        assert not verify_bundle_v2(forged, public_pem)

    def test_tampered_v2_signature_fails_even_when_optional(self, ed25519_keys):
        # A present-but-invalid v2 signature must fail validation even when
        # the caller did not require v2 (opportunistic verification).
        private_pem, public_pem = ed25519_keys
        bundle = _signed(private_pem)
        bundle.signature_v2 = "AAAA" + bundle.signature_v2[4:]
        assert not validate_bundle(bundle, public_key_pem=public_pem,
                                   require_signature=True, require_signature_v2=False)

    def test_legacy_only_bundle_stays_valid_unless_v2_required(self, ed25519_keys):
        # Back-compat: bundles from control planes that predate the v2
        # envelope carry only the legacy signature.
        private_pem, public_pem = ed25519_keys
        bundle = _signed(private_pem)
        bundle.signature_v2 = None
        assert validate_bundle(bundle, public_key_pem=public_pem, require_signature=True)
        assert not validate_bundle(bundle, public_key_pem=public_pem,
                                   require_signature=True, require_signature_v2=True)

    def test_older_payload_without_v2_field_parses(self, ed25519_keys):
        private_pem, public_pem = ed25519_keys
        data = _signed(private_pem).to_dict()
        data.pop("signature_v2")
        restored = PolicyBundle.from_dict(data)
        assert restored.signature_v2 is None
        assert validate_bundle(restored, public_key_pem=public_pem, require_signature=True)

    def test_missing_v2_signature_never_verifies(self, ed25519_keys):
        _, public_pem = ed25519_keys
        assert not verify_bundle_v2(PolicyBundle(policies=POLICIES), public_pem)


class TestResign:
    def test_resign_after_mutation_restores_verification(self, ed25519_keys):
        private_pem, public_pem = ed25519_keys
        bundle = _signed(private_pem)
        bundle.policies = {"secrets": {"enabled": False}}
        assert not verify_bundle_v2(bundle, public_pem)
        resign_bundle(bundle, private_pem)
        assert bundle.policy_hash == bundle.compute_hash()
        assert verify_bundle(bundle, public_pem)
        assert verify_bundle_v2(bundle, public_pem)

    def test_resign_without_key_clears_stale_v2_signature(self, ed25519_keys):
        private_pem, _ = ed25519_keys
        bundle = _signed(private_pem)
        bundle.policies = {"secrets": {"enabled": False}}
        resign_bundle(bundle, None)
        assert bundle.signature_v2 is None
        assert bundle.policy_hash == bundle.compute_hash()

    def test_sign_bundle_v2_is_deterministic_for_same_content(self, ed25519_keys):
        # Ed25519 is deterministic, so equal envelopes sign identically - a
        # cheap check that the canonical payload has no unstable inputs.
        private_pem, _ = ed25519_keys
        bundle = _signed(private_pem)
        assert sign_bundle_v2(bundle, private_pem) == sign_bundle_v2(_copy(bundle), private_pem)
