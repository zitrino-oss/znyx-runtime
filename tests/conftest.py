"""Shared fixtures for the package test suite.

The suite runs against znyx-core, znyx-runtime, and znyx-inference exactly as
installed - no network, no ML dependencies, all fixtures inline.
"""
import os

# An unconfigured environment is treated as production (secure by default),
# which turns on gates the tests do not exercise (mandatory auth, the
# Redis-only rate limiter). Mark the process as non-production before any
# znyx module is imported; conftest runs first, so this covers every test.
os.environ.setdefault("ZNYX_ENVIRONMENT", "development")

# Governance audit records (Phase 5 part 4) go to a spool under the home directory by
# default; keep the suite's records out of it.
import tempfile

os.environ.setdefault("ZNYX_GOVERNANCE_SPOOL_PATH",
                      os.path.join(tempfile.mkdtemp(prefix="znyx-test-"), "governance-audit.spool"))

import pytest


@pytest.fixture(autouse=True)
def _fresh_circuit_breakers():
    """Remote-detector circuit breakers are shared per endpoint for the life of
    the process; start every test with none, so one test's open circuit cannot
    leak into the next."""
    from znyx_core.detectors.remote import reset_circuit_breakers
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


@pytest.fixture(autouse=True)
def _fresh_state_store():
    """Rate limits, budgets and history live in a process-wide state store
    (znyx_core.state_store); give every test an empty one."""
    from znyx_core.state_store import MemoryStateStore, set_state_store
    set_state_store(MemoryStateStore())
    yield
    set_state_store(None)

MINIMAL_POLICY_YAML = """\
default:
  secrets:
    enabled: true
  jailbreak:
    enabled: true
    threshold: 60
  pii:
    enabled: true
    action: REDACT
"""


@pytest.fixture(scope="session")
def fake_pat() -> str:
    # Built at runtime so the literal never appears in the source tree
    # (the repo's secret scanner would flag a pattern-matching literal).
    return "ghp_" + "Ab1" * 12


@pytest.fixture(scope="session")
def ed25519_keys():
    """A fresh (private_pem, public_pem) Ed25519 keypair."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


@pytest.fixture()
def policy_file(tmp_path):
    """A minimal policies.yaml on disk."""
    path = tmp_path / "policies.yaml"
    path.write_text(MINIMAL_POLICY_YAML)
    return path
