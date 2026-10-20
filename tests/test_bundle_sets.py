"""One runtime serving several projects: the bundle set.

The control plane serves a runtime one signed bundle per environment it serves, in a
signed set. These tests pin the contract (``znyx_core.policy.bundle_set``), the bundle
manager reading it (and older control planes that serve one bundle), and the runtime
app in managed mode serving two projects whose policies and gateway keys differ:
every call gets its own project's policy, provider and credential, never another's.
"""
import hashlib
import json
import os

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

import gateway_fake_provider as fake
from znyx_core.policy.bundle import build_bundle
from znyx_core.policy.bundle_set import (
    BundleSet, BundleSetMember, sign_bundle_set, validate_bundle_set,
)

ORG = "8d0c1f9e-0000-4000-8000-000000000001"
ALPHA, BETA = "8d0c1f9e-0000-4000-8000-0000000000a1", "8d0c1f9e-0000-4000-8000-0000000000b1"
ENV_ALPHA, ENV_BETA = "8d0c1f9e-0000-4000-8000-0000000000a2", "8d0c1f9e-0000-4000-8000-0000000000b2"
KEY_ALPHA, KEY_BETA = "zxg_alpha_app_key", "zxg_beta_app_key"
INSTANCE_KEY = "zx_set_instance_key"
EMAIL = "jane.doe@example.com"


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture(scope="module")
def keys():
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()).decode()
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return private_pem, public_pem


def _gateway(name, secret_var, key, project):
    return {"version": 1,
            "providers": {"openai": {"format": "openai", "base_url": "https://api.openai.com/v1",
                                     "api_key": f"env:{secret_var}", "region": "us"}},
            "keys": [{"id": f"{name}-app", "key_sha256": _sha(key), "tenant_id": ORG, "app_id": project,
                      "environment": "prod"}]}


def _members(private_pem, *, alpha_pii="REDACT", pins=None):
    alpha_policy = {"pii": {"enabled": True, "action": alpha_pii},
                    "gateway": _gateway("alpha", "ZNYX_SECRET_ALPHA", KEY_ALPHA, ALPHA)}
    beta_policy = {"pii": {"enabled": False}, "gateway": _gateway("beta", "ZNYX_SECRET_BETA", KEY_BETA, BETA)}
    if pins:
        alpha_policy["runtime_policy"] = {"inference": {"pins": pins[0]}}
        beta_policy["runtime_policy"] = {"inference": {"pins": pins[1]}}
    alpha = build_bundle(alpha_policy, ORG, ALPHA, "prod", private_key_pem=private_pem)
    beta = build_bundle(beta_policy, ORG, BETA, "prod", private_key_pem=private_pem)
    return [BundleSetMember(ENV_ALPHA, ALPHA, "prod", alpha, "alpha"),
            BundleSetMember(ENV_BETA, BETA, "prod", beta, "beta")]


def _set(private_pem, *, keep=("alpha", "beta"), missing=(), serves_several=True, **kw):
    members = [m for m in _members(private_pem, **kw) if m.project_slug in keep]
    bundle_set = BundleSet(org_id=ORG, set_id="key-1", members=members, serves_several=serves_several,
                           missing=[dict(entry) for entry in missing])
    sign_bundle_set(bundle_set, private_pem)
    return bundle_set


MISSING_ALPHA = {"environment_id": ENV_ALPHA, "project_id": ALPHA, "environment": "prod",
                 "reason": "secret_references_required"}


# -- the contract ---------------------------------------------------------------------------

def test_a_signed_set_validates_and_round_trips(keys):
    private_pem, public_pem = keys
    bundle_set = _set(private_pem)
    assert validate_bundle_set(bundle_set, public_pem, True, True) == (True, "")
    again = BundleSet.from_dict(json.loads(json.dumps(bundle_set.to_dict())))
    assert validate_bundle_set(again, public_pem, True, True) == (True, "")
    assert again.content_hash() == bundle_set.content_hash()


def test_members_cannot_be_swapped_dropped_or_repeated(keys):
    private_pem, public_pem = keys
    swapped = _set(private_pem)
    swapped.members[0].bundle, swapped.members[1].bundle = swapped.members[1].bundle, swapped.members[0].bundle
    assert not validate_bundle_set(swapped, public_pem, True, True)[0]
    dropped = _set(private_pem)
    dropped.members.pop()
    assert not validate_bundle_set(dropped, public_pem, True, True)[0]
    repeated = _set(private_pem)
    repeated.members.append(repeated.members[0])
    assert not validate_bundle_set(repeated, public_pem, True, True)[0]


def test_an_unsigned_set_is_refused_when_signatures_are_required(keys):
    private_pem, public_pem = keys
    unsigned = _set(private_pem)
    unsigned.signature = None
    assert not validate_bundle_set(unsigned, public_pem, True, False)[0]
    assert validate_bundle_set(unsigned, None, False, False)[0]


def test_a_member_from_another_organisation_is_refused(keys):
    private_pem, public_pem = keys
    bundle_set = _set(private_pem)
    stranger = build_bundle({"pii": {"enabled": False}}, "other-org", BETA, "prod", private_key_pem=private_pem)
    bundle_set.members[1] = BundleSetMember(ENV_BETA, BETA, "prod", stranger, "beta")
    sign_bundle_set(bundle_set, private_pem)
    assert not validate_bundle_set(bundle_set, public_pem, True, True)[0]


def test_the_content_hash_changes_only_with_what_is_served(keys):
    private_pem, _ = keys
    first = _set(private_pem)
    assert first.content_hash() != _set(private_pem, alpha_pii="BLOCK").content_hash()
    before = first.content_hash()
    first.generated_at = "2030-01-01T00:00:00+00:00"
    sign_bundle_set(first, private_pem)
    assert first.content_hash() == before


# -- the bundle manager ----------------------------------------------------------------------

class _ControlPlane:
    def __init__(self, payload, legacy=False):
        self.payload, self.legacy, self.calls = payload, legacy, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.url.path, request.headers.get("if-none-match")))
        if request.url.path == "/v1/bundle-sets/latest":
            if self.legacy:
                return httpx.Response(404, json={"detail": "Not Found"})
            etag = BundleSet.from_dict(self.payload).content_hash()
            if request.headers.get("if-none-match") == etag:
                return httpx.Response(304)
            return httpx.Response(200, json=self.payload, headers={"ETag": etag})
        if request.url.path == "/v1/bundles/latest":
            return httpx.Response(200, json=self.payload, headers={"ETag": self.payload.get("policy_hash", "")})
        return httpx.Response(404)


def _manager(tmp_path, public_pem, cp, monkeypatch):
    from znyx_runtime.bundle_manager import BundleManager
    from znyx_runtime.config import RuntimeConfig
    bm = BundleManager(RuntimeConfig(mode="managed", control_plane_url="https://cp.test",
                                     runtime_token="rt", bundle_public_key=public_pem,
                                     require_signed_bundles=True, require_bundle_sig_v2=True,
                                     fail_mode="closed", cache_dir=str(tmp_path)))
    bm._transport = httpx.MockTransport(cp.handler)
    return bm


@pytest.mark.asyncio
async def test_the_manager_serves_each_environment_its_own_policy(keys, tmp_path, monkeypatch):
    from znyx_runtime.bundle_manager import ScopeNotServed
    private_pem, public_pem = keys
    bundle_set = _set(private_pem)
    cp = _ControlPlane(bundle_set.to_dict())
    bm = _manager(tmp_path, public_pem, cp, monkeypatch)
    assert await bm._fetch_bundle() is True
    assert bm.serves_several and bm.effective_env is None and bm._bundle is None

    alpha, beta = bm.members
    assert bm.get_policy(member=alpha)["pii"]["action"] == "REDACT"
    assert bm.get_policy(member=beta)["pii"]["enabled"] is False
    assert bm.get_policy(member=alpha)["policy_version"] == alpha.bundle.bundle_id
    # named by project id or slug, and environment
    assert bm.member_for(None, "beta", "prod") is beta
    assert bm.member_for(None, ALPHA, "staging") is alpha        # one environment of that project
    with pytest.raises(ScopeNotServed):
        bm.member_for(None, "my-app", "prod")
    with pytest.raises(ScopeNotServed):
        bm.member_by_key("not-a-member")
    assert [i["environment_id"] for i in bm.report_infos()] == [ENV_ALPHA, ENV_BETA]

    # unchanged: 304, nothing replaced
    assert await bm._fetch_bundle() is False
    assert cp.calls[-1] == ("/v1/bundle-sets/latest", bundle_set.content_hash())
    assert (tmp_path / "bundle_set_cache.json").exists() and not (tmp_path / "bundle_cache.json").exists()


@pytest.mark.asyncio
async def test_a_tampered_set_is_not_taken(keys, tmp_path, monkeypatch):
    private_pem, public_pem = keys
    payload = _set(private_pem).to_dict()
    payload["members"][1]["bundle"]["policies"]["pii"]["enabled"] = True
    bm = _manager(tmp_path, public_pem, _ControlPlane(payload), monkeypatch)
    with pytest.raises(ValueError, match="failed validation"):
        await bm._fetch_bundle()
    assert not bm.members


@pytest.mark.asyncio
async def test_boot_without_the_control_plane_uses_the_cached_set(keys, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    private_pem, public_pem = keys
    bm = _manager(tmp_path, public_pem, _ControlPlane(_set(private_pem).to_dict()), monkeypatch)
    await bm._fetch_bundle()

    async def no_sleep(_):
        return None
    monkeypatch.setattr("znyx_runtime.bundle_manager.asyncio.sleep", no_sleep)
    later = _manager(tmp_path, public_pem, _ControlPlane({}), monkeypatch)
    later._fetch_bundle = AsyncMock(side_effect=RuntimeError("unreachable"))
    await later._load_managed()
    assert len(later.members) == 2

    # a cache someone edited is not trusted
    cache = tmp_path / "bundle_set_cache.json"
    doc = json.loads(cache.read_text())
    doc["members"][0]["bundle"]["policies"]["pii"]["enabled"] = False
    cache.write_text(json.dumps(doc))
    poisoned = _manager(tmp_path, public_pem, _ControlPlane({}), monkeypatch)
    poisoned._fetch_bundle = AsyncMock(side_effect=RuntimeError("unreachable"))
    with pytest.raises(RuntimeError, match="fail-closed"):
        await poisoned._load_managed()


@pytest.mark.asyncio
async def test_an_older_control_plane_is_read_as_a_set_of_one(keys, tmp_path, monkeypatch):
    private_pem, public_pem = keys
    single = build_bundle({"pii": {"enabled": True}}, ORG, ALPHA, "prod", private_key_pem=private_pem)
    cp = _ControlPlane(single.to_dict(), legacy=True)
    bm = _manager(tmp_path, public_pem, cp, monkeypatch)
    assert await bm._fetch_bundle() is True
    assert bm.effective_env == "prod" and bm._bundle.bundle_id == single.bundle_id
    assert bm.member_for(None, "anything", "dev") is bm.members[0]
    await bm._fetch_bundle()
    assert [path for path, _ in cp.calls] == ["/v1/bundle-sets/latest", "/v1/bundles/latest",
                                              "/v1/bundles/latest"]
    assert (tmp_path / "bundle_cache.json").exists()


@pytest.mark.asyncio
async def test_model_pins_of_every_environment_reach_the_one_model_server(keys, tmp_path, monkeypatch):
    private_pem, public_pem = keys
    pins = ({"pii_ner": {"model": "a"}, "language": {"model": "l"}},
            {"pii_ner": {"model": "b"}, "toxicity": {"model": "t"}})
    bm = _manager(tmp_path, public_pem, _ControlPlane(_set(private_pem, pins=pins).to_dict()), monkeypatch)
    await bm._fetch_bundle()
    merged = bm._active_policy_for_listeners()["runtime_policy"]["inference"]["pins"]
    assert merged == {"pii_ner": {"model": "a"}, "language": {"model": "l"}, "toxicity": {"model": "t"}}


# -- the runtime app serving two projects ------------------------------------------------------

@pytest.fixture(scope="module")
def runtime(tmp_path_factory, keys):
    private_pem, public_pem = keys
    tmp = tmp_path_factory.mktemp("bundle-sets")
    cp = _ControlPlane(_set(private_pem).to_dict())
    env = {"ZNYX_MODE": "managed", "ZNYX_CONTROL_PLANE_URL": "https://cp.test", "ZNYX_RUNTIME_TOKEN": "rt",
           "ZNYX_BUNDLE_PUBLIC_KEY": public_pem, "ZNYX_REQUIRE_BUNDLE_SIG_V2": "true",
           "ZNYX_CACHE_DIR": str(tmp / "cache"), "ZNYX_BUNDLE_POLL_INTERVAL": "3600",
           "RUNTIME_API_KEY": INSTANCE_KEY, "RUNTIME_REQUIRE_AUTH": "true", "ZNYX_TELEMETRY": "false",
           "ZNYX_TELEMETRY_ENABLED": "false", "ZNYX_AUDIT_SPOOL_PATH": str(tmp / "egress.spool"),
           "ZNYX_JUDGE_AUDIT_SPOOL_PATH": str(tmp / "judge.spool"),
           "ZNYX_SECRET_ALPHA": "sk-alpha-upstream", "ZNYX_SECRET_BETA": "sk-beta-upstream"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    import znyx_runtime.install_state as install_state
    from znyx_runtime.bundle_manager import BundleManager
    from znyx_runtime.gateway.upstream import Upstream
    from znyx_runtime.runtime_report import RuntimeReporter
    from znyx_runtime.shipper import SpoolShipper
    saved_state = (install_state.STATE_DIR, install_state.STATE_FILE)
    install_state.STATE_DIR = tmp / ".znyx"
    install_state.STATE_FILE = install_state.STATE_DIR / "state.json"
    provider_app = fake.build_app()
    originals = (Upstream.client, BundleManager._transport, RuntimeReporter.report, SpoolShipper.start)
    reports = []

    async def fake_client(self, provider):
        client = self._clients.get(provider.name)
        if client is None:
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=provider_app))
            self._clients[provider.name] = client
        return client

    async def record_report(self, info, **kw):
        reports.append(dict(info))
        return True

    Upstream.client = fake_client
    BundleManager._transport = httpx.MockTransport(cp.handler)
    RuntimeReporter.report = record_report
    SpoolShipper.start = lambda self: None
    try:
        from znyx_runtime.main import app
        with TestClient(app) as c:
            c.reports = reports
            yield c
    finally:
        Upstream.client, BundleManager._transport, RuntimeReporter.report, SpoolShipper.start = originals
        install_state.STATE_DIR, install_state.STATE_FILE = saved_state
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@pytest.fixture(autouse=True)
def _fresh():
    fake.reset()
    yield


def _chat(client, key, text):
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                       json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}]})


def test_each_app_key_gets_its_own_projects_policy_and_credential(runtime):
    resp = _chat(runtime, KEY_ALPHA, f"SAY:ok, mail {EMAIL}")
    assert resp.status_code == 200, resp.text
    sent = fake.received[-1]
    assert "[EMAIL]" in sent["body"]["messages"][0]["content"]
    assert sent["headers"]["authorization"] == "Bearer sk-alpha-upstream"

    resp = _chat(runtime, KEY_BETA, f"SAY:ok, mail {EMAIL}")
    assert resp.status_code == 200, resp.text
    sent = fake.received[-1]
    assert EMAIL in sent["body"]["messages"][0]["content"]
    assert sent["headers"]["authorization"] == "Bearer sk-beta-upstream"


def test_the_instance_key_names_no_project_on_a_shared_runtime(runtime):
    resp = _chat(runtime, INSTANCE_KEY, "SAY:ok")
    assert resp.status_code == 403 and not fake.received


def test_evaluate_routes_use_the_keys_project(runtime):
    body = {"request_id": "r1", "tenant_id": ORG, "app_id": "default", "text": f"mail {EMAIL}"}
    alpha = runtime.post("/v1/evaluate/input", headers={"X-API-Key": KEY_ALPHA}, json=body).json()
    beta = runtime.post("/v1/evaluate/input", headers={"X-API-Key": KEY_BETA}, json=body).json()
    assert alpha["decision"] == "REDACT" and beta["decision"] == "ALLOW"
    bm = runtime.app.state.gateway.bundle_manager
    assert alpha["policy_version"] == bm.members[0].bundle.bundle_id
    assert beta["policy_version"] == bm.members[1].bundle.bundle_id


def test_the_instance_key_must_name_the_project(runtime):
    body = {"request_id": "r2", "tenant_id": ORG, "text": f"mail {EMAIL}", "env": "prod"}
    resp = runtime.post("/v1/evaluate/input", headers={"X-API-Key": INSTANCE_KEY},
                        json={**body, "app_id": "my-app"})
    assert resp.status_code == 400 and "several projects" in resp.json()["detail"]
    resp = runtime.post("/v1/evaluate/input", headers={"X-API-Key": INSTANCE_KEY},
                        json={**body, "app_id": "alpha"})
    assert resp.status_code == 200 and resp.json()["decision"] == "REDACT"


def test_an_app_key_cannot_evaluate_for_another_project(runtime):
    body = {"request_id": "r3", "tenant_id": ORG, "app_id": BETA, "text": "hello"}
    resp = runtime.post("/v1/evaluate/input", headers={"X-API-Key": KEY_ALPHA}, json=body)
    assert resp.status_code == 403


def test_the_runtime_reports_each_environment(runtime):
    assert {r.get("environment_id") for r in runtime.reports} >= {ENV_ALPHA, ENV_BETA}


def test_the_status_names_each_environment_only_for_the_instance_key(runtime):
    status = runtime.get("/v1/bundle/status", headers={"X-API-Key": INSTANCE_KEY}).json()
    assert status["bundle_set"] is True
    assert [m["environment_id"] for m in status["members"]] == [ENV_ALPHA, ENV_BETA]
    public = runtime.get("/v1/bundle/status").json()
    assert public == {"ready": True, "environments": 2}
    assert ALPHA not in json.dumps(public)


# -- review fixes (2026-10-20) ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_shared_key_with_one_member_left_never_answers_for_another_project(keys, tmp_path, monkeypatch):
    """The set says it is for several projects; with alpha missing, a call naming alpha
    must not get beta's (weaker) policy, and the instance key names no project."""
    from znyx_runtime.bundle_manager import ScopeNotServed
    from znyx_runtime.gateway.config import GatewayKey, from_bundle_members
    private_pem, public_pem = keys
    cp = _ControlPlane(_set(private_pem, keep=("beta",), missing=[{**MISSING_ALPHA, "reason": "no_published_bundle"}]).to_dict())
    bm = _manager(tmp_path, public_pem, cp, monkeypatch)
    await bm._fetch_bundle()
    assert len(bm.members) == 1 and bm.serves_several and bm.effective_env is None
    with pytest.raises(ScopeNotServed):
        bm.member_for(None, "alpha", "prod")
    with pytest.raises(ScopeNotServed):
        bm.member_by_key(None)
    assert bm.member_for(None, "beta", "prod") is bm.members[0]
    config = from_bundle_members(bm.gateway_documents(), shared=bm.serves_several)
    with pytest.raises(LookupError):
        config.provider_for(GatewayKey(id="instance", key_sha256="", instance_key=True), "openai")


@pytest.mark.asyncio
async def test_an_environment_listed_missing_keeps_its_last_good_bundle(keys, tmp_path, monkeypatch):
    """The control plane cannot serve alpha any more (its secrets need references):
    alpha keeps its last bundle, as a runtime keeps its bundle when the control plane
    answers with an error, and keeps it across a restart."""
    from unittest.mock import AsyncMock
    private_pem, public_pem = keys
    cp = _ControlPlane(_set(private_pem).to_dict())
    bm = _manager(tmp_path, public_pem, cp, monkeypatch)
    await bm._fetch_bundle()
    alpha_bundle = bm.members[0].bundle.bundle_id
    cp.payload = _set(private_pem, keep=("beta",), missing=[MISSING_ALPHA]).to_dict()
    assert await bm._fetch_bundle() is True
    assert [m.project_slug for m in bm.members] == ["beta", "alpha"]
    assert bm.get_policy(member=bm.member_for(None, "alpha", "prod"))["policy_version"] == alpha_bundle

    async def no_sleep(_):
        return None
    monkeypatch.setattr("znyx_runtime.bundle_manager.asyncio.sleep", no_sleep)
    restarted = _manager(tmp_path, public_pem, _ControlPlane({}), monkeypatch)
    restarted._fetch_bundle = AsyncMock(side_effect=RuntimeError("unreachable"))
    await restarted._load_managed()
    assert sorted(m.project_slug for m in restarted.members) == ["alpha", "beta"]

    # a member the set does not name as missing (the environment was removed from the
    # key) is not kept
    cp.payload = _set(private_pem, keep=("beta",)).to_dict()
    await bm._fetch_bundle()
    assert [m.project_slug for m in bm.members] == ["beta"]


@pytest.mark.asyncio
async def test_a_single_environment_runtime_keeps_its_bundle_when_it_cannot_be_served(keys, tmp_path, monkeypatch):
    private_pem, public_pem = keys
    one = BundleSet(org_id=ORG, set_id=f"env:{ENV_ALPHA}", members=[_members(private_pem)[0]])
    sign_bundle_set(one, private_pem)
    cp = _ControlPlane(one.to_dict())
    bm = _manager(tmp_path, public_pem, cp, monkeypatch)
    await bm._fetch_bundle()
    empty = BundleSet(org_id=ORG, set_id=f"env:{ENV_ALPHA}", members=[], missing=[MISSING_ALPHA])
    sign_bundle_set(empty, private_pem)
    cp.payload = empty.to_dict()
    await bm._fetch_bundle()
    assert bm.effective_env == "prod"
    assert bm.get_policy()["pii"]["action"] == "REDACT"


@pytest.mark.asyncio
async def test_a_transient_404_does_not_pin_the_runtime_to_single_bundles(keys, tmp_path, monkeypatch):
    private_pem, public_pem = keys
    payload = _set(private_pem).to_dict()
    state = {"set_404": True}

    def handler(request):
        if request.url.path == "/v1/bundle-sets/latest":
            if state["set_404"]:
                return httpx.Response(404)
            return httpx.Response(200, json=payload, headers={"ETag": "x"})
        return httpx.Response(409, json={"detail": "This runtime key serves several environments"})
    from znyx_runtime.bundle_manager import BundleManager
    from znyx_runtime.config import RuntimeConfig
    monkeypatch.setattr(BundleManager, "_transport", httpx.MockTransport(handler))
    bm = BundleManager(RuntimeConfig(mode="managed", control_plane_url="https://cp.test", runtime_token="rt",
                                     bundle_public_key=public_pem, require_signed_bundles=True,
                                     fail_mode="closed", cache_dir=str(tmp_path)))
    with pytest.raises(httpx.HTTPStatusError):
        await bm._fetch_bundle()
    state["set_404"] = False
    assert await bm._fetch_bundle() is True and len(bm.members) == 2


def test_the_legacy_flag_is_never_read_from_a_response(keys):
    private_pem, public_pem = keys
    doc = _set(private_pem).to_dict()
    doc["members"].pop()
    doc["legacy"], doc["signature"] = True, None
    forged = BundleSet.from_dict(doc)
    assert forged.legacy is False
    assert not validate_bundle_set(forged, public_pem, True, True)[0]


def test_renaming_a_project_or_environment_changes_the_etag(keys):
    private_pem, _ = keys
    before = _set(private_pem)
    after = _set(private_pem)
    after.members[0].project_slug = "alpha-renamed"
    assert before.content_hash() != after.content_hash()


def test_the_same_key_id_in_two_projects_has_its_own_budget(runtime):
    from znyx_runtime.gateway.config import GatewayKey
    gw = runtime.app.state.gateway
    a = GatewayKey(id="app", key_sha256="", member=ENV_ALPHA)
    b = GatewayKey(id="app", key_sha256="", member=ENV_BETA)
    assert gw._day_key(a) != gw._day_key(b)


def test_development_traffic_does_not_share_abuse_state_with_production():
    from znyx_core.core.models import EvaluationRequest
    from znyx_core.engine.orchestrator import DetectorOrchestrator
    seen = []

    class Recorder:
        def detect(self, text, **kw):
            seen.append(kw["app_id"])
            from znyx_core.core.models import Decision, DetectorResult
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

    orch = DetectorOrchestrator.__new__(DetectorOrchestrator)
    for env in ("dev", "prod"):
        req = EvaluationRequest(request_id="r", tenant_id="t", app_id="p", env=env, text="hi")
        orch._invoke_detector("abuse", Recorder(), "hi", req, "input")
    assert seen == ["p@dev", "p@prod"]


def test_a_disabled_key_stops_working_with_the_next_fetch(runtime, keys):
    """The gateway is rebuilt the moment the bundles change, not after the slower
    cycle listeners."""
    import asyncio
    private_pem, _ = keys
    bm = runtime.app.state.gateway.bundle_manager
    cp_payload = _set(private_pem).to_dict()
    for member in cp_payload["members"]:
        if member["project_slug"] == "beta":
            member["bundle"]["policies"]["gateway"]["keys"][0]["disabled"] = True
    from znyx_core.policy.bundle import resign_bundle
    rebuilt = BundleSet.from_dict(cp_payload)
    for member in rebuilt.members:
        resign_bundle(member.bundle, private_pem)
    sign_bundle_set(rebuilt, private_pem)
    original = bm._transport
    bm._transport = httpx.MockTransport(lambda request: httpx.Response(200, json=rebuilt.to_dict(),
                                                                       headers={"ETag": "new"}))
    try:
        asyncio.run(bm._fetch_bundle())
        assert _chat(runtime, KEY_BETA, "SAY:ok").status_code == 401
        assert _chat(runtime, KEY_ALPHA, "SAY:ok").status_code == 200
    finally:
        bm._transport = original
