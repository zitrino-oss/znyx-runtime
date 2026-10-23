import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from znyx_core import defence
from znyx_runtime.gateway.defence import DefenceOutbox
from test_phase4_optimise import _fresh, client, openai_call, user


def feed(text):
    return {"enabled": True, "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "signatures": [defence.signature(text, "prompt_injection")]}


def test_normalization_matches_known_attacks_but_never_transmits_the_text(tmp_path):
    text = "IGNORE   instructions"
    assert defence.fingerprint(text) == defence.fingerprint("ｉｇｎｏｒｅ instructions")
    value = defence.signature(text, "prompt_injection")
    assert set(value) == {"version", "digest", "category"} and "IGNORE" not in json.dumps(value)
    assert defence.match(text, feed(text))
    expired = {**feed(text), "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    assert not defence.match(text, expired)
    assert not defence.match(text, {**feed(text), "enabled": False})
    with pytest.raises(ValueError):
        defence.clean_signature({**value, "text": text})


def test_outbox_is_inspectable_and_opt_out_discards_without_network_calls(tmp_path):
    outbox = DefenceOutbox(tmp_path / "defence.spool")
    shared = feed("known attack")
    outbox.record("known attack", ["jailbreak.pattern"], shared)
    assert len(outbox.inspect()) == 1
    assert "known attack" not in outbox.path.read_text()
    sender = AsyncMock(return_value=True)
    gateway = type("Gateway", (), {"_controls": lambda self: {"shared_defence": {"enabled": False}}})()
    assert asyncio.run(outbox.sender(gateway, sender)("https://cp.test", outbox.inspect()))
    sender.assert_not_awaited()


def test_all_outbound_history_is_pageable_without_raw_text(tmp_path):
    from znyx_core.utils.spool import append_lines
    outbox = DefenceOutbox(tmp_path / "defence.spool")
    values = [defence.signature(f"attack {index}", "jailbreak") for index in range(503)]
    append_lines(outbox.history, [json.dumps(value) for value in values])
    first = outbox.inspect_page()
    second = outbox.inspect_page(first["next_cursor"])
    assert first["contributions"] + second["contributions"] == values
    assert second["next_cursor"] is None and "attack " not in outbox.history.read_text()
    with pytest.raises(ValueError):
        outbox.inspect_page(1)


@pytest.mark.parametrize("stream", [False, True])
def test_reviewed_output_fingerprint_is_withheld_in_both_modes(client, monkeypatch, stream):
    text = "a newly reviewed attack fingerprint"
    monkeypatch.setattr(client.gateway, "_controls", lambda: {"shared_defence": feed(text)})
    response = openai_call(client, [user("SAY:" + text)], key="nothing", stream=stream)
    assert response.status_code == 200, response.text
    assert text not in response.text and "content_filter" in response.text


def test_new_signature_invalidates_cached_allow_for_historical_input(client, monkeypatch):
    text = "a previously allowed historical message"
    current = {}
    monkeypatch.setattr(client.gateway, "_controls", lambda: current)
    messages = [user(text), {"role": "assistant", "content": "acknowledged"}, user("hello")]
    assert openai_call(client, messages, key="nothing").status_code == 200
    current["shared_defence"] = feed(text)
    response = openai_call(client, messages, key="nothing")
    assert response.status_code == 400 and "content_filter" in response.text


def test_feed_renewal_rechecks_history_allowed_during_expiry_even_if_revision_is_unchanged(client, monkeypatch):
    text = "historical message allowed while the network was unavailable"
    shared = {**feed(text), "revision": "same-reviewed-signatures",
              "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    monkeypatch.setattr(client.gateway, "_controls", lambda: {"shared_defence": shared})
    messages = [user(text), {"role": "assistant", "content": "acknowledged"}, user("hello")]
    assert openai_call(client, messages, key="nothing").status_code == 200
    shared["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    response = openai_call(client, messages, key="nothing")
    assert response.status_code == 400 and "content_filter" in response.text


@pytest.mark.asyncio
async def test_signed_feed_is_applied_to_direct_sdk_evaluation_not_only_gateway(tmp_path, monkeypatch, ed25519_keys):
    from test_bundle_sets import _set, _manager, _ControlPlane
    from znyx_core.policy.bundle_set import sign_bundle_set
    from znyx_core.engine.evaluator import GuardrailsEvaluator
    from znyx_core.core.models import Decision, EvaluationRequest
    private, public = ed25519_keys
    text = "reviewed attack visible to a direct SDK evaluation"
    bundle = _set(private)
    bundle.controls["shared_defence"] = feed(text)
    sign_bundle_set(bundle, private)
    manager = _manager(tmp_path, public, _ControlPlane(bundle.to_dict()), monkeypatch)
    assert await manager._fetch_bundle()
    policy = manager.get_policy(member=manager.members[0])
    result = await GuardrailsEvaluator().evaluate(EvaluationRequest(
        request_id="sdk-defence", tenant_id="acme", app_id="test", text=text), policy=policy)
    assert result.decision == Decision.BLOCK
    manager._set.controls["shared_defence"] = {"enabled": False}
    assert "shared_defence" not in manager.get_policy(member=manager.members[0])
