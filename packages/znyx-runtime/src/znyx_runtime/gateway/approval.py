"""Holding a tool call until a person approves it (Phase 5, technique 33).

The policy decides which calls need approval (``znyx_core.approvals``). Two ways to wait:

- ``ask``: the gateway holds the call (an MCP ``tools/call``, a tool call in a model's answer)
  until a person decides or the policy's ``timeout_seconds`` passes.
- ``request`` then ``check``: the evaluate API answers at once with the approval's id
  (``pending_review_id``) and the SDK calls again with that id once someone decided. One
  approval is used once.

Where the decision comes from:

- managed mode: the control plane (``POST /v1/runtime/approvals``, then ``GET`` with a
  long-poll). It signs decisions with the bundle signing key; a runtime that has the public key
  verifies them, and one that requires signed bundles requires them. A decision counts only for
  the same id, tool and argument hash. If the control plane cannot be reached the call is
  refused.
- no console: the shared state store, decided with the approver key (``ZNYX_APPROVER_KEY``,
  never a gateway or instance key, so an agent cannot approve its own calls).

Requests are also kept on the runtime with the full arguments (viewable inside the network);
the console only ever gets the guard-redacted preview, or nothing under ``share: metadata``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from znyx_core import approvals

logger = logging.getLogger(__name__)

_LOCAL_PREFIX = "pin:gw:approval:"
_LOCAL_INDEX = "pin:gw:approvals"
_USED_PREFIX = "pin:gw:approval-used:"
POLL_SECONDS = 25
KEEP_SECONDS = 86400          # decided requests stay listed for a day
MAX_HOLDS_PER_KEY = 20        # calls of one key waiting at once (in this process)


class ApprovalBroker:
    def __init__(self, gateway) -> None:
        self.gw = gateway
        self._client = None
        self._holds: Counter = Counter()

    # -- where decisions come from ------------------------------------------------

    def _settings(self):
        return getattr(self.gw.bundle_manager, "config", None)

    def managed(self) -> bool:
        cfg = self._settings()
        return bool(cfg is not None and getattr(cfg, "mode", "") == "managed"
                    and getattr(cfg, "control_plane_url", "") and getattr(cfg, "runtime_token", ""))

    def _http(self):
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(POLL_SECONDS + 10, connect=10),
                                             follow_redirects=False)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _cp(self) -> Tuple[str, Dict[str, str]]:
        cfg = self._settings()
        return cfg.control_plane_url.rstrip("/"), {"X-API-Key": cfg.runtime_token}

    # -- building a request ---------------------------------------------------------

    @staticmethod
    def _store():
        from znyx_core.state_store import get_state_store
        return get_state_store()

    async def _create(self, *, rules: approvals.ApprovalRules, reason: str, tool: str, arguments: Any,
                      server: Optional[str], record, member) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Store the request (locally, and at the control plane in managed mode). Raises
        ``approvals.AmbiguousArguments`` for arguments that cannot be approved."""
        value = approvals.parse_arguments(arguments)          # a key given twice: refused here
        approval_id = "zapr_" + uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        entry: Dict[str, Any] = {
            "id": approval_id, "request_id": record.request_id,
            "environment_id": member.environment_id if member is not None else None,
            "project_id": member.project_id if member is not None else None,
            "app_id": record.app_id, "agent_id": record.agent_id, "key_id": record.key_id, "user": record.user,
            "server": server, "tool": tool, "reason": reason, "arguments_sha256": approvals.arguments_sha256(value),
            "trace_id": record.trace_id or None,
            "expires_at": (now + timedelta(seconds=rules.timeout_seconds)).isoformat(),
        }
        preview = await asyncio.to_thread(approvals.preview, value, rules.share)
        if preview is not None:
            entry["arguments_preview"] = preview
        local = {**entry, "status": "pending", "created_at": now.isoformat(), "managed": self.managed(),
                 "arguments": value if isinstance(value, str) else json.dumps(value, default=str)}
        store = self._store()
        await asyncio.to_thread(self._write, store, local, KEEP_SECONDS + rules.timeout_seconds)
        await asyncio.to_thread(store.push_capped, _LOCAL_INDEX, approval_id, 500, 7 * 86400)
        if self.managed():
            base, headers = self._cp()
            created = await self._http().post(f"{base}/v1/runtime/approvals", json=entry, headers=headers)
            created.raise_for_status()
        record.note("approval_asked")
        self._audit("approval.asked", entry, record, member)
        return entry, local

    def _audit(self, kind: str, entry: Dict[str, Any], record=None, member=None, **extra: Any) -> None:
        """A governance record (part 4) of a request or decision; never the arguments."""
        from znyx_runtime.governance_audit import call_fields
        try:
            fields = call_fields(record, member) if record is not None else {
                "key_id": entry.get("key_id"), "app_id": entry.get("app_id"), "agent_id": entry.get("agent_id"),
                "trace_id": entry.get("trace_id"), "project_id": entry.get("project_id"),
                "environment_id": entry.get("environment_id")}
            self.gw.governance.record(kind, **fields, approval_id=entry.get("id"), tool=entry.get("tool"),
                                      server=entry.get("server"), reason=entry.get("reason"),
                                      arguments_sha256=entry.get("arguments_sha256"), **extra)
        except Exception:  # noqa: BLE001 - the audit never fails a call
            logger.debug("gateway: governance record failed", exc_info=True)

    # -- waiting (gateway calls) ----------------------------------------------------

    async def ask(self, *, rules: approvals.ApprovalRules, reason: str, tool: str, arguments: Any,
                  server: Optional[str], record, member=None) -> Tuple[bool, str]:
        """Hold until a person decides or the policy's time runs out. ``(approved, why)``."""
        holder = f"{record.member}/{record.key_id}"
        if self._holds[holder] >= MAX_HOLDS_PER_KEY:
            record.note("approval_refused")
            return False, "too many of this key's calls are already waiting for approval"
        self._holds[holder] += 1
        entry: Dict[str, Any] = {"id": "-"}
        try:
            entry, local = await self._create(rules=rules, reason=reason, tool=tool, arguments=arguments,
                                              server=server, record=record, member=member)
            deadline = time.monotonic() + rules.timeout_seconds
            if self.managed():
                status = await self._wait_console(entry, deadline)
                await asyncio.to_thread(self._write, self._store(), {**local, "status": status}, KEEP_SECONDS)
            else:
                status = await self._wait_locally(local, deadline)
        except asyncio.CancelledError:
            raise
        except approvals.AmbiguousArguments as exc:
            record.note("approval_refused")
            return False, f"the arguments are ambiguous ({exc})"
        except Exception as exc:  # noqa: BLE001 - when in doubt the call does not run
            logger.warning("gateway: an approval for %s could not be asked: %s", tool, exc)
            record.note("approval_refused")
            return False, "the approval could not be asked (refused, fail closed)"
        finally:
            self._holds[holder] -= 1
            if self._holds[holder] <= 0:
                del self._holds[holder]
        self._audit("approval.outcome", entry, record, member, status=status)
        if status == "approved":
            record.note("approval_granted")
            return True, f"approved ({entry['id']})"
        record.note("approval_refused")
        return False, {"denied": f"a person refused it ({entry['id']})",
                       "expired": f"no one approved it in time ({entry['id']})"}.get(
            status, f"not approved ({entry['id']}: {status})")

    async def _wait_console(self, entry: Dict[str, Any], deadline: float) -> str:
        base, headers = self._cp()
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return "expired"
            body = await self._console_status(base, headers, entry["id"], max(0, min(POLL_SECONDS, int(left))))
            status = body.get("status")
            if status == "pending":
                continue
            if status in ("denied", "expired"):
                return status
            if status == "approved" and self._trusted(body, entry):
                return "approved"
            logger.warning("gateway: approval %s came back %r and is not trusted", entry["id"], status)
            return "untrusted"

    async def _console_status(self, base: str, headers: Dict[str, str], approval_id: str, wait: int) -> Dict[str, Any]:
        resp = await self._http().get(f"{base}/v1/runtime/approvals/{approval_id}", params={"wait": wait},
                                      headers=headers)
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, dict) else {}

    def _trusted(self, body: Dict[str, Any], entry: Dict[str, Any]) -> bool:
        """An approved decision for this exact call: same id, tool and argument hash, and
        signed when this runtime requires signed bundles (verified whenever it has the key)."""
        decision = body.get("decision")
        if not isinstance(decision, dict):
            return False
        if (decision.get("id") != entry["id"] or decision.get("status") != "approved"
                or decision.get("tool") != entry["tool"]
                or decision.get("arguments_sha256") != entry["arguments_sha256"]):
            return False
        cfg = self._settings()
        public_key = getattr(cfg, "bundle_public_key", "") if cfg is not None else ""
        if public_key:
            return approvals.verify_decision(decision, body.get("signature"), public_key)
        return not getattr(cfg, "require_signed_bundles", False)

    async def _wait_locally(self, local: Dict[str, Any], deadline: float) -> str:
        store = self._store()
        while time.monotonic() < deadline:
            current = await asyncio.to_thread(self._read, store, local["id"])
            status = (current or {}).get("status")
            if status in ("approved", "denied"):
                return status
            await asyncio.sleep(1)
        current = await asyncio.to_thread(self._read, store, local["id"]) or local
        if current.get("status") in ("approved", "denied"):
            return current["status"]
        await asyncio.to_thread(self._write, store, {**current, "status": "expired"}, KEEP_SECONDS)
        return "expired"

    # -- asking without waiting (the evaluate API) ----------------------------------

    async def request(self, *, rules: approvals.ApprovalRules, reason: str, tool: str, arguments: Any,
                      server: Optional[str], record, member=None) -> str:
        """Ask and return the approval's id at once; the caller asks again with it."""
        entry, _ = await self._create(rules=rules, reason=reason, tool=tool, arguments=arguments, server=server,
                                      record=record, member=member)
        return entry["id"]

    async def check(self, approval_id: str, tool: str, arguments: Any, key_id: str) -> str:
        """The state of an earlier request for this exact call by this key: ``approved`` (once:
        using it spends it), ``pending``, or why not (``denied``, ``expired``, ``used``,
        ``unknown``, ``untrusted``)."""
        store = self._store()
        local = await asyncio.to_thread(self._read, store, str(approval_id)[:64])
        if local is None:
            return "unknown"
        try:
            digest = approvals.arguments_sha256(approvals.parse_arguments(arguments))
        except approvals.AmbiguousArguments:
            return "untrusted"
        if local.get("tool") != tool or local.get("arguments_sha256") != digest or local.get("key_id") != key_id:
            return "untrusted"                    # another call: an approval covers exactly one
        if local.get("managed"):
            base, headers = self._cp()
            body = await self._console_status(base, headers, local["id"], 0)
            status = body.get("status") if body.get("status") in approvals.STATUSES else "unknown"
            if status == "approved" and not self._trusted(body, local):
                status = "untrusted"
        else:
            status = local.get("status") or "unknown"
            if status == "pending" and _past(local.get("expires_at")):
                status = "expired"
        if status != "approved":
            return status
        used = await asyncio.to_thread(store.set_value, _USED_PREFIX + local["id"], "1", KEEP_SECONDS, True)
        if used:
            self._audit("approval.used", local)
        return "approved" if used else "used"

    # -- the runtime's own view (inside the network) ----------------------------------

    @staticmethod
    def _read(store, approval_id: str) -> Optional[Dict[str, Any]]:
        raw = store.get_value(_LOCAL_PREFIX + approval_id)
        try:
            value = json.loads(raw) if raw else None
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write(store, entry: Dict[str, Any], ttl: float) -> None:
        store.set_value(_LOCAL_PREFIX + entry["id"], json.dumps(entry, default=str), ttl)

    def local_list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        store = self._store()
        out = []
        for approval_id in dict.fromkeys(reversed(store.get_list(_LOCAL_INDEX))):
            entry = self._read(store, approval_id)
            if not entry:
                continue
            if entry.get("status") == "pending" and _past(entry.get("expires_at")):
                entry["status"] = "expired"
            if status is None or entry.get("status") == status:
                out.append(entry)
        return out

    def local_get(self, approval_id: str) -> Optional[Dict[str, Any]]:
        return self._read(self._store(), approval_id)

    def local_decide(self, approval_id: str, decision: str, by: str, note: Optional[str]) -> Dict[str, Any]:
        """Decide a request kept in the shared state (no console). The first decision wins:
        a request no longer pending (decided, or past its time) is refused."""
        store = self._store()
        entry = self._read(store, approval_id)
        if not entry:
            raise LookupError("No such approval request")
        if entry.get("managed"):
            raise PermissionError("This runtime takes decisions from the console")
        if entry.get("status") != "pending" or _past(entry.get("expires_at")):
            raise RuntimeError("This request is no longer waiting")
        # One decision per request, across workers: the first to write the decision marker wins.
        if not store.set_value(_LOCAL_PREFIX + approval_id + ":decided", decision, KEEP_SECONDS, only_if_absent=True):
            raise RuntimeError("This request was already decided")
        entry.update(status="approved" if decision == "approve" else "denied", decided_by=by,
                     decided_at=datetime.now(timezone.utc).isoformat(), note=(note or "")[:500] or None)
        self._write(store, entry, KEEP_SECONDS)
        self._audit("approval.decided", entry, status=entry["status"], decided_by=by)
        return entry


def _past(moment: Optional[str]) -> bool:
    try:
        return datetime.fromisoformat(str(moment)) <= datetime.now(timezone.utc)
    except ValueError:
        return True
