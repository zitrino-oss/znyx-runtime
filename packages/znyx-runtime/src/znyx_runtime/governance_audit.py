"""Governance audit records (AI gateway Phase 5, part 4).

What an administrator or auditor needs to reconstruct about agents' actions, metadata only:
MCP tool calls (server, tool, outcome, whether a person approved it), approval requests and
decisions, and delegated tokens issued and revoked. Written to a hash-chained spool
(``znyx_core.utils.spool.append_chained``): each record carries the hash of the one before, so
a record removed or changed afterwards shows. In managed mode the leader ships the spool to the
control plane, which checks the chain and anchors its head in its own audit chain.

Writing never delays or fails a call: a record that cannot be written is logged.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Set

from znyx_core.utils.spool import append_chained

logger = logging.getLogger(__name__)

DEFAULT_GOVERNANCE_SPOOL = Path.home() / ".znyx" / "governance-audit.spool"


class GovernanceAudit:
    LOG_EVERY_SECONDS = 60.0

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.getenv("ZNYX_GOVERNANCE_SPOOL_PATH") or DEFAULT_GOVERNANCE_SPOOL)
        self._pending: Set[asyncio.Future] = set()
        self.failures = 0                      # records that could not be written (lost)
        self._logged_at = float("-inf")
        self._unlogged = 0

    def _write(self, event: dict) -> None:
        try:
            append_chained(self.path, [event])
        except Exception as exc:  # noqa: BLE001 - an audit record never fails the call it describes
            self.failures += 1
            self._unlogged += 1
            now = time.monotonic()
            if now - self._logged_at >= self.LOG_EVERY_SECONDS:
                self._logged_at = now
                logger.error("Governance audit spool %s cannot be written (%d record(s) lost since the last "
                             "message, %d in all): %s", self.path, self._unlogged, self.failures, exc)
                self._unlogged = 0

    def record(self, kind: str, **fields: Any) -> None:
        """Write one record in the background (from async code) or now (elsewhere)."""
        event = {"event_id": uuid.uuid4().hex, "occurred_at": datetime.now(timezone.utc).isoformat(), "type": kind,
                 **{k: v for k, v in fields.items() if v not in (None, "", [], {})}}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write(event)
            return
        task = loop.run_in_executor(None, self._write, event)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)


def call_fields(record, member=None) -> dict:
    """Who a record is about, from a gateway call record."""
    return {"org_scope": record.tenant_id or None, "key_id": record.key_id or None, "app_id": record.app_id or None,
            "agent_id": record.agent_id or None, "person": record.user if getattr(record, "delegated", False) else None,
            "trace_id": record.trace_id or None,
            "project_id": getattr(member, "project_id", None), "environment_id": getattr(member, "environment_id", None)}
