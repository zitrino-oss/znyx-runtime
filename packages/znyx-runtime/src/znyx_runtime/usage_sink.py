"""Where each gateway call's usage goes: a usage event (``znyx_core.usage_events``,
metadata only) appended to a local spool that the leader worker ships to the control
plane (managed mode), and a span for the OTLP exporter (when one is configured).

The spool is shared by the worker processes (appends are locked) and survives a
restart or an unreachable control plane: usage, and so spend, is not lost.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from znyx_core import usage_events

logger = logging.getLogger(__name__)

DEFAULT_USAGE_SPOOL = Path.home() / ".znyx" / "llm-calls.spool"


class UsageSink:
    def __init__(self, bundle_manager, spool_path: Optional[str] = None, exporter=None):
        self.bundle_manager = bundle_manager
        self.spool_path = spool_path
        self.exporter = exporter
        self._warned = False

    def _scope(self, record):
        """The environment and project the call belongs to: its key's bundle-set member."""
        member = None
        try:
            member = self.bundle_manager.member_by_key(record.member or None)
        except Exception:  # noqa: BLE001 - a call the runtime refused belongs to no member
            member = None
        if member is None:
            return "", ""
        return member.environment_id or "", member.project_id or member.bundle.project_id or ""

    def event_for(self, record) -> dict:
        environment_id, project_id = self._scope(record)
        return usage_events.build(
            occurred_at=record.occurred_at, request_id=record.request_id,
            trace_id=record.trace_id or None, span_id=record.span_id or None,
            parent_span_id=record.parent_span_id, environment_id=environment_id or None,
            project_id=project_id or None, tenant_id=record.tenant_id, app_id=record.app_id,
            agent_id=record.agent_id, key_id=record.key_id, user=record.user, api=record.api,
            provider=record.provider or None, provider_format=record.provider_format,
            model=record.model, response_model=record.response_model, stream=record.stream,
            status=record.status, outcome=record.outcome, reason=record.reason,
            input_decision=record.input_decision, output_decision=record.output_decision,
            rule_ids=list(record.rule_ids), input_tokens=record.input_tokens,
            output_tokens=record.output_tokens, cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens, usage_estimated=record.usage_estimated,
            cost_usd=record.cost_usd, price_known=record.price_known, latency_ms=record.latency_ms,
            upstream_ms=record.upstream_ms, ttft_ms=record.ttft_ms, guard_ms=record.guard_ms,
            techniques=dict(record.techniques), profile=getattr(record, "profile", None),
            saved_usd=getattr(record, "saved_usd", None),
            baseline_cost_usd=getattr(record, "baseline_cost_usd", None),
            optimisations=list(getattr(record, "optimisations", None) or []) or None,
            tools_offered=list(getattr(record, "tools_offered", None) or []) or None,
            tools_called=list(getattr(record, "tools_called", None) or []) or None,
            mcp_tools_changed=list(getattr(record, "mcp_tools_changed", None) or []) or None,
            mcp_tools_seen=list(getattr(record, "mcp_tools_seen", None) or []) or None,
            delegated=True if getattr(record, "delegated", False) else None,
            identity_provider=getattr(record, "identity_provider", None),
            prompt_hash=getattr(record, "prompt_hash", None), prompt_ref=getattr(record, "prompt_ref", None))

    def __call__(self, record) -> None:
        if not record.key_id:
            # Refused before any key was recognised: not usage of anyone's, and recording
            # it would let anyone who can reach the port fill the spool (metrics count it).
            return
        event = self.event_for(record)
        if self.spool_path:
            try:
                from znyx_runtime.shipper import append_record
                append_record(self.spool_path, event)
            except OSError as exc:
                if not self._warned:
                    self._warned = True
                    logger.error("Usage spool %s cannot be written (%s); usage is not delivered",
                                 self.spool_path, exc)
        if self.exporter is not None:
            self.exporter.export_call(record, event)
