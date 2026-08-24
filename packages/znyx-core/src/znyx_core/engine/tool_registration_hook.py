"""``tool_registration`` lifecycle hook (OWASP LLM01/LLM03/LLM04).

A lifecycle hook (not per-request traffic): scan a tool / MCP manifest at registration
time. Invoked from the control-plane tool-registration service path, mirroring how the
``policy_publish`` hook (policy contradiction analysis) runs inside policy validate /
bundle publish rather than the message pipeline.
"""
import json
from typing import Any, Dict, Optional

from znyx_core.core.models import DetectorResult
from znyx_core.detectors.mcp_manifest_scanner import McpManifestScannerDetector
from znyx_core.detectors.tool_permission_audit import ToolPermissionAuditDetector


def scan_tool_manifest(manifest: Any, config: Optional[Dict[str, Any]] = None) -> DetectorResult:
    """Run the MCP manifest scanner over ``manifest`` at registration time.

    The hook is invoked explicitly (a management action), so the scanner defaults to
    enabled unless the caller's config disables it.
    """
    cfg = dict(config or {})
    cfg.setdefault("enabled", True)
    return McpManifestScannerDetector(cfg).detect(manifest)


def audit_tool_permissions(manifest: Any, config: Optional[Dict[str, Any]] = None) -> DetectorResult:
    """Audit ``manifest`` for excessive functionality and permissions (LLM03).

    Runs at the same moment as ``scan_tool_manifest`` and over the same artifact, but
    answers a different question: the scanner asks whether the manifest is trying to
    attack you, this asks how much power it is asking for. A completely honest manifest
    can still declare a shell.

    Like the scanner it defaults to enabled — the hook is an explicit management action,
    so the caller has already decided to look.
    """
    cfg = dict(config or {})
    cfg.setdefault("enabled", True)
    payload = manifest if isinstance(manifest, str) else json.dumps(manifest, default=str)
    return ToolPermissionAuditDetector(cfg).detect(payload)
