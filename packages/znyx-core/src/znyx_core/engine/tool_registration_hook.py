"""``tool_registration`` lifecycle hook (P1b, OWASP LLM01/LLM03).

A lifecycle hook (not per-request traffic): scan a tool / MCP manifest at registration
time. Invoked from the control-plane tool-registration service path, mirroring how the
``policy_publish`` hook (policy contradiction analysis) runs inside policy validate /
bundle publish rather than the message pipeline.
"""
from typing import Any, Dict, Optional

from znyx_core.core.models import DetectorResult
from znyx_core.detectors.mcp_manifest_scanner import McpManifestScannerDetector


def scan_tool_manifest(manifest: Any, config: Optional[Dict[str, Any]] = None) -> DetectorResult:
    """Run the MCP manifest scanner over ``manifest`` at registration time.

    The hook is invoked explicitly (a management action), so the scanner defaults to
    enabled unless the caller's config disables it.
    """
    cfg = dict(config or {})
    cfg.setdefault("enabled", True)
    return McpManifestScannerDetector(cfg).detect(manifest)
