"""Human approval for risky tool calls (AI gateway Phase 5, technique 33).

A policy names the tools whose calls need a person's approval (``approval.tools``, name
patterns) and can add MCP tools annotated destructive (``approval.mcp_destructive``). The
gateway holds such a call while a person decides, in the console (managed mode) or with the
instance key (no console), and lets it through only when approved. One approval covers exactly
one call: it is bound to the SHA-256 of the call's arguments.

The person deciding sees the tool, who asked and why, and by default the arguments after the
guard's redaction (PII and secrets removed, at most ``PREVIEW_CHARS`` characters); a policy can
choose metadata only (``share: metadata``). Full arguments never leave the runtime.

The control plane signs each decision with the bundle signing key (Ed25519), so a runtime that
verifies bundles verifies decisions the same way.
"""
from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120
MIN_TIMEOUT_SECONDS = 10
MAX_TIMEOUT_SECONDS = 600
PREVIEW_CHARS = 2000
SHARES = ("redacted", "metadata")
STATUSES = ("pending", "approved", "denied", "expired")
RULE_NOT_APPROVED = "approval.not_approved"
RULE_BATCH = "approval.not_possible_in_batch"
RULE_AMBIGUOUS = "approval.ambiguous_arguments"
_DOMAIN = b"znyx-approval-v1\n"


@dataclass(frozen=True)
class ApprovalRules:
    tools: Tuple[str, ...] = ()
    mcp_destructive: bool = False
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    share: str = "redacted"

    def reason(self, tool_name: Optional[str], annotations: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """Why a call to this tool needs approval, or None when it does not."""
        name = tool_name or ""
        pattern = next((p for p in self.tools if fnmatch.fnmatchcase(name, p)), None)
        if pattern is not None:
            return f"the policy asks for approval of {pattern}"
        if self.mcp_destructive and annotations is not None:
            # The MCP spec's defaults: a tool may change things unless it says it only reads,
            # and a tool that may change things is destructive unless it says it is not.
            hints = annotations if isinstance(annotations, Mapping) else {}
            if hints.get("readOnlyHint") is not True and hints.get("destructiveHint") is not False:
                return "the MCP server does not say this tool is read-only or harmless"
        return None


def rules_of(policy: Optional[Mapping[str, Any]]) -> Optional[ApprovalRules]:
    """The policy's approval rules, or None when it asks for none. Lenient: odd values fall
    back to the defaults rather than switching approval off."""
    section = policy.get("approval") if isinstance(policy, Mapping) else None
    if not isinstance(section, Mapping) or section.get("enabled") is False:
        return None
    tools = section.get("tools") or []
    tools = tuple(str(t) for t in tools if isinstance(t, str) and t.strip())[:200] if isinstance(tools, list) else ()
    destructive = section.get("mcp_destructive") is True
    if not tools and not destructive:
        return None
    timeout = section.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        timeout = DEFAULT_TIMEOUT_SECONDS
    timeout = max(MIN_TIMEOUT_SECONDS, min(MAX_TIMEOUT_SECONDS, timeout))
    share = section.get("share") if section.get("share") in SHARES else "redacted"
    return ApprovalRules(tools=tools, mcp_destructive=destructive, timeout_seconds=timeout, share=share)


class AmbiguousArguments(ValueError):
    """Arguments whose meaning depends on the reader (a key given twice): never approved."""


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise AmbiguousArguments(f"the key {key!r} appears twice")
        out[key] = value
    return out


def parse_arguments(arguments: Any) -> Any:
    """The arguments as one unambiguous value: JSON text parsed (a key given twice is refused,
    since readers disagree on which one counts), anything else as it is."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments, object_pairs_hook=_no_duplicates)
        except AmbiguousArguments:
            raise
        except ValueError:
            return arguments
    return arguments


def arguments_sha256(arguments: Any) -> str:
    """The hash an approval is bound to: of the canonical JSON of the arguments, or of the
    text as sent when it is not JSON."""
    arguments = parse_arguments(arguments)
    if isinstance(arguments, str):
        return hashlib.sha256(arguments.encode("utf-8", "surrogatepass")).hexdigest()
    text = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


MAX_PREVIEW_INPUT = 64 * 1024
_PEM = re.compile(r"-----BEGIN [A-Z0-9 ]+-----.*?(?:-----END [A-Z0-9 ]+-----|$)", re.S)


def preview(arguments: Any, share: str) -> Optional[str]:
    """What the person deciding sees of the arguments: the canonical form of what will run
    (so the preview cannot differ from it), PII and secrets redacted first and only then
    shortened, the shortening said plainly; None under ``share: metadata`` or when the
    arguments are too large to clean."""
    if share != "redacted":
        return None
    value = parse_arguments(arguments)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) > MAX_PREVIEW_INPUT:
        return None
    text = _PEM.sub("[PRIVATE KEY]", text)
    from znyx_core.detectors.pii import PIIDetector
    from znyx_core.detectors.secrets import SecretsDetector
    for detector in (SecretsDetector({"enabled": True, "action": "REDACT"}),
                     PIIDetector({"enabled": True, "action": "REDACT"})):
        try:
            result = detector.detect(text)
        except Exception:  # noqa: BLE001 - a preview that cannot be cleaned is not shown
            logger.warning("approval preview could not be redacted; metadata only")
            return None
        if result.sanitized_text is not None:
            text = result.sanitized_text
    if len(text) <= PREVIEW_CHARS:
        return text
    marker = f" [{len(text) - (PREVIEW_CHARS - 40):,} more characters not shown]"
    return text[:PREVIEW_CHARS - 40] + marker


def canonical(decision: Mapping[str, Any]) -> bytes:
    return _DOMAIN + json.dumps(dict(decision), sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True).encode("utf-8")


def sign_decision(decision: Mapping[str, Any], private_key_pem: Optional[str]) -> Optional[str]:
    if not private_key_pem:
        return None
    from znyx_core.policy.bundle_set import _load_key
    key = _load_key(private_key_pem, private=True)
    return base64.b64encode(key.sign(canonical(decision))).decode()


def verify_decision(decision: Mapping[str, Any], signature: Optional[str], public_key_pem: str) -> bool:
    if not signature:
        return False
    try:
        from znyx_core.policy.bundle_set import _load_key
        _load_key(public_key_pem, private=False).verify(base64.b64decode(signature), canonical(decision))
        return True
    except Exception:  # noqa: BLE001 - any failure is "not verified"
        return False
