"""Tool-permission audit (OWASP LLM03 - Excessive Agency).

LLM03 names three root causes: excessive functionality, excessive permissions, and
excessive autonomy. ``human_approval_gate`` covers autonomy and ``excessive_agency``
scores what a plan does at runtime. This audits the DECLARATION — the tool or function
schema itself — so an over-broad capability is caught when it is registered, before any
agent has the chance to misuse it.

Runs in the ``tool_registration`` stage, alongside ``mcp_manifest_scanner``. The two
look at the same artifact for different things: the scanner hunts injection and
exfiltration markers hidden in a manifest, while this measures how much power the tool
is asking for. A perfectly honest manifest can still declare a shell.

What it flags, taken from LLM03's own examples:

* **Open-ended tools** — run a shell command, fetch an arbitrary URL, execute raw SQL.
  OWASP's guidance is explicit that these should be replaced with granular tools.
* **Bundled destructive capability** — a tool described as reading that also declares
  delete/drop/purge, the "read tool that can also modify and delete" case.
* **Wildcard or admin scopes** — ``*``, ``admin``, ``write:*``, ``root``.
* **Unconstrained free-form parameters** — a ``command``/``query``/``path`` string with
  no ``enum``, ``pattern``, or ``const`` to bound it, which is an open-ended tool wearing
  a schema.

Verbs are matched against the tool's NAME and declared capabilities rather than its
prose description, so a description that merely mentions deletion ("never deletes
anything") does not trip it.
"""
import json
import re
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

_MAX_DEPTH = 6

# Open-ended executors. These are the tools OWASP tells you to replace with granular ones.
_OPEN_ENDED = re.compile(
    r'\b(?:exec|eval|shell|bash|sh|cmd|command|system|subprocess|spawn|'
    r'run[_\s-]?(?:code|script|shell|command|query)|'
    r'raw[_\s-]?(?:sql|query)|sql[_\s-]?query|execute[_\s-]?sql|'
    r'fetch[_\s-]?url|http[_\s-]?request|curl|wget|browse)\b',
    re.IGNORECASE)

_DESTRUCTIVE = re.compile(
    r'\b(?:delete|destroy|drop|truncate|purge|wipe|erase|remove|revoke|'
    r'terminate|deprovision|uninstall)\b', re.IGNORECASE)

_READ_ONLY_INTENT = re.compile(
    r'\b(?:read|get|list|fetch|view|search|query|lookup|show|describe|summar)',
    re.IGNORECASE)

# Scope strings that grant more than one thing.
_WILDCARD_SCOPE = re.compile(r'(?:^|[\s,:])\*$|:\*|\*:|\ball\b|\beverything\b', re.IGNORECASE)
_ADMIN_SCOPE = re.compile(r'\b(?:admin|root|superuser|owner|full[_\s-]?access|write[_\s-]?all)\b',
                          re.IGNORECASE)

# Parameter names that are dangerous when unconstrained.
_FREEFORM_RISKY_PARAMS = {"command", "cmd", "query", "sql", "script", "code",
                          "path", "file", "filepath", "url", "endpoint", "expression"}

# Where a declaration may carry its permissions.
_SCOPE_FIELDS = ("scopes", "permissions", "scope", "permission", "grants", "roles", "access")
_NAME_FIELDS = ("name", "tool_name", "function_name", "id")
_CAPABILITY_FIELDS = ("capabilities", "operations", "actions", "methods", "verbs")


def _as_declarations(payload: Any) -> List[Dict[str, Any]]:
    """Normalise the shapes a caller sends into a flat list of tool declarations."""
    if isinstance(payload, dict):
        for key in ("tools", "functions", "tool_schemas", "manifest"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [d for d in inner if isinstance(d, dict)]
            if isinstance(inner, dict):
                return [inner]
        return [payload]
    if isinstance(payload, list):
        return [d for d in payload if isinstance(d, dict)]
    return []


def _unwrap(decl: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-style declarations nest the real body under "function"."""
    inner = decl.get("function")
    return inner if isinstance(inner, dict) else decl


def _collect_strings(node: Any, depth: int = 0) -> List[str]:
    """Every string in a subtree, depth-bounded so a deep manifest cannot recurse away."""
    if depth > _MAX_DEPTH:
        return []
    if isinstance(node, str):
        return [node]
    out: List[str] = []
    if isinstance(node, dict):
        for v in node.values():
            out.extend(_collect_strings(v, depth + 1))
    elif isinstance(node, list):
        for v in node:
            out.extend(_collect_strings(v, depth + 1))
    return out


def _declared_scopes(body: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for f in _SCOPE_FIELDS:
        out.extend(_collect_strings(body.get(f)))
    return out


def _tool_identity(body: Dict[str, Any]) -> List[str]:
    """Name plus declared capabilities — the fields that state what the tool DOES.

    Returned in two forms, raw and separator-normalised, and callers search both.
    ``_`` and ``-`` are word characters to ``re``, so a trailing ``\b`` never matches
    inside an identifier: ``run_shell_command`` would slip past a pattern that matches
    ``run shell`` perfectly well. Normalising catches snake_case and kebab-case; keeping
    the raw form too means nothing is lost when a pattern relies on the separator.

    The prose description is excluded on purpose: "this tool never deletes records"
    contains the word delete and would otherwise be flagged as destructive."""
    parts: List[str] = []
    for f in _NAME_FIELDS:
        v = body.get(f)
        if isinstance(v, str):
            parts.append(v)
    for f in _CAPABILITY_FIELDS:
        parts.extend(_collect_strings(body.get(f)))
    raw = " ".join(parts)
    return [raw, re.sub(r'[_\-]+', ' ', raw)]


def _unconstrained_params(body: Dict[str, Any]) -> List[str]:
    """Free-form string params with a risky name and no enum/pattern/const bound."""
    params = body.get("parameters") if isinstance(body.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    risky: List[str] = []
    for pname, spec in props.items():
        if not isinstance(spec, dict):
            continue
        if str(pname).lower() not in _FREEFORM_RISKY_PARAMS:
            continue
        if spec.get("type") not in (None, "string"):
            continue
        if any(k in spec for k in ("enum", "pattern", "const", "format")):
            continue        # bounded: the schema constrains what can be passed
        risky.append(str(pname))
    return risky


class ToolPermissionAuditDetector:
    """Audits tool declarations for excessive functionality and permissions (LLM03)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        # WARN by default. Registration-time findings are design feedback, and blocking a
        # tool the app already depends on breaks the app rather than the attack — an org
        # that wants a hard gate sets BLOCK deliberately.
        self.action = (self.config.get("action") or "WARN").upper()
        self.flag_open_ended = bool(self.config.get("flag_open_ended", True))
        self.flag_destructive_bundling = bool(self.config.get("flag_destructive_bundling", True))
        self.flag_wildcard_scopes = bool(self.config.get("flag_wildcard_scopes", True))
        self.flag_unconstrained_params = bool(self.config.get("flag_unconstrained_params", True))
        allow = self.config.get("allowed_tools") or []
        self.allowed_tools = {str(a).lower() for a in allow}

    def detect(self, text: str,
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        for decl in _as_declarations(payload):
            body = _unwrap(decl)
            name = ""
            for f in _NAME_FIELDS:
                if isinstance(body.get(f), str):
                    name = body[f]
                    break
            if name.lower() in self.allowed_tools:
                continue        # explicitly sanctioned by the operator

            identities = _tool_identity(body)
            label = name or "<unnamed tool>"

            if self.flag_open_ended and any(_OPEN_ENDED.search(i) for i in identities):
                rule_hits.append(RuleHit(
                    rule_id="tool_permission_audit.open_ended_tool",
                    severity=Severity.HIGH,
                    message=(f"Tool '{label}' declares open-ended execution; prefer a "
                             f"granular tool over a general executor"),
                ))

            if (self.flag_destructive_bundling
                    and any(_DESTRUCTIVE.search(i) for i in identities)
                    and any(_READ_ONLY_INTENT.search(i) for i in identities)):
                rule_hits.append(RuleHit(
                    rule_id="tool_permission_audit.destructive_capability_bundled",
                    severity=Severity.HIGH,
                    message=(f"Tool '{label}' bundles destructive capability with read "
                             f"access; split them so the agent gets only what it needs"),
                ))

            if self.flag_wildcard_scopes:
                for scope in _declared_scopes(body):
                    if _WILDCARD_SCOPE.search(scope):
                        rule_hits.append(RuleHit(
                            rule_id="tool_permission_audit.wildcard_scope",
                            severity=Severity.HIGH,
                            message=f"Tool '{label}' requests wildcard scope '{scope}'",
                        ))
                        break
                for scope in _declared_scopes(body):
                    if _ADMIN_SCOPE.search(scope):
                        rule_hits.append(RuleHit(
                            rule_id="tool_permission_audit.admin_scope",
                            severity=Severity.HIGH,
                            message=f"Tool '{label}' requests administrative scope '{scope}'",
                        ))
                        break

            if self.flag_unconstrained_params:
                risky = _unconstrained_params(body)
                if risky:
                    rule_hits.append(RuleHit(
                        rule_id="tool_permission_audit.unconstrained_parameter",
                        severity=Severity.MEDIUM,
                        message=(f"Tool '{label}' takes unbounded free-form parameter(s) "
                                 f"{', '.join(risky)}; add an enum or pattern"),
                    ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"tool_permission_audit: {', '.join(sorted({h.rule_id for h in rule_hits}))}"
        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="This tool requests more access than the policy allows.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
