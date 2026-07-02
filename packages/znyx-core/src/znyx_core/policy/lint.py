"""Policy contradiction / lint analysis (F2).

Beyond schema validation (which catches malformed values), this flags *logical*
contradictions a schema-valid policy can still contain — most importantly a detector
whose strategy requests a remote execution mode while the policy forbids external
calls, so that mode could never run. Surfaced by POST /v1/policies/validate for the
console editor's inline lint; also the basis for P1's policy_contradiction hook.
"""
from __future__ import annotations

from typing import Any, Dict, List

from znyx_core.policy.schema import PolicyValidationIssue

_REMOTE_MODES = {"remote_llm", "remote_api"}
_NON_DETERMINISTIC = {"local_ml", "local_embedding", "local_llm", "remote_llm", "remote_api"}
# Top-level keys that are not detector slots (skip them when scanning detectors).
_NON_DETECTOR_KEYS = {
    "runtime_policy", "policy_version", "schema_enforcement", "tool_governance",
    "_multilingual", "metadata", "description", "name", "version", "on_fail",
    "custom_detectors", "output_contract", "quality_scoring",
}


def _no_external_calls(policy: Dict[str, Any]) -> bool:
    runtime = policy.get("runtime_policy") or {}
    return bool(runtime.get("no_external_calls") or policy.get("no_external_calls"))


def analyze_policy(policy: Dict[str, Any]) -> List[PolicyValidationIssue]:
    """Return contradiction findings (as warnings) for a schema-valid policy dict."""
    issues: List[PolicyValidationIssue] = []
    if not isinstance(policy, dict):
        return issues

    no_external = _no_external_calls(policy)

    for key, cfg in policy.items():
        if key in _NON_DETECTOR_KEYS or not isinstance(cfg, dict):
            continue
        strat = cfg.get("strategy")
        if not isinstance(strat, dict):
            continue
        order = strat.get("order") or []
        backends = cfg.get("backends") or {}

        # 1) remote mode requested but external calls are forbidden → can never run.
        remote = [m for m in order if m in _REMOTE_MODES]
        if remote and no_external:
            issues.append(PolicyValidationIssue(
                code="contradiction",
                loc=f"{key}.strategy.order",
                message=(f"requests remote mode(s) {remote} but runtime_policy.no_external_calls is set — "
                         "those modes can never run and will always fall back"),
            ))

        # 2) a non-deterministic mode in the order with no matching backends block.
        for mode in order:
            if mode in _NON_DETERMINISTIC and mode not in backends:
                issues.append(PolicyValidationIssue(
                    code="missing_backend",
                    loc=f"{key}.backends.{mode}",
                    message=f"strategy.order includes '{mode}' but no backends.{mode} is configured",
                ))

    return issues
