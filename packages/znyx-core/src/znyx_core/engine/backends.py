"""DetectorBackend resolver + escalation strategy.

Turns a detector's policy ``strategy`` + per-mode ``backends`` config into a typed
``BackendStrategy`` the escalation engine runs (deterministic → ml → llm). The six
execution modes live in core.models (``ExecutionMode``); this module is the runtime
view that maps an ordered mode list + escalate predicates onto concrete backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from znyx_core.core.models import ExecutionMode

_VALID_MODES = frozenset(m.value for m in ExecutionMode)
_DETERMINISTIC = ExecutionMode.local_deterministic.value

# Modes served by the co-located inference sidecar. Single owner: ``escalation`` imports
# this rather than keeping its own copy, because ``_resolve_ml_endpoints`` below and
# ``escalation._normalize_kind`` must agree on exactly which modes get a sidecar address.
ML_MODES = frozenset({"local_ml", "local_embedding"})

# Fields copied from a policy backends.<mode> block into a DetectorBackend.
_BACKEND_FIELDS = (
    "endpoint_url", "model_id", "revision", "sha256", "task", "threshold",
    "provider", "judge_id", "timeout_ms", "auth_type", "auth_value",
    "region", "in_boundary", "provider_config", "params",
)


@dataclass
class DetectorBackend:
    """How to reach one execution mode's model/judge/vendor."""
    mode: str
    endpoint_url: Optional[str] = None
    model_id: Optional[str] = None
    revision: Optional[str] = None
    sha256: Optional[str] = None
    task: Optional[str] = None
    threshold: Optional[float] = None
    provider: Optional[str] = None
    judge_id: Optional[str] = None
    timeout_ms: Optional[int] = None
    auth_type: Optional[str] = None
    auth_value: Optional[str] = None
    region: Optional[str] = None
    in_boundary: Optional[bool] = None      # None → default per mode (see egress.is_boundary_crossing)
    # Vendor-adapter-specific settings (e.g. Azure api-version/severity threshold,
    # Bedrock guardrail id/version) merged into the adapter's config dict.
    provider_config: Optional[Dict[str, Any]] = None
    # Per-request runner params (e.g. allowed_languages for the language runner).
    # Passed through to the inference sidecar in the request body.
    params: Optional[Dict[str, Any]] = None

    @property
    def model_version(self) -> Optional[str]:
        if self.model_id and self.revision:
            return f"{self.model_id}@{self.revision}"
        return self.model_id


@dataclass
class BackendStrategy:
    """Ordered execution modes + escalation predicates + resolved egress policy."""
    order: List[str]
    escalate_when: Optional[Dict[str, Any]] = None
    fallback: Optional[str] = None
    timeout_ms: Optional[int] = None
    # Additive: the escalated ML layer augments (worst-of) the deterministic result rather
    # than replacing it — so a detector's deterministic decision (e.g. PII regex redaction)
    # survives while the ML layer adds what it alone catches (unstructured PII, language).
    additive: bool = False
    backends: Dict[str, DetectorBackend] = field(default_factory=dict)
    # Egress policy: per-detector allowlist/redaction + resolved runtime_policy.
    no_external_calls: bool = False
    allowed_regions: Optional[List[str]] = None
    egress_allowlist: Optional[List[str]] = None
    redact_pii: bool = False
    redact_secrets: bool = False
    # The org's PII / secrets detector configs, so egress redaction covers exactly the
    # types the org redacts in-pipeline (incl. default-disabled ones). None → defaults.
    redact_pii_config: Optional[Dict[str, Any]] = None
    redact_secrets_config: Optional[Dict[str, Any]] = None

    def backend_for(self, mode: str) -> Optional[DetectorBackend]:
        return self.backends.get(mode)


def layer_org_default(detector_config: Dict[str, Any],
                      org_default: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Layer an org-level default ``{strategy, backends}`` UNDER one detector config.

    The policy's own ``strategy`` always wins — the org default only fills a MISSING strategy
    (and its backends). One exception: ``auth_value`` (API keys) is back-filled from the org
    default into any policy backend that has a matching mode but no key set, even when the
    policy owns the strategy. This lets the ExecutionModeSelector UI store credentials in the
    org defaults without requiring a full strategy override to propagate them.
    Returns a new dict when it layered something, else the original.
    Pure (no DB), so the publish path, the CP eval path, and ``build_strategy`` all share the
    exact same merge semantics."""
    if not isinstance(org_default, dict) or not isinstance(org_default.get("strategy"), dict):
        return detector_config
    if isinstance(detector_config.get("strategy"), dict):
        # Policy strategy wins overall — but back-fill missing auth_value fields from the
        # org default's backends so credentials set via the UI reach the bundle.
        org_backends = org_default.get("backends")
        if not isinstance(org_backends, dict):
            return detector_config
        # Mutable copy — safe even when the policy has no "backends" key at all
        policy_backends = dict(detector_config.get("backends") or {})
        order = list((detector_config.get("strategy") or {}).get("order") or [])
        changed = False
        for mode, org_be in org_backends.items():
            if not isinstance(org_be, dict) or not org_be.get("auth_value"):
                continue
            pol_be = policy_backends.get(mode)
            if isinstance(pol_be, dict):
                # Mode in policy but missing auth_value → back-fill
                if not pol_be.get("auth_value"):
                    policy_backends[mode] = {**pol_be, "auth_value": org_be["auth_value"]}
                    changed = True
            elif mode in order:
                # Mode in strategy order but no backend config yet → add from org default
                policy_backends[mode] = org_be
                changed = True
        if not changed:
            return detector_config
        return {**detector_config, "backends": policy_backends}
    merged = dict(detector_config)
    merged["strategy"] = org_default["strategy"]
    if "backends" not in merged and isinstance(org_default.get("backends"), dict):
        merged["backends"] = org_default["backends"]
    return merged


def merge_org_defaults_into_policy(policies: Dict[str, Any],
                                   defaults_map: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of ``policies`` with each org detector default layered under the matching
    detector config. Only detectors ALREADY PRESENT in the policy are affected — org defaults
    fill in a missing execution strategy, they do NOT silently enable detectors the policy
    never mentioned. Reserved/non-dict keys (runtime_policy, _scorecard_gate, …) are untouched.
    This is the seam that gets the stored org defaults INTO the effective policy at compile/
    publish time (so the DB-less runtime sees them in the bundle)."""
    if not defaults_map:
        return policies
    out = dict(policies)
    for key, default in defaults_map.items():
        cfg = out.get(key)
        if isinstance(cfg, dict):
            out[key] = layer_org_default(cfg, default)
    return out


def _resolve_ml_endpoints(backends: Dict[str, "DetectorBackend"]) -> None:
    """Fill in ``endpoint_url`` / ``in_boundary`` for ML backends that carry neither.

    The inference sidecar is co-located with this runtime at a fixed convention address —
    loopback when it runs as a same-pod container, a service name on a compose network.
    The control plane cannot know which, so it ships no address at all and actively strips
    any stale one at publish time. This function is therefore the ONLY place a ``local_ml``
    backend acquires an address, and without it escalation raises "no endpoint configured
    for mode local_ml" and silently degrades every ML detector to rules.

    Deliberately narrow:

    * **ML modes only.** ``remote_api`` / ``remote_llm`` / ``local_llm`` name third-party or
      operator-chosen endpoints; defaulting those to the inference sidecar would send judge
      traffic somewhere it was never configured to go.
    * **Never overrides an explicit address.** An operator pointing one detector at a
      dedicated sidecar still wins.
    * **Requires ``task``**, since the address is per-task (``/v1/infer/{task}``). A backend
      with no task is left alone to fail loudly rather than be pointed at ``/v1/infer/None``.

    ``in_boundary`` is inferred safe-by-default from the resolved host, mirroring
    ``ml_catalog.default_strategy_for``: true only for genuine loopback (where the call never
    leaves the pod, so the egress gate correctly does not apply), false otherwise so an
    off-box sidecar is treated as a boundary crossing and gets allowlist / residency /
    redaction / fail-closed audit. An explicit ``in_boundary`` in the policy is preserved.
    """
    from znyx_core.engine.ml_catalog import _is_loopback, inference_url

    base: Optional[str] = None
    for backend in backends.values():
        if backend.mode not in ML_MODES or backend.endpoint_url or not backend.task:
            continue
        if base is None:
            base = inference_url()          # resolved once per strategy build
        backend.endpoint_url = f"{base}/v1/infer/{backend.task}"
        if backend.in_boundary is None:
            backend.in_boundary = _is_loopback(backend.endpoint_url)


def _inject_language_params(detector_config: Dict[str, Any],
                            backends: Dict[str, "DetectorBackend"]) -> None:
    """If a backend serves the ``language`` task, copy the detector policy's
    ``allowed_languages`` / ``blocked_languages`` into its ``params`` so the
    LanguageRunner can enforce the same policy as the deterministic detector."""
    lang_keys = {}
    al = detector_config.get("allowed_languages")
    if al is not None:
        lang_keys["allowed_languages"] = al
    bl = detector_config.get("blocked_languages")
    if bl is not None:
        lang_keys["blocked_languages"] = bl
    if not lang_keys:
        return
    for be in backends.values():
        if be.task == "language":
            be.params = {**(be.params or {}), **lang_keys}


def build_strategy(detector_config: Dict[str, Any],
                   runtime_policy: Optional[Dict[str, Any]] = None,
                   policy: Optional[Dict[str, Any]] = None,
                   org_default: Optional[Dict[str, Any]] = None) -> Optional[BackendStrategy]:
    """Build a BackendStrategy from a (raw, runtime) detector policy dict + the
    top-level runtime_policy (for the egress gate). ``policy`` is the full effective
    policy, read only for the org's PII/secrets configs so egress redaction matches
    the org's in-pipeline redaction. ``org_default`` (optional) is the org-level
    default {strategy, backends} for this detector — layered UNDER the policy: used only
    when the policy doesn't set its own ``strategy``.

    Returns None when no ``strategy`` is present (policy or org default) — the
    orchestrator then keeps the deterministic-only path (zero behaviour change)."""
    # Layer the org default under the policy: the policy's own strategy always wins.
    detector_config = layer_org_default(detector_config, org_default)
    strat = detector_config.get("strategy")
    if not isinstance(strat, dict):
        return None
    order = [m for m in (strat.get("order") or []) if m in _VALID_MODES]
    if not order:
        return None

    backends: Dict[str, DetectorBackend] = {}
    for mode, cfg in (detector_config.get("backends") or {}).items():
        if mode in _VALID_MODES and isinstance(cfg, dict):
            backends[mode] = DetectorBackend(mode=mode, **{f: cfg.get(f) for f in _BACKEND_FIELDS})

    # Resolve the co-located sidecar's address for ML modes that don't carry one. The
    # control plane deliberately ships no address (it doesn't know the deployment's
    # topology), so this is where an ML backend actually acquires one.
    _resolve_ml_endpoints(backends)

    # Auto-populate per-request params for the language runner: copy the detector
    # policy's allowed_languages / blocked_languages into the backend's params so the
    # ML runner enforces the same policy as the deterministic detector — no manual
    # config needed on the inference sidecar.
    _inject_language_params(detector_config, backends)

    runtime_policy = runtime_policy or {}
    rbe = detector_config.get("redact_before_egress")
    redact_pii = redact_secrets = False
    if isinstance(rbe, dict):
        redact_pii = rbe.get("pii", True)
        redact_secrets = rbe.get("secrets", True)

    policy = policy or {}
    pii_cfg = policy.get("pii") if isinstance(policy.get("pii"), dict) else None
    secrets_cfg = policy.get("secrets") if isinstance(policy.get("secrets"), dict) else None

    return BackendStrategy(
        order=order,
        escalate_when=strat.get("escalate_when") if isinstance(strat.get("escalate_when"), dict) else None,
        # strategy-level wins; fall back to the detector-level scalar.
        fallback=strat.get("fallback") or detector_config.get("fallback"),
        timeout_ms=strat.get("timeout_ms") or detector_config.get("timeout_ms"),
        additive=bool(strat.get("additive")),
        backends=backends,
        no_external_calls=bool(runtime_policy.get("no_external_calls")),
        allowed_regions=runtime_policy.get("allowed_regions"),
        egress_allowlist=detector_config.get("egress_allowlist"),
        redact_pii=redact_pii,
        redact_secrets=redact_secrets,
        redact_pii_config=pii_cfg,
        redact_secrets_config=secrets_cfg,
    )


def should_escalate(escalate_when: Optional[Dict[str, Any]], current_mode: str,
                    current_risk: Optional[float], current_confidence: Optional[float]) -> bool:
    """Decide whether to run the NEXT backend after ``current_mode``.

    - From the deterministic layer: if ``deterministic_score_between: [low, high]`` is
      set, escalate only when the deterministic risk score falls in that band.
    - From an ml/embedding layer: if ``ml_confidence_below: x`` is set, escalate only
      when the layer's confidence is known and below ``x``.
    - When no relevant predicate is configured, the ``order`` itself is the intent →
      escalate. (An empty escalate_when means "always run the next mode".)
    """
    if not escalate_when:
        return True

    if current_mode == _DETERMINISTIC:
        band = escalate_when.get("deterministic_score_between")
        if band is not None and len(band) == 2:
            low, high = band
            score = current_risk if current_risk is not None else 0
            return low <= score <= high
        return True

    # ml / embedding → next (llm)
    threshold = escalate_when.get("ml_confidence_below")
    if threshold is not None:
        # Fail SAFE: confidence is optional in the contract, so a missing/unparsed
        # confidence escalates to the stronger layer rather than silently stopping.
        return current_confidence is None or current_confidence < threshold
    return True
