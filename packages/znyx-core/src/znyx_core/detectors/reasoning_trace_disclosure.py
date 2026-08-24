"""Reasoning-trace disclosure (OWASP LLM02 - Sensitive Information Disclosure).

2026 widened LLM02's channel list past the final answer: "tool-call arguments, reasoning
traces, retrieved chunks, multimodal output, logs, telemetry, embeddings ... are all
disclosure surfaces. Treat each as an output subject to the same classification and
redaction rules." The entry is equally direct about the operational failure — "treat
reasoning traces and tool arguments as outputs, not debugging leftovers" — and gives the
scenario: extended-thinking traces logged verbatim to a shared APM project exposing
retrieved PII to hundreds of engineers while the answer itself stays sanitised.

The gap is structural rather than lexical. PII, secrets, and the other LLM02 detectors
already know how to recognise sensitive content; they were simply never pointed at the
trace, because the trace is not the answer. So this detector does not re-implement any of
that. It flags the two conditions that let a trace leak:

* **The trace was never inspected.** A trace is present and the caller has not marked it
  as scanned. That is the shared-APM scenario exactly: the answer went through the output
  pipeline and the trace went straight to a log.
* **The trace says more than the answer.** The answer is sanitised while the trace still
  carries what was redacted out of it — the specific failure LLM02 describes, where the
  visible response looks clean and the trace does not.

It also flags raw tool-call ARGUMENTS travelling alongside, for the same reason: 2026
names them as an output, and they routinely carry the record identifiers and query
parameters that the answer deliberately omits.

    metadata = {
        "reasoning_trace": "...",       # or thinking / trace / reasoning_content
        "trace_scanned": True,          # set by a caller that ran it through evaluation
        "tool_calls": [{"name": "...", "arguments": {...}}],
    }
"""
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

_TRACE_KEYS = ("reasoning_trace", "thinking", "reasoning", "reasoning_content",
               "trace", "chain_of_thought", "thought")
_SCANNED_FLAGS = ("trace_scanned", "reasoning_scanned", "trace_evaluated")
_TOOL_CALL_KEYS = ("tool_calls", "function_calls", "tool_invocations")
# Observation-time side channels. 2026 names token length, latency, and
# log-probabilities as disclosure surfaces in their own right: a caller who can read the
# per-token distribution can recover which of several candidate continuations the hidden
# context made likely, without the answer ever stating it. Membership-inference and
# system-prompt-recovery attacks both run on exactly this signal.
_LOGPROB_KEYS = ("logprobs", "log_probs", "logprob", "token_logprobs")
_TOP_LOGPROB_KEYS = ("top_logprobs", "top_k_logprobs")
_TIMING_KEYS = ("token_timings", "per_token_latency", "token_latencies")
# Where a provider nests the per-token detail.
_RESPONSE_BLOCKS = ("response", "raw_response", "completion", "usage")
_ENV_KEYS = ("env", "environment", "deployment_env")
# Environments where the audience is not the developer who asked for the detail.
_DEFAULT_GATED_ENVS = ("prod", "production", "live")


def _iter_channel_scopes(metadata: Dict[str, Any]):
    """The metadata dicts a provider may hide per-token detail in: the top level, the
    usual response wrappers, and each choice of an OpenAI-shaped response."""
    yield metadata
    for key in _RESPONSE_BLOCKS:
        block = metadata.get(key)
        if isinstance(block, dict):
            yield block
    for holder in (metadata, *(metadata.get(k) for k in _RESPONSE_BLOCKS)):
        if not isinstance(holder, dict):
            continue
        choices = holder.get("choices")
        if isinstance(choices, list):
            for choice in choices[:8]:
                if isinstance(choice, dict):
                    yield choice


def _present(scope: Dict[str, Any], keys) -> bool:
    for k in keys:
        v = scope.get(k)
        if v not in (None, {}, [], ""):
            return True
    return False


def _logprob_channels(metadata: Optional[Dict[str, Any]]) -> List[str]:
    """Which observation-time channels this response carries."""
    if not isinstance(metadata, dict):
        return []
    found: List[str] = []
    for scope in _iter_channel_scopes(metadata):
        if not isinstance(scope, dict):
            continue
        if "top_logprobs" not in found and _present(scope, _TOP_LOGPROB_KEYS):
            found.append("top_logprobs")
        if "logprobs" not in found and _present(scope, _LOGPROB_KEYS):
            found.append("logprobs")
        if "token_timings" not in found and _present(scope, _TIMING_KEYS):
            found.append("token_timings")
    return found
# Markers the output pipeline leaves behind when it has redacted something.
_REDACTION_MARKERS = ("[REDACTED]", "[PII]", "[EMAIL]", "[PHONE]", "[SSN]",
                      "[CARD]", "[SECRET]", "[MASKED]", "███")


def _first_trace(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    for k in _TRACE_KEYS:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            return v
        # Some providers ship the trace as a list of thought blocks.
        if isinstance(v, list):
            parts = [p for p in v if isinstance(p, str)]
            if parts:
                return "\n".join(parts)
    return None


def _first_str_env(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    for k in _ENV_KEYS:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _tool_arguments(metadata: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(metadata, dict):
        return []
    out: List[str] = []
    for k in _TOOL_CALL_KEYS:
        calls = metadata.get(k)
        if not isinstance(calls, list):
            continue
        for call in calls[:32]:
            if not isinstance(call, dict):
                continue
            # Two shapes in the wild. OpenAI and every SDK modelled on it nest the
            # arguments under "function"; Anthropic and the raw-tool shapes put them at
            # the top level as "arguments"/"input". Reading only the flat key missed the
            # single most common wire format.
            fn = call.get("function")
            candidates = [call.get("arguments"), call.get("input"), call.get("parameters")]
            if isinstance(fn, dict):
                candidates.extend([fn.get("arguments"), fn.get("input"),
                                   fn.get("parameters")])
            for args in candidates:
                if args is not None and args != "" and args != {}:
                    out.append(str(args))
                    break
    return out


class ReasoningTraceDisclosureDetector:
    """Flags reasoning traces and tool arguments escaping the output controls (LLM02)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        # WARN by default: the finding is usually a pipeline wiring problem, and blocking
        # an otherwise-clean answer because its trace was not scanned punishes the user
        # for the operator's plumbing.
        self.action = (self.config.get("action") or "WARN").upper()
        self.require_trace_scanned = bool(self.config.get("require_trace_scanned", True))
        self.flag_tool_arguments = bool(self.config.get("flag_tool_arguments", True))
        # Minimum trace length worth flagging; a two-word trace carries nothing.
        self.min_trace_chars = max(0, int(self.config.get("min_trace_chars", 40)))
        # Observation-time side channels (LLM02, 2026). Gated by ENVIRONMENT rather than
        # banned outright: log-probabilities are a legitimate debugging and evaluation
        # tool, and the objection is to shipping them on an endpoint whose audience is
        # not the developer who asked for them.
        self.flag_logprobs = bool(self.config.get("flag_logprobs", True))
        gated = self.config.get("logprob_gated_envs") or _DEFAULT_GATED_ENVS
        self.logprob_gated_envs = {str(e).strip().lower() for e in gated if str(e).strip()}

    def detect(self, text: str,
               metadata: Optional[Dict[str, Any]] = None,
               env: Optional[str] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        trace = _first_trace(metadata)
        rule_hits: List[RuleHit] = []

        if trace and len(trace) >= self.min_trace_chars:
            scanned = any(bool((metadata or {}).get(f)) for f in _SCANNED_FLAGS)
            if self.require_trace_scanned and not scanned:
                rule_hits.append(RuleHit(
                    rule_id="reasoning_trace_disclosure.unscanned_trace",
                    severity=Severity.HIGH,
                    message=("A reasoning trace accompanies this response and has not been "
                             "evaluated; 2026 treats the trace as an output, not a debug artefact"),
                ))

            # The answer was redacted and the trace was not: whatever the output pipeline
            # decided to remove is still sitting in the trace.
            answer_redacted = any(m in (text or "") for m in _REDACTION_MARKERS)
            trace_redacted = any(m in trace for m in _REDACTION_MARKERS)
            if answer_redacted and not trace_redacted:
                rule_hits.append(RuleHit(
                    rule_id="reasoning_trace_disclosure.trace_exceeds_answer",
                    severity=Severity.HIGH,
                    message=("The answer was redacted but its reasoning trace was not; the "
                             "trace still carries what the response removed"),
                ))

        if self.flag_logprobs:
            channels = _logprob_channels(metadata)
            if channels:
                resolved_env = (env or _first_str_env(metadata) or "").strip().lower()
                # An unknown environment is treated as gated. A response carrying a token
                # distribution with no idea where it is going is the case to be loud about,
                # not the one to wave through.
                gated = (not resolved_env) or resolved_env in self.logprob_gated_envs
                if gated:
                    where = resolved_env or "an unidentified environment"
                    rule_hits.append(RuleHit(
                        rule_id="reasoning_trace_disclosure.logprobs_exposed",
                        severity=Severity.HIGH if "top_logprobs" in channels else Severity.MEDIUM,
                        message=(f"Response carries {', '.join(channels)} on {where}; a "
                                 f"per-token distribution leaks what the hidden context "
                                 f"made likely without the answer ever saying it"),
                    ))

        if self.flag_tool_arguments:
            args = _tool_arguments(metadata)
            if args:
                rule_hits.append(RuleHit(
                    rule_id="reasoning_trace_disclosure.raw_tool_arguments",
                    severity=Severity.MEDIUM,
                    message=(f"{len(args)} tool-call argument set(s) travel with this response; "
                             f"2026 classifies tool arguments as an output channel"),
                ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"reasoning_trace_disclosure: {', '.join(sorted({h.rule_id for h in rule_hits}))}"
        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="The response was withheld pending review of its reasoning trace.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
