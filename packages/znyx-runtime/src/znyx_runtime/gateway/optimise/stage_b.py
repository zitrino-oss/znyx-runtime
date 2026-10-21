"""Stage B techniques that change the request's limits, not its text.

- ``output_caps`` (6): the answer length a request asks for is capped at the key's
  ``max_output_tokens``; a session (or a trace the caller names) that keeps calling is
  stopped once it reaches the key's call or token allowance within a window, counted from
  the provider's real usage after each call.
- ``budget_degrade`` (7): when one of the key's budgets is nearly used, a tighter cap
  applies instead of the calls failing later.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

DEFAULT_LOOP_WINDOW_SECONDS = 600
DEFAULT_DEGRADE_AT_PERCENT = 80
DEFAULT_DEGRADE_CAP = 512


def _limit_fields(api: str, provider) -> tuple:
    """The fields that limit the answer length, in the order they are honoured."""
    if api == "anthropic":
        return ("max_tokens",)
    return ("max_completion_tokens", "max_tokens")


def apply_output_cap(api: str, provider, body: Dict[str, Any], cap: int) -> bool:
    """Cap the answer length the request asks for. Returns True when the request changed.
    A request naming no limit gets one: ``max_completion_tokens`` on OpenAI's own API
    (its reasoning models refuse ``max_tokens``), ``max_tokens`` elsewhere."""
    changed = False
    named = False
    for name in _limit_fields(api, provider):
        value = body.get(name)
        if value is None:
            continue
        named = True
        if isinstance(value, bool) or not isinstance(value, int) or value > cap:
            body[name] = cap
            changed = True
    if not named:
        host = (urlsplit(provider.base_url).hostname or "").lower()
        newer_azure = provider.format == "azure_openai" and str(provider.api_version or "") >= "2024-09-01"
        if api != "anthropic" and (host == "api.openai.com" or newer_azure):
            body["max_completion_tokens"] = cap         # reasoning models refuse max_tokens
        else:
            body["max_tokens"] = cap
        changed = True
    return changed


def degrade_cap(settings_cap: Optional[int], degrade_setting: Optional[int]) -> int:
    if degrade_setting:
        return degrade_setting
    if settings_cap:
        return max(1, settings_cap // 2)
    return DEFAULT_DEGRADE_CAP


def loop_key(caller: str, session: str) -> str:
    # Not "pin:": an idle session's counter may be evicted under pressure; it then starts over.
    return f"gw:loop:{caller}:{session[:128]}"


# -- PII tokenise and restore (30) ---------------------------------------------------
#
# The input guard reads the request as the client sent it, so every check (secrets
# included) sees the real text; only the PII check's own redaction changes, writing a
# placeholder per value (the PII detector's "token" strategy) instead of "[EMAIL]". The
# gateway keeps the values of this call (from the original text) and puts them back in the
# answer's plain text before the output guard reads it; never into tool-call arguments or
# inside a link, where a model told to by injected text could carry them elsewhere.

class PiiVault:
    """The placeholders of one call and the values they stand for. In memory, for this
    call only; never recorded."""

    def __init__(self) -> None:
        self.by_placeholder: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.by_placeholder)


def pii_applies_to_input(pii_config: Any) -> bool:
    """Whether the policy redacts PII in requests (only then may its redaction be made
    reversible: a policy that blocks PII, or does not check it, is left as it is)."""
    if not isinstance(pii_config, dict) or not pii_config.get("enabled", False):
        return False
    from znyx_core.detectors.pii import PIIDetector
    if PIIDetector._normalize_action(pii_config.get("action", "REDACT")) != "REDACT":
        return False
    stages = pii_config.get("stages")
    return not stages or "input" in stages


def tokenising_policy(policy: Dict[str, Any], token_key: str) -> Dict[str, Any]:
    """The policy for the input guard: the same checks, the PII check writing placeholders."""
    import copy
    out = copy.deepcopy(policy)
    out["pii"] = {**out.get("pii", {}), "redaction_strategy": "token", "token_key": token_key}
    return out


def build_vault(fmt, body: Dict[str, Any], check_roles, pii_config: Dict[str, Any], token_key: str) -> PiiVault:
    """The values the PII check will replace in this request, by placeholder (read from the
    request before the input guard changes it)."""
    from znyx_core.detectors.pii import PIIDetector, pii_token
    detector = PIIDetector({**pii_config, "enabled": True, "redaction_strategy": "token", "token_key": token_key})
    vault = PiiVault()
    for part in fmt.input_parts(body, check_roles):
        if part.kind not in ("text", "tool_result", "history_tool_call") or not part.text:
            continue
        result = detector.detect(part.text)
        for start, end, label in result.redaction_spans or []:
            value = part.text[start:end]
            vault.by_placeholder[pii_token(str(label), value, token_key)] = value
    return vault


_PLACEHOLDER = re.compile(r"\[\[[A-Z_]+_[0-9a-f]{8}\]\]")
_LINKISH = re.compile(r"[/:?=&%#@\\]")


def restore_pii(fmt, payload: Dict[str, Any], vault: PiiVault) -> int:
    """Put the values back in the answer's text, where a placeholder stands on its own (not
    inside a link, a path or a query string). Returns how many were restored."""
    if not len(vault):
        return 0
    restored = 0
    for part in fmt.output_parts(payload):
        if part.kind != "text" or "[[" not in part.text:
            continue
        text = part.text

        def put_back(match) -> str:
            nonlocal restored
            value = vault.by_placeholder.get(match.group(0))
            if value is None:
                return match.group(0)
            start, end = match.start(), match.end()
            while start > 0 and not text[start - 1].isspace():
                start -= 1
            while end < len(text) and not text[end].isspace():
                end += 1
            around = text[start:match.start()] + text[match.end():end]
            if _LINKISH.search(around.strip(".,;:!?()[]{}<>\"'*_`")):
                return match.group(0)          # inside a link or path: left as the placeholder
            restored += 1
            return value

        new_text = _PLACEHOLDER.sub(put_back, text)
        if new_text != text:
            fmt.set_text(payload, part, new_text)
    return restored


# -- risk-adaptive checking (32) ----------------------------------------------------

DEFAULT_RISK_THRESHOLD = 60
RISK_TTL_SECONDS = 3600
RISK_POINTS_BLOCK = 40
RISK_POINTS_FLAG = 10
_TIGHTEN = 0.8


def risk_key(caller: str, session: str) -> str:
    return f"gw:risk:{caller}:{session[:128]}"


def risk_points(record) -> int:
    """What one call adds to its session's risk: a block counts most, any other finding
    (a redaction, a warning) a little."""
    if record.outcome in ("blocked_input", "blocked_output"):
        return RISK_POINTS_BLOCK
    if record.rule_ids or record.input_decision != "ALLOW" or record.output_decision != "ALLOW":
        return RISK_POINTS_FLAG
    return 0


def stricter(policy: Dict[str, Any]) -> Dict[str, Any]:
    """The policy for a session whose risk climbed: block thresholds 20% lower (the
    jailbreak check's included), streams buffered so nothing is released before the whole
    answer is checked. Only ever tightens: nothing is switched off or raised."""
    import copy
    out = copy.deepcopy(policy)
    for name, section in out.items():
        if not isinstance(section, dict):
            continue
        for field in ("block_threshold",) + (("threshold",) if name == "jailbreak" else ()):
            value = section.get(field)
            if value is None and name == "jailbreak" and field == "threshold":
                value = 50                                  # the detector's default
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1:
                section[field] = max(1, int(value * _TIGHTEN))
    streaming = out.get("streaming") if isinstance(out.get("streaming"), dict) else {}
    out["streaming"] = {**streaming, "mode": "buffered"}
    return out


# -- parallel input check (40) -------------------------------------------------------

# Actions known never to change text. Anything else (REDACT, TRANSFORM, MASK, ANONYMIZE,
# RESTRICT, a value a detector reads as a redaction, or one this list does not know) counts
# as a possible rewrite.
_KEEPS_TEXT = frozenset({"BLOCK", "DENY", "WARN", "LOG", "ALLOW", "NONE", "ALLOW_WITH_NOTICE", "FLAG"})


def _keeps_text(action: Any, default: str) -> bool:
    return str(action if action is not None else default).strip().upper() in _KEEPS_TEXT


def input_may_be_rewritten(policy: Dict[str, Any]) -> bool:
    """Whether the input checks may change the text: then the request must not be sent
    before they finish, since the original would already have left. Only checks whose
    configured actions are all known to keep the text count as not rewriting."""
    from znyx_core.engine.orchestrator import _DETECTOR_PIPELINE
    defaults = {"pii": "REDACT", "secrets": "BLOCK", "competitor": "ALLOW_WITH_NOTICE"}
    for key, _default_enabled, _stages, can_transform in _DETECTOR_PIPELINE:
        section = policy.get(key)
        if not isinstance(section, dict) or not section.get("enabled", False):
            continue
        stages = section.get("stages")
        if stages and "input" not in stages:
            continue
        if can_transform and not _keeps_text(section.get("action"), defaults.get(key, "REDACT")):
            return True
        if not can_transform and section.get("action") is not None and not _keeps_text(section["action"], "BLOCK"):
            return True
        types = section.get("types")
        if isinstance(types, dict):
            for setting in types.values():
                if isinstance(setting, dict) and setting.get("action") is not None \
                        and not _keeps_text(setting["action"], "REDACT"):
                    return True
        if key == "pii" and section.get("patterns"):
            return True                      # custom patterns redact by default
    if policy.get("custom_detectors"):
        return True                          # a plug-in check may return changed text
    return False
