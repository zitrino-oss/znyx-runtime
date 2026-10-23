"""The usage event of one gateway call: what the runtime ships to the control plane for
observability and FinOps.

Metadata only, never text: who called (key, app, agent, end user as a pseudonym), which
provider and model, tokens as the provider reported them (or an estimate, flagged),
cost from the price table, timings, what the guardrails decided, which techniques ran,
where the call sits in a trace, and what the optimisation layer saved (the ledger). One
flat record with scalar fields (plus small lists and one small map), so it fits a row
store with rollups now and a columnar store later without reshaping.

The runtime builds events with ``build``; the control plane accepts them with
``validate``, which never trusts the sender: it keeps only known fields, coerces types,
bounds every string and list, and refuses events it cannot attribute.
"""
from __future__ import annotations

import hashlib
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# field -> kind. str fields are bounded to MAX_STR characters.
FIELDS: Dict[str, str] = {
    "schema": "int",
    "event_id": "str",              # unique per call: the control plane drops repeats
    "occurred_at": "time",          # when the call started (UTC)
    "request_id": "str",
    "trace_id": "hex32",            # W3C trace id of the call's trace
    "span_id": "hex16",             # the call's own span
    "parent_span_id": "hex16",      # the caller's span, when it sent traceparent
    "environment_id": "str",        # the bundle-set member (an environment id) when known
    "project_id": "str",
    "tenant_id": "str",
    "app_id": "str",
    "agent_id": "str",
    "key_id": "str",                # the gateway key's id (never the key)
    "user": "str",                  # the end user, pseudonymised by default
    "api": "str",                   # openai | azure | anthropic | mcp
    "provider": "str",              # the provider's configured name
    "provider_format": "str",       # openai | azure_openai | anthropic
    "model": "str",                 # as requested
    "response_model": "str",        # as the provider reported serving
    "stream": "bool",
    "status": "int",
    "outcome": "str",               # ok | blocked_input | blocked_output | denied | error
    "reason": "str",
    "input_decision": "str",
    "output_decision": "str",
    "rule_ids": "strlist",
    "input_tokens": "int",
    "output_tokens": "int",
    "cached_input_tokens": "int",
    "cache_write_tokens": "int",
    "usage_estimated": "bool",      # tokens estimated: the provider's count never arrived
    "cost_usd": "float",
    "price_known": "bool",
    "latency_ms": "int",
    "upstream_ms": "int",
    "ttft_ms": "int",
    "guard_ms": "int",
    "techniques": "countmap",       # technique -> how many times it acted
    # The optimisation ledger (Phase 4). saved_usd may be negative (a cache written and
    # never read); baseline_cost_usd is what the call would have cost unoptimised.
    "profile": "str",
    "saved_usd": "sfloat",
    "baseline_cost_usd": "float",
    "optimisations": "effects",     # [{technique, stage, saved_input_tokens, saved_usd, ...}]
    # Phase 5: tool names (never arguments), for the AI inventory.
    "tools_offered": "strlist",     # the tools the model was offered
    "tools_called": "strlist",      # the tools it called in its answer
    "mcp_tools_changed": "strlist",  # MCP tools held because their definition changed: "tool@sha256"
    "mcp_tools_seen": "strlist",     # every tool an MCP server listed: "tool@sha256"
    # Phase 5 part 3: the call ran for a person a delegated token names (``user`` is that person).
    "delegated": "bool",
    "identity_provider": "str",
    # Phase 5 part 4: which version of the instructions ran (never the text).
    "prompt_hash": "hex16",         # prompt_fingerprint of the system and developer instructions
    "prompt_ref": "str",            # the caller's own name for them (X-Znyx-Prompt: name@version)
}

MAX_STR = 200
MAX_LIST = 50
MAX_MAP = 30
# Per-event ceilings: far above any real call, low enough that sums over millions of
# events never overflow a rollup column.
_CAPS = {"input_tokens": 10_000_000, "output_tokens": 10_000_000, "cached_input_tokens": 10_000_000,
         "cache_write_tokens": 10_000_000, "latency_ms": 86_400_000, "upstream_ms": 86_400_000,
         "ttft_ms": 86_400_000, "guard_ms": 86_400_000, "status": 999}
MAX_COST_USD = 100_000.0
MAX_COUNT = 1_000_000
MAX_EFFECTS = 20
MAX_PROFILE = 40
_TECHNIQUE_ID = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_EFFECT_FIELDS = ("technique", "stage", "saved_input_tokens", "saved_usd", "latency_ms", "verified",
                  "estimated")
# Characters a database text column refuses (NUL) or that cannot be encoded (lone surrogates).
_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")
OUTCOMES = ("ok", "blocked_input", "blocked_output", "denied", "error")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")


_PROMPT_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,99}(@[A-Za-z0-9][A-Za-z0-9._+-]{0,39})?$")


def prompt_fingerprint(texts: List[str], key: Optional[bytes] = None, scope: str = "") -> Optional[str]:
    """The first 16 hex of a hash of the instructions, whitespace normalised (runs of whitespace
    become one space, ends trimmed), so a reformatted prompt keeps its fingerprint and any change
    of wording gets a new one. With ``key`` (the runtime's user-id key) it is an HMAC, so whoever
    sees fingerprints cannot confirm a guessed prompt; replicas sharing the key agree. None when
    there are no instructions."""
    joined = " ".join(" ".join(t.split()) for t in texts if isinstance(t, str))
    joined = " ".join(joined.split())
    if not joined:
        return None
    data = joined.encode("utf-8", "surrogatepass")
    if key:
        import hmac
        return hmac.new(key, scope.encode("utf-8", "surrogatepass") + b"\x1fprompt\x1f" + data,
                        hashlib.sha256).hexdigest()[:16]
    return hashlib.sha256(data).hexdigest()[:16]


def prompt_ref(value: Optional[str]) -> Optional[str]:
    """The caller's name for its prompt (``name@version``), or None when it is not one."""
    value = (value or "").strip()
    return value if _PROMPT_REF.match(value) else None


def new_trace_id() -> str:
    return uuid.uuid4().hex


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def parse_traceparent(value: Optional[str]):
    """``(trace_id, parent_span_id)`` from a W3C ``traceparent`` header, or None."""
    if not value or not isinstance(value, str):
        return None
    parts = value.strip().lower().split("-")
    if len(parts) < 4 or parts[0] == "ff" or len(parts[0]) != 2:
        return None
    trace_id, span_id = parts[1], parts[2]
    if not _HEX32.match(trace_id) or not _HEX16.match(span_id):
        return None
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id, span_id


def traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def build(**fields: Any) -> Dict[str, Any]:
    """An event from the runtime's call record (unknown fields dropped)."""
    event = {name: (clean_str(value) if FIELDS[name] == "str" and isinstance(value, str) else value)
             for name, value in ((n, fields.get(n)) for n in FIELDS) if value is not None}
    event["schema"] = SCHEMA_VERSION
    event.setdefault("event_id", uuid.uuid4().hex)
    event.setdefault("occurred_at", datetime.now(timezone.utc).isoformat())
    return event


def clean_str(value: Any) -> str:
    """A string any store accepts: NUL and lone surrogates removed, bounded."""
    return _UNSTORABLE.sub("", str(value))[:MAX_STR]


def _coerce(kind: str, value: Any, cap: int = 2 ** 53):
    if value is None:
        return None
    if kind == "str":
        return clean_str(value)
    if kind == "hex32":
        value = str(value).lower()
        return value if _HEX32.match(value) else None
    if kind == "hex16":
        value = str(value).lower()
        return value if _HEX16.match(value) else None
    if kind == "int":
        if isinstance(value, bool):
            return int(value)
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0, min(number, cap))
    if kind == "float":
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and 0 <= number <= MAX_COST_USD else None
    if kind == "sfloat":
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and abs(number) <= MAX_COST_USD else None
    if kind == "bool":
        return value is True
    if kind == "time":
        try:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    if kind == "strlist":
        if not isinstance(value, list):
            return None
        return [clean_str(v) for v in value[:MAX_LIST]]
    if kind == "countmap":
        if not isinstance(value, dict):
            return None
        out = {}
        for k, v in list(value.items())[:MAX_MAP]:
            number = _coerce("int", v, MAX_COUNT)
            if number is not None:
                out[clean_str(k)] = number
        return out
    if kind == "effects":
        return _effects(value)
    return None


def _effects(value: Any) -> Optional[List[Dict[str, Any]]]:
    """The ledger entries: one per technique (a repeat is dropped), known fields only,
    numbers bounded."""
    if not isinstance(value, list):
        return None
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in value[:MAX_EFFECTS]:
        if not isinstance(item, dict):
            continue
        technique = item.get("technique")
        if not isinstance(technique, str) or not _TECHNIQUE_ID.fullmatch(technique) or technique in seen:
            continue
        seen.add(technique)
        entry: Dict[str, Any] = {
            "technique": technique,
            "stage": item.get("stage") if item.get("stage") in ("A", "B", "C") else "",
            "saved_input_tokens": _coerce("int", item.get("saved_input_tokens"), 10_000_000) or 0,
            "estimated": item.get("estimated") is True,
        }
        saved = _coerce("sfloat", item.get("saved_usd"))
        if saved is not None:
            entry["saved_usd"] = saved
        latency = item.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool) and math.isfinite(latency):
            entry["latency_ms"] = int(max(-86_400_000, min(int(latency), 86_400_000)))
        if item.get("verified") in ("pass", "fail"):
            entry["verified"] = item["verified"]
        if item.get("shadow") is True:
            _shadow_fields(item, entry)
        out.append(entry)
    return out


def _shadow_fields(item: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """A shadow test's verdict: numbers and fixed words only."""
    entry["shadow"] = True
    agreement = _coerce("sfloat", item.get("agreement"))
    if agreement is not None:
        entry["agreement"] = max(0.0, min(1.0, agreement))
    projected = _coerce("sfloat", item.get("projected_saved_usd"))
    if projected is not None:
        entry["projected_saved_usd"] = projected
    if item.get("routed") in ("small", "stay"):
        entry["routed"] = item["routed"]
    if item.get("method") in ("nli", "lexical"):
        entry["method"] = item["method"]
    features = item.get("features")
    if isinstance(features, list):
        values = []
        for v in features[:16]:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                values.append(max(-1e6, min(1e6, float(v))))
        if values:
            entry["features"] = values


def validate(event: Any) -> Optional[Dict[str, Any]]:
    """The event as the control plane may store it, or None when it is unusable (not a
    mapping, an unknown schema, no id or time). Strings bounded, numbers clamped,
    unknown fields dropped, ``occurred_at`` a timezone-aware datetime."""
    if not isinstance(event, dict) or event.get("schema") != SCHEMA_VERSION:
        return None
    out: Dict[str, Any] = {}
    for name, kind in FIELDS.items():
        value = _coerce(kind, event.get(name), _CAPS.get(name, 2 ** 53))
        if value is not None:
            out[name] = value
    if not out.get("event_id") or not out.get("occurred_at"):
        return None
    if out.get("outcome") not in OUTCOMES:
        out["outcome"] = "error" if out.get("status", 0) >= 400 else "ok"
    if "profile" in out:
        out["profile"] = out["profile"][:MAX_PROFILE]
    _settle_ledger(out)
    return out


def _settle_ledger(out: Dict[str, Any]) -> None:
    """Make the ledger agree with itself whatever the sender wrote: the call's saving is
    the sum of its techniques', and its baseline is its cost plus that saving."""
    effects = out.get("optimisations")
    if not effects:
        out.pop("optimisations", None)
        out.pop("saved_usd", None)
        out.pop("baseline_cost_usd", None)
        return
    savings = [e["saved_usd"] for e in effects if "saved_usd" in e]
    if out.get("cost_usd") is None:
        savings = []                         # a saving is only shown against a known cost
    if savings:
        total = round(sum(savings), 8)
        cost = out.get("cost_usd")
        # A call cannot have cost less than nothing without optimisation: a loss is at most
        # the call's own cost, so baseline = cost + saved always holds and is never negative.
        floor = -cost if cost is not None else -MAX_COST_USD
        out["saved_usd"] = max(floor, min(total, MAX_COST_USD - (cost or 0.0)))
    else:
        out.pop("saved_usd", None)
    if out.get("cost_usd") is not None and "saved_usd" in out:
        out["baseline_cost_usd"] = round(out["cost_usd"] + out["saved_usd"], 8)
    else:
        out.pop("baseline_cost_usd", None)


def total_tokens(event: Dict[str, Any]) -> int:
    return int(event.get("input_tokens") or 0) + int(event.get("output_tokens") or 0)


def columns() -> List[str]:
    """The field names, in schema order (a columnar store's column list)."""
    return list(FIELDS)
