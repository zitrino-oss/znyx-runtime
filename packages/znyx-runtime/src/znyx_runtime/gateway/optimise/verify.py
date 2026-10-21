"""Checking an answer before it is returned, and asking again once when it fails
(stage B, techniques 15 to 19, 21 and 22).

- An output contract comes from the policy (``output_contract`` with a JSON schema and
  optional ``rules``, or a ``structure`` / ``schema_enforcement`` schema); the pipeline
  reads it and hands it here, so these checks never see the policy itself.
- ``structured_output`` (15): OpenAI is asked for schema-constrained output when the app
  set none; every answer is validated against the contract.
- ``repair`` (16): a simple fault (a code fence, prose around the JSON, trailing commas)
  is fixed in place; otherwise the model is asked again once with the errors.
- ``grounded_retry`` (17): the request's documents, search results and tool results are
  the grounding sources; an answer they do not support is asked again once.
- ``computed_claims`` (18): wrong arithmetic, percentages, conversions and dates
  (``znyx_core.engine.claims``) are asked again once with the corrections.
- ``business_rules`` (19): rules on the answer (allowed values, ranges, required or
  forbidden text) checked exactly; asked again once on failure.
- ``escalate`` (21): an answer still failing is tried once on the key's stronger model.
- ``voting`` (22): several answers for a high-stakes call; the one most agree with wins.

Retries change only what the model is asked, never a check: the final answer still goes
through the output guard. Every retry is paid for and shown in the ledger.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from znyx_runtime.gateway import formats as fmts

MAX_SOURCES = 50
MAX_SOURCE_CHARS = 200_000
MAX_PROBLEMS = 8
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# Models that accept response_format json_schema on OpenAI's own API (and older or special
# ones that do not).
_SCHEMA_MODELS = ("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "o1", "o3", "o4")
_NO_SCHEMA_MODELS = ("gpt-4o-2024-05-13", "o1-mini", "o1-preview", "gpt-4o-audio", "gpt-4o-realtime",
                     "gpt-4o-search")


@dataclass
class Contract:
    json_schema: Optional[Dict[str, Any]]
    name: str = "answer"
    rules: List[Dict[str, Any]] = field(default_factory=list)
    on_fail: Optional[str] = None


@dataclass
class Faults:
    """What is wrong with an answer, by the technique that checks it."""
    by_technique: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, technique: str, problems: List[str]) -> None:
        if problems:
            self.by_technique.setdefault(technique, []).extend(problems[:MAX_PROBLEMS])

    def __bool__(self) -> bool:
        return bool(self.by_technique)

    def problems(self) -> List[str]:
        return [p for items in self.by_technique.values() for p in items][:MAX_PROBLEMS]


def contract_of(policy: Dict[str, Any]) -> Optional[Contract]:
    oc = policy.get("output_contract")
    if isinstance(oc, dict) and oc.get("enabled") and (oc.get("json_schema") or oc.get("rules")):
        rules = [r for r in (oc.get("rules") or []) if isinstance(r, dict)]
        return Contract(oc.get("json_schema") if isinstance(oc.get("json_schema"), dict) else None,
                        str(oc.get("schema_name") or "answer"), rules, oc.get("on_fail"))
    for name in ("structure", "schema_enforcement"):
        section = policy.get(name)
        schema = section.get("json_schema") or section.get("schema") if isinstance(section, dict) else None
        if isinstance(section, dict) and section.get("enabled") and isinstance(schema, dict):
            return Contract(schema, str(section.get("schema_name") or "answer"))
    return None


# -- the answer's text ------------------------------------------------------------

def answer_text(fmt, payload: Dict[str, Any]) -> Optional[str]:
    """The one text answer, or None when the answer is a tool call, has several choices
    or carries no text (those are not verified)."""
    if fmt is fmts.AnthropicMessages:
        blocks = payload.get("content") or []
        if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks):
            return None
        texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(t for t in texts if isinstance(t, str)) if texts else None
    choices = payload.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message") or {}
    if message.get("tool_calls") or not isinstance(message.get("content"), str):
        return None
    return message["content"]


def choice_texts(fmt, payload: Dict[str, Any]) -> List[Tuple[int, Optional[str]]]:
    """Each answer choice's text (None for a choice that is only a tool call)."""
    if fmt is fmts.AnthropicMessages:
        return [(0, answer_text(fmt, payload))]
    out: List[Tuple[int, Optional[str]]] = []
    for i, choice in enumerate(payload.get("choices") or []):
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        out.append((i, content if isinstance(content, str) and not message.get("tool_calls") else None))
    return out


def set_answer_text(fmt, payload: Dict[str, Any], text: str) -> None:
    if fmt is fmts.AnthropicMessages:
        blocks = payload.get("content") or []
        out, placed = [], False
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                if not placed:
                    out.append({"type": "text", "text": text})
                    placed = True
                continue
            out.append(block)
        payload["content"] = out if placed else out + [{"type": "text", "text": text}]
        return
    payload["choices"][0]["message"]["content"] = text


# -- checks -----------------------------------------------------------------------

def check_contract(text: str, contract: Contract) -> List[str]:
    if not contract.json_schema:
        return []
    from znyx_core.detectors.structure import StructureDetector
    result = StructureDetector({"enabled": True, "action": "BLOCK", "expected_format": "json",
                                "json_schema": contract.json_schema}).detect(text)
    problems = [f"{e.path or '/'}: {e.message}" for e in result.field_errors]
    if not problems and result.rule_hits:
        problems = [h.message for h in result.rule_hits]
    return problems


def simple_repair(text: str) -> Optional[str]:
    """The answer with a simple JSON fault fixed (a code fence, prose around the JSON,
    trailing commas), or None when that does not make it parse."""
    candidate = text
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end > start:
            candidate = candidate[start:end + 1]
    candidate = _TRAILING_COMMA.sub(r"\1", candidate)
    try:
        json.loads(candidate)
    except (ValueError, RecursionError):
        return None
    return candidate if candidate != text else None


def _at(value: Any, path: str) -> List[Any]:
    """Values at a dotted path ("items.*.price", "status", "rows.0")."""
    found = [value]
    for step in [s for s in str(path).split(".") if s]:
        nxt = []
        for item in found:
            if step == "*" and isinstance(item, list):
                nxt.extend(item)
            elif isinstance(item, dict) and step in item:
                nxt.append(item[step])
            elif isinstance(item, list) and step.isdigit() and int(step) < len(item):
                nxt.append(item[int(step)])
        found = nxt
    return found


def check_rules(text: str, rules: List[Dict[str, Any]]) -> List[str]:
    """Rules on the answer: ``{path, allowed}``, ``{path, min, max}``, ``{required_text}``,
    ``{forbidden_text}`` (text rules ignore case)."""
    problems: List[str] = []
    lowered = text.lower()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        parsed = None
    for rule in rules[:50]:
        if isinstance(rule.get("required_text"), str) and rule["required_text"].lower() not in lowered:
            problems.append(f'the answer must include "{rule["required_text"][:200]}"')
        if isinstance(rule.get("forbidden_text"), str) and rule["forbidden_text"].lower() in lowered:
            problems.append(f'the answer must not include "{rule["forbidden_text"][:200]}"')
        path = rule.get("path")
        if not isinstance(path, str) or parsed is None:
            continue
        values = _at(parsed, path)
        if not values and rule.get("required", True) and ("allowed" in rule or "min" in rule or "max" in rule):
            problems.append(f"{path} is missing")
        for value in values[:100]:
            if isinstance(rule.get("allowed"), list) and value not in rule["allowed"]:
                allowed = ", ".join(json.dumps(v) for v in rule["allowed"][:20])
                problems.append(f"{path} is {json.dumps(value)[:80]}; allowed: {allowed}")
            number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            if "min" in rule or "max" in rule:
                if number is None:
                    problems.append(f"{path} must be a number")
                elif isinstance(rule.get("min"), (int, float)) and number < rule["min"]:
                    problems.append(f"{path} is {number}; the minimum is {rule['min']}")
                elif isinstance(rule.get("max"), (int, float)) and number > rule["max"]:
                    problems.append(f"{path} is {number}; the maximum is {rule['max']}")
    return problems[:MAX_PROBLEMS]


def check_claims(text: str) -> List[str]:
    from znyx_core.engine.claims import find_wrong_claims
    return [c.describe() for c in find_wrong_claims(text, limit=MAX_PROBLEMS)]


def grounding_sources(fmt, body: Dict[str, Any]) -> List[Dict[str, str]]:
    """The documents, search results and tool results the request carries, as the
    grounding sources of the answer (bounded)."""
    out: List[Dict[str, str]] = []
    total = 0
    for part in fmt.input_parts(body, ("system", "developer", "user", "tool", "function")):
        is_document = fmt is fmts.AnthropicMessages and _in_source_block(body, part.path)
        if part.kind != "tool_result" and not is_document:
            continue
        if not part.text.strip():
            continue
        text = part.text[:MAX_SOURCE_CHARS - total]
        out.append({"source_id": f"{part.tool_name or 'document'}-{len(out) + 1}", "text": text})
        total += len(text)
        if len(out) >= MAX_SOURCES or total >= MAX_SOURCE_CHARS:
            break
    return out


def _in_source_block(body: Dict[str, Any], path) -> bool:
    node: Any = body
    for step in path[:-1]:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return False
        if isinstance(node, dict) and node.get("type") in ("document", "search_result"):
            return True
    return False


def check_grounded(text: str, sources: List[Dict[str, str]], config: Dict[str, Any], nli_scorer) -> List[str]:
    """Claims of the answer its sources do not support (the policy's hallucination
    settings, NLI when it has an endpoint, else token overlap)."""
    if not sources:
        return []
    from znyx_core.detectors.hallucination import HallucinationDetector
    settings = {k: v for k, v in (config or {}).items() if k not in ("enabled", "action", "stages")}
    settings.update({"enabled": True, "action": "WARN", "grounding_sources": sources,
                     "source_context": "\n\n".join(s["text"] for s in sources)})
    detector = HallucinationDetector(settings)
    if nli_scorer is not None:
        detector.nli_scorer = nli_scorer
    result = detector.detect(text)
    if result.decision is None or result.decision.value == "ALLOW":
        return []
    return [h.message for h in result.rule_hits][:MAX_PROBLEMS]


# -- asking again -----------------------------------------------------------------

def retry_body(fmt, body: Dict[str, Any], answer: str, faults: Faults) -> Optional[Dict[str, Any]]:
    """The request again, with the failed answer and what was wrong with it, or None when
    the conversation cannot take another turn."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") == "assistant":
        return None                       # an assistant prefill: a further turn would not parse
    lines = ["Your previous answer did not pass these checks:"]
    lines += [f"- {p}" for p in faults.problems()]
    if "grounded_retry" in faults.by_technique:
        lines.append("Answer only from the documents and tool results in this conversation, and say so "
                     "where they do not contain the answer.")
    if "structured_output" in faults.by_technique:
        lines.append("Reply with JSON that matches the required schema and nothing else.")
    lines.append("Reply again with the corrected answer only.")
    out = copy.deepcopy(body)
    out.pop("stream", None)
    out.pop("stream_options", None)
    if fmt is fmts.AnthropicMessages:
        out["messages"] = messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]},
                                      {"role": "user", "content": "\n".join(lines)}]
    else:
        out["messages"] = messages + [{"role": "assistant", "content": answer},
                                      {"role": "user", "content": "\n".join(lines)}]
    return out


def request_schema(provider, body: Dict[str, Any], contract: Optional[Contract], model: Optional[str]) -> bool:
    """Technique 15 on the request: ask OpenAI for output that follows the contract's
    schema, when the app asked for no format and the model supports it."""
    if contract is None or not contract.json_schema or "response_format" in body:
        return False
    if provider.format != "openai" or (urlsplit(provider.base_url).hostname or "").lower() != "api.openai.com":
        return False
    name = (model or "").split("/")[-1].lower()
    if not name.startswith(_SCHEMA_MODELS) or name.startswith(_NO_SCHEMA_MODELS):
        return False
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", contract.name)[:64] or "answer"
    body["response_format"] = {"type": "json_schema",
                               "json_schema": {"name": safe, "schema": contract.json_schema, "strict": False}}
    return True


# -- voting -----------------------------------------------------------------------

def _normal(text: str) -> str:
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except (ValueError, RecursionError):
        return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _similar(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def choose_by_vote(texts: List[Optional[str]]) -> Tuple[int, float]:
    """The answer most others agree with, and the share of answers that agree with it
    (identical once normalised, or at least 80% the same words)."""
    usable = [(i, _normal(t)) for i, t in enumerate(texts) if t is not None]
    if not usable:
        return 0, 0.0
    best, best_votes = usable[0][0], -1
    for i, norm in usable:
        votes = sum(1 for _, other in usable if other == norm or _similar(norm, other) >= 0.8)
        if votes > best_votes:
            best, best_votes = i, votes
    return best, best_votes / len(texts)
