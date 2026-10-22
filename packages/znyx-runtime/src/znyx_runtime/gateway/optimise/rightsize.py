"""Model right-sizing (technique 14): simple requests go to a smaller model the key
allows; hard or uncertain ones stay on the model the app asked for.

Rules first (symbolic): a short request asking for a short answer goes small; a long one
stays. In between, a difficulty score: a logistic model over numbers computed from the
request (never its text), with score bands like ``should_escalate``: confidently easy
goes small, anything else stays. The default model is a hand-set prior; the control plane
fits a better one on shadow results (technique 23) and delivers it in the signed bundle.

Who may not be right-sized is decided by the pipeline: streamed calls, agent steps, calls
whose input checks found something, and apps the policy marks critical.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.optimisation import RIGHT_SIZE_FEATURES as FEATURES

DEFAULT_MODEL: Dict[str, Any] = {
    "version": "default-1",
    "bias": 6.0,
    "weights": {"user_chars": -0.45, "prompt_chars": -0.15, "questions": -0.3, "code": -1.5, "maths": -0.8,
                "reasoning": -0.9, "constraints": -0.25, "turns": -0.1, "answer_tokens": -0.25,
                "structured": 0.3},
    "easy_at": 0.8,           # at or above: the smaller model
}
SHORT_REQUEST_CHARS = 400
SHORT_ANSWER_TOKENS = 32
LONG_REQUEST_CHARS = 32_000
_REASONING = re.compile(r"\b(why|explain|analy[sz]e|prove|derive|compare|evaluate|design|plan|reason|"
                        r"step by step|trade-?offs?|justify|critique)\b", re.I)
# [ \t]* rather than \s*: a run of blank lines must not make every line start a rescan.
_CONSTRAINT = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d{1,3}[.)])[ \t]+|\b(must|should|exactly|at least|no more than)\b", re.I)
_MATHS = re.compile(r"[=+*/^%]|\d")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def features(body: Dict[str, Any]) -> Dict[str, float]:
    """Numbers describing the request (never its text)."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    last = _text_of(users[-1].get("content")) if users else ""
    everything = _text_of(body.get("system")) + "".join(
        _text_of(m.get("content")) for m in messages if isinstance(m, dict))
    asked = body.get("max_completion_tokens") or body.get("max_tokens")
    asked = asked if isinstance(asked, int) and not isinstance(asked, bool) and asked > 0 else 1000
    sample = last[:20_000]
    maths = len(_MATHS.findall(sample)) / max(1, len(sample))
    fmt = body.get("response_format")
    return {
        "user_chars": math.log1p(len(last)),
        "prompt_chars": math.log1p(len(everything)),
        "questions": float(min(5, sample.count("?"))),
        "code": 1.0 if "```" in sample or re.search(r"\b(def|class|function|SELECT|import)\b", sample) else 0.0,
        "maths": min(1.0, maths * 5),
        "reasoning": float(min(5, len(_REASONING.findall(sample)))),
        "constraints": float(min(10, len(_CONSTRAINT.findall(sample)))),
        "turns": float(min(20, len(users))),
        "answer_tokens": math.log1p(asked),
        "structured": 1.0 if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema") else 0.0,
    }


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _usable(model: Any) -> bool:
    """A tuned model with a finite bias, a weight for every feature and a threshold from 0.5
    to 1. Bundles are cleaned when they load; anything else here is not used."""
    if not isinstance(model, dict) or not isinstance(model.get("weights"), dict):
        return False
    easy_at = _number(model.get("easy_at"))
    return (_number(model.get("bias")) is not None and easy_at is not None and 0.5 <= easy_at < 1.0
            and all(_number(model["weights"].get(name)) is not None for name in FEATURES))


def p_easy(values: Dict[str, float], model: Optional[Dict[str, Any]] = None) -> float:
    model = model if _usable(model) else DEFAULT_MODEL
    z = float(model["bias"])
    for name in FEATURES:
        z += float(model["weights"].get(name, 0.0)) * values.get(name, 0.0)
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def decide(body: Dict[str, Any], tuned: Optional[Dict[str, Any]] = None) -> Tuple[str, str, float, Dict[str, float]]:
    """("small" | "stay", why, the easy probability, the features)."""
    values = features(body)
    model = tuned.get("right_size") if isinstance(tuned, dict) else None
    model = model if _usable(model) else DEFAULT_MODEL
    probability = p_easy(values, model)
    user_chars = math.expm1(values["user_chars"])
    prompt_chars = math.expm1(values["prompt_chars"])
    answer_tokens = math.expm1(values["answer_tokens"])
    if prompt_chars > LONG_REQUEST_CHARS:
        return "stay", "long request", probability, values
    if user_chars <= SHORT_REQUEST_CHARS and answer_tokens <= SHORT_ANSWER_TOKENS:
        return "small", "short request, short answer", probability, values
    if probability >= model["easy_at"]:
        return "small", "easy by score", probability, values
    return "stay", "hard or uncertain by score", probability, values


def is_agent_step(body: Dict[str, Any]) -> bool:
    """A call inside an agent loop: tools offered, or tool results in the conversation."""
    if body.get("tools") or body.get("functions"):
        return True
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") in ("tool", "function") or m.get("tool_calls"):
            return True
        content = m.get("content")
        if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
                                             for b in content):
            return True
    return False


def vector(values: Dict[str, float]) -> List[float]:
    return [round(values.get(name, 0.0), 4) for name in FEATURES]
