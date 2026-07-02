"""OpenAI Moderation remote_api adapter (plan P4).

Maps the OpenAI Moderation API response — a stable, public, industry-standard
contract — onto the F1 DetectorResult. The response shape is:

    {"id": "...", "model": "omni-moderation-latest",
     "results": [{"flagged": bool,
                  "categories": {"hate": false, "violence": true, ...},
                  "category_scores": {"hate": 0.01, "violence": 0.93, ...}}]}

The egress gate + audit are applied by the escalation path BEFORE this runs, so
the adapter only builds the request, posts, and maps the reply. The HTTP poster
is injectable so the mapper can be contract-tested against recorded JSON with no
network. A flagged result returns BLOCK (model-backed → the scorecard gate keeps
it advisory/WARN until a passing scorecard, per P2)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.net_guard import assert_safe_egress_url

# (url, json_payload, headers, timeout_seconds) -> parsed JSON dict
PostFn = Callable[[str, Dict[str, Any], Dict[str, str], float], Dict[str, Any]]

DEFAULT_URL = "https://api.openai.com/v1/moderations"
DEFAULT_MODEL = "omni-moderation-latest"


def _default_post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    import httpx
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


class OpenAIModerationAdapter:
    """Vendor adapter: provider key ``openai_moderation`` (and OpenAI-compatible
    moderation endpoints via a custom ``endpoint_url``)."""

    name = "openai_moderation"

    def build_request(self, text: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        url = config.get("endpoint_url") or DEFAULT_URL
        key = config.get("auth_value") or config.get("api_key") or ""
        model = config.get("model_id") or DEFAULT_MODEL
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return url, {"model": model, "input": text}, headers

    def map_response(self, data: Dict[str, Any], config: Dict[str, Any]) -> DetectorResult:
        results = data.get("results") or []
        r = results[0] if isinstance(results, list) and results else {}
        flagged = bool(r.get("flagged"))
        scores = r.get("category_scores") or {}
        cats = r.get("categories") or {}
        try:
            max_score = max((float(v) for v in scores.values() if v is not None), default=0.0)
        except (TypeError, ValueError):
            max_score = 0.0
        max_score = min(max(max_score, 0.0), 1.0)

        action = str(config.get("action") or "BLOCK").upper()
        try:
            flagged_decision = Decision(action)
        except ValueError:
            flagged_decision = Decision.BLOCK
        decision = flagged_decision if flagged else Decision.ALLOW

        hits = [
            RuleHit(rule_id=f"openai_moderation.{cat}", message=f"flagged category: {cat}", severity=Severity.HIGH)
            for cat, on in cats.items() if on
        ]
        return DetectorResult(
            decision=decision,
            risk_score=int(round(max_score * 100)) if flagged else 0,
            confidence=max_score,
            model_version=data.get("model"),
            rule_hits=hits,
            external_egress=True,
            execution_mode="remote_api",
            developer_message=f"openai_moderation: {'flagged' if flagged else 'clean'} (max category score {max_score:.3f})",
        )

    def evaluate(self, text: str, config: Dict[str, Any], *,
                 post: Optional[PostFn] = None, timeout: float = 8.0) -> DetectorResult:
        url, payload, headers = self.build_request(text, config)
        poster = post or _default_post
        # SSRF guard (defense-in-depth; the escalation egress gate also runs upstream).
        # Skipped when a poster is injected — the injected transport owns the I/O, and the
        # guard's DNS resolution would otherwise require network access in tests.
        if post is None:
            assert_safe_egress_url(url, allow_private=False)
        data = poster(url, payload, headers, timeout)
        return self.map_response(data, config)
