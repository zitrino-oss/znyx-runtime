"""Azure AI Content Safety remote_api adapter.

Maps the Azure AI Content Safety "Analyze Text" response — a stable, GA, public
contract — onto the DetectorResult. The contract (api-version 2024-09-01):

    POST {endpoint}/contentsafety/text:analyze?api-version=2024-09-01
    Headers: Ocp-Apim-Subscription-Key: <key>, Content-Type: application/json
    Body:    {"text": "...", "categories": ["Hate","SelfHarm","Sexual","Violence"],
              "outputType": "FourSeverityLevels"}

    Response: {"blocklistsMatch": [...],
               "categoriesAnalysis": [{"category": "Hate", "severity": 0},
                                      {"category": "Violence", "severity": 4}, ...]}

Severity is 0/2/4/6 in FourSeverityLevels (0..7 in EightSeverityLevels). Azure
returns severities, not a flagged boolean — the caller decides the cut-off, so a
``block_severity`` threshold (default 4 = "medium") gates BLOCK vs ALLOW.

The egress gate + audit run upstream in the escalation path; this only builds the
request, posts, and maps. The poster is injectable for contract tests (no network)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.net_guard import assert_safe_egress_url

PostFn = Callable[[str, Dict[str, Any], Dict[str, str], float], Dict[str, Any]]

DEFAULT_API_VERSION = "2024-09-01"
DEFAULT_BLOCK_SEVERITY = 4           # 0 safe · 2 low · 4 medium · 6 high (FourSeverityLevels)
_ANALYZE_PATH = "/contentsafety/text:analyze"


def _default_post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    import httpx
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _severity_label(sev: int, block_severity: int) -> Severity:
    if sev >= block_severity:
        return Severity.HIGH
    if sev >= max(2, block_severity - 2):
        return Severity.MEDIUM
    return Severity.LOW


class AzureContentSafetyAdapter:
    """Vendor adapter: provider key ``azure_content_safety``."""

    name = "azure_content_safety"

    def build_request(self, text: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        base = (config.get("endpoint_url") or "").rstrip("/")
        api_version = config.get("api_version") or DEFAULT_API_VERSION
        # Accept either a full analyze URL or a bare resource endpoint.
        if _ANALYZE_PATH in base:
            url = base
        else:
            url = f"{base}{_ANALYZE_PATH}?api-version={api_version}"
        key = config.get("auth_value") or config.get("api_key") or ""
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Ocp-Apim-Subscription-Key"] = key
        payload: Dict[str, Any] = {"text": text}
        categories = config.get("categories")
        if isinstance(categories, list) and categories:
            payload["categories"] = categories
        payload["outputType"] = config.get("output_type") or "FourSeverityLevels"
        return url, payload, headers

    def map_response(self, data: Dict[str, Any], config: Dict[str, Any]) -> DetectorResult:
        analysis = data.get("categoriesAnalysis") or []
        block_severity = int(config.get("block_severity") or DEFAULT_BLOCK_SEVERITY)
        # Eight-level severities top out at 7; four-level at 6. Normalise risk by the scale.
        scale = 7 if str(config.get("output_type") or "").lower().startswith("eight") else 6

        max_sev = 0
        hits: List[RuleHit] = []
        for item in analysis:
            if not isinstance(item, dict):
                continue
            cat = item.get("category") or "Unknown"
            try:
                sev = int(item.get("severity") or 0)
            except (TypeError, ValueError):
                sev = 0
            if sev <= 0:
                continue
            max_sev = max(max_sev, sev)
            hits.append(RuleHit(
                rule_id=f"azure_content_safety.{str(cat).lower()}",
                message=f"{cat} severity {sev}",
                severity=_severity_label(sev, block_severity),
            ))

        flagged = max_sev >= block_severity
        risk = min(100, max(0, round(max_sev / scale * 100))) if max_sev else 0

        action = str(config.get("action") or "BLOCK").upper()
        try:
            flagged_decision = Decision(action)
        except ValueError:
            flagged_decision = Decision.BLOCK
        decision = flagged_decision if flagged else Decision.ALLOW

        # Only surface the threshold-crossing hits when flagged; below-threshold
        # categories stay informational (no decision impact) but are noted.
        return DetectorResult(
            decision=decision,
            risk_score=risk if flagged else 0,
            confidence=(risk / 100.0) if max_sev else 0.0,
            rule_hits=hits if flagged else [],
            external_egress=True,
            execution_mode="remote_api",
            developer_message=(
                f"azure_content_safety: {'flagged' if flagged else 'clean'} "
                f"(max severity {max_sev}, block_severity {block_severity})"
            ),
        )

    def evaluate(self, text: str, config: Dict[str, Any], *,
                 post: Optional[PostFn] = None, timeout: float = 8.0) -> DetectorResult:
        url, payload, headers = self.build_request(text, config)
        poster = post or _default_post
        # SSRF guard only on the live transport (injected posters own their I/O; the
        # escalation egress gate runs upstream regardless).
        if post is None:
            assert_safe_egress_url(url, allow_private=False)
        data = poster(url, payload, headers, timeout)
        return self.map_response(data, config)
