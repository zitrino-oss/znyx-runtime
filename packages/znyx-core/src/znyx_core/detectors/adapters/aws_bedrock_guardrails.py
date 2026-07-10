"""AWS Bedrock Guardrails remote_api adapter.

Maps the Bedrock Runtime ``ApplyGuardrail`` response onto the DetectorResult.
Unlike OpenAI/Azure (header-key auth) this API requires AWS SigV4 request signing,
implemented here with the stdlib (hashlib/hmac) so the runtime gains no new
dependency.

Contract (Bedrock Runtime ``ApplyGuardrail``):

    POST https://bedrock-runtime.{region}.amazonaws.com
         /guardrail/{guardrailId}/version/{guardrailVersion}/apply
    Body: {"source": "INPUT"|"OUTPUT", "content": [{"text": {"text": "..."}}]}

    Response: {"action": "GUARDRAIL_INTERVENED"|"NONE",
               "actionReason": "...",
               "assessments": [{"topicPolicy": {...}, "contentPolicy": {...},
                                "wordPolicy": {...}, "sensitiveInformationPolicy": {...}}],
               "outputs": [{"text": "..."}], "usage": {...}}

Credentials/config (carried via the backend's auth_value + provider_config):
    access_key_id     ← config["auth_value"] (or provider_config["access_key_id"])
    secret_access_key ← provider_config["secret_access_key"]
    session_token     ← provider_config["session_token"]  (optional, for STS creds)
    region            ← backend.region / provider_config["region"]
    guardrail_id, guardrail_version (default "DRAFT"), source (default "INPUT")

The egress gate + audit run upstream in the escalation path; this only signs,
builds the request, posts, and maps. The poster is injectable for contract tests."""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.net_guard import assert_safe_egress_url

PostFn = Callable[[str, Dict[str, Any], Dict[str, str], float], Dict[str, Any]]

_SERVICE = "bedrock"           # SigV4 signing name for the bedrock-runtime endpoint
_ALGORITHM = "AWS4-HMAC-SHA256"
DEFAULT_GUARDRAIL_VERSION = "DRAFT"
DEFAULT_FLAGGED_RISK = 90      # Bedrock returns no numeric score; intervened = high risk


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = _hmac(("AWS4" + secret).encode("utf-8"), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    return _hmac(k_service, "aws4_request")


def _body_bytes(payload: Dict[str, Any]) -> bytes:
    # Deterministic, compact JSON — must match exactly what the poster sends.
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def sigv4_headers(method: str, url: str, body: bytes, *, access_key: str, secret_key: str,
                  region: str, service: str = _SERVICE, session_token: Optional[str] = None,
                  now: Optional[datetime] = None) -> Dict[str, str]:
    """Build the SigV4 Authorization + x-amz-* headers for a request. Pure/testable."""
    now = now or datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    parts = urlsplit(url)
    host = parts.netloc
    canonical_uri = quote(parts.path or "/", safe="/-_.~")
    canonical_qs = ""  # ApplyGuardrail has no query string
    payload_hash = _sha256_hex(body)

    signed = {
        "content-type": "application/json",
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if session_token:
        signed["x-amz-security-token"] = session_token
    signed_header_names = ";".join(sorted(signed))
    canonical_headers = "".join(f"{k}:{signed[k]}\n" for k in sorted(signed))

    canonical_request = "\n".join([
        method, canonical_uri, canonical_qs, canonical_headers, signed_header_names, payload_hash,
    ])
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        _ALGORITHM, amz_date, credential_scope, _sha256_hex(canonical_request.encode("utf-8")),
    ])
    signature = hmac.new(
        _signing_key(secret_key, date_stamp, region, service),
        string_to_sign.encode("utf-8"), hashlib.sha256,
    ).hexdigest()
    authorization = (
        f"{_ALGORITHM} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_header_names}, Signature={signature}"
    )
    out = {
        "Authorization": authorization,
        "X-Amz-Date": amz_date,
        "X-Amz-Content-Sha256": payload_hash,
        "Content-Type": "application/json",
    }
    if session_token:
        out["X-Amz-Security-Token"] = session_token
    return out


def _default_post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    import httpx
    body = _body_bytes(payload)  # MUST match the bytes signed in build_request
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, content=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


class AwsBedrockGuardrailsAdapter:
    """Vendor adapter: provider key ``aws_bedrock_guardrails`` (SigV4-signed)."""

    name = "aws_bedrock_guardrails"

    def _endpoint(self, config: Dict[str, Any], region: str) -> str:
        if config.get("endpoint_url"):
            return str(config["endpoint_url"])
        gid = config.get("guardrail_id") or ""
        ver = config.get("guardrail_version") or DEFAULT_GUARDRAIL_VERSION
        host = f"https://bedrock-runtime.{region}.amazonaws.com"
        return f"{host}/guardrail/{quote(str(gid), safe='')}/version/{quote(str(ver), safe='')}/apply"

    def build_request(self, text: str, config: Dict[str, Any], *,
                      now: Optional[datetime] = None) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        region = config.get("region") or "us-east-1"
        url = self._endpoint(config, region)
        source = str(config.get("source") or "INPUT").upper()
        payload = {"source": source, "content": [{"text": {"text": text}}]}

        access_key = config.get("access_key_id") or config.get("auth_value") or ""
        secret_key = config.get("secret_access_key") or ""
        session_token = config.get("session_token")
        headers = sigv4_headers(
            "POST", url, _body_bytes(payload),
            access_key=access_key, secret_key=secret_key, region=region,
            session_token=session_token, now=now,
        )
        return url, payload, headers

    def map_response(self, data: Dict[str, Any], config: Dict[str, Any]) -> DetectorResult:
        flagged = str(data.get("action") or "").upper() == "GUARDRAIL_INTERVENED"

        # Collect the policy items that fired across all assessments (defensive about shape).
        hits: List[RuleHit] = []
        for assessment in (data.get("assessments") or []):
            if not isinstance(assessment, dict):
                continue
            for topic in (assessment.get("topicPolicy", {}) or {}).get("topics", []) or []:
                if isinstance(topic, dict) and str(topic.get("action", "")).upper() == "BLOCKED":
                    hits.append(RuleHit(rule_id=f"aws_bedrock.topic.{topic.get('name', 'topic')}",
                                        message=f"blocked topic: {topic.get('name')}", severity=Severity.HIGH))
            for f in (assessment.get("contentPolicy", {}) or {}).get("filters", []) or []:
                if isinstance(f, dict) and str(f.get("action", "")).upper() == "BLOCKED":
                    hits.append(RuleHit(rule_id=f"aws_bedrock.content.{str(f.get('type', 'filter')).lower()}",
                                        message=f"content filter: {f.get('type')} ({f.get('confidence')})",
                                        severity=Severity.HIGH))
            for w in (assessment.get("wordPolicy", {}) or {}).get("customWords", []) or []:
                if isinstance(w, dict) and str(w.get("action", "")).upper() == "BLOCKED":
                    hits.append(RuleHit(rule_id="aws_bedrock.word.custom",
                                        message=f"blocked word: {w.get('match')}", severity=Severity.MEDIUM))
            sip = assessment.get("sensitiveInformationPolicy", {}) or {}
            for pii in (sip.get("piiEntities", []) or []):
                if isinstance(pii, dict) and str(pii.get("action", "")).upper() in ("BLOCKED", "ANONYMIZED"):
                    hits.append(RuleHit(rule_id=f"aws_bedrock.pii.{str(pii.get('type', 'pii')).lower()}",
                                        message=f"sensitive info: {pii.get('type')}", severity=Severity.HIGH))

        action = str(config.get("action") or "BLOCK").upper()
        try:
            flagged_decision = Decision(action)
        except ValueError:
            flagged_decision = Decision.BLOCK
        decision = flagged_decision if flagged else Decision.ALLOW
        risk = int(config.get("flagged_risk") or DEFAULT_FLAGGED_RISK) if flagged else 0

        return DetectorResult(
            decision=decision,
            risk_score=risk,
            confidence=(risk / 100.0) if flagged else 0.0,
            rule_hits=hits if flagged else [],
            external_egress=True,
            execution_mode="remote_api",
            developer_message=(
                f"aws_bedrock_guardrails: {'intervened' if flagged else 'none'}"
                + (f" — {data.get('actionReason')}" if data.get("actionReason") else "")
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
