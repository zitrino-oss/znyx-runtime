"""BYOK LLM provider abstraction for the Playground round-trip runner.

Design
======

A thin, protocol-driven adapter layer over the four LLM families we need
to support for customer-facing "test with real LLM" flows:

- **OpenAI** (and OpenAI-compatible endpoints — Groq, Together, Fireworks,
  vLLM / Ollama behind a reverse proxy…)
- **Anthropic** (Claude Sonnet / Opus / Haiku)
- **Google** (Gemini 2.x via the Generative Language API)
- **custom** — any ``POST /completions``-style HTTPS endpoint that accepts
  an OpenAI-shaped payload; used for self-hosted models and internal
  gateways.

Everything is async (``httpx.AsyncClient``) because the round-trip runner
calls this from inside the FastAPI request path. Every adapter raises a
shared :class:`LLMCallError` on any network or decode failure so the
runner can wrap the failure into a trace stage rather than bubbling a
stack trace to the client.

Security posture
----------------

- API keys never hit logs. Adapters accept them as arguments, not via
  ``os.getenv``, and the orchestrator above this layer is responsible for
  **never writing them to DB unless the session-save opt-in is explicit**.
- HTTPS-only by default — attempts to hit http:// endpoints raise unless
  ``allow_http=True`` is explicitly set (used by localhost dev targets).
- Response bodies are size-capped so a malicious / runaway model can't
  DOS the runner.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# Max bytes we accept from a provider. Real model responses are well under
# this; the cap exists so a pathological or hostile endpoint can't force the
# runner to allocate arbitrary amounts of memory.
_MAX_RESPONSE_BYTES = 512 * 1024  # 512 KiB

# Default per-request timeout. The round-trip runner owns the total budget
# and may override on a per-call basis.
_DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMCallError(Exception):
    """Normalised error from any provider adapter.

    Carries a machine-readable ``kind`` so the trace UI can render distinct
    affordances (retry for rate limit, re-key for auth error, etc.).
    """

    KIND_AUTH = "auth"              # 401 / invalid key
    KIND_RATE_LIMIT = "rate_limit"  # 429
    KIND_TIMEOUT = "timeout"        # network timeout
    KIND_NETWORK = "network"        # dns / connection reset
    KIND_BAD_REQUEST = "bad_request"  # 4xx other than 401/429
    KIND_SERVER = "server"          # 5xx
    KIND_DECODE = "decode"          # unexpected response shape
    KIND_OTHER = "other"

    def __init__(self, kind: str, message: str, *, status_code: Optional[int] = None):
        self.kind = kind
        self.status_code = status_code
        super().__init__(message)


@dataclass
class CompletionResult:
    """Normalised provider output.

    Every adapter converts its native response into this so the runner
    never has to branch on provider-specific shapes.
    """

    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0


@dataclass
class CompletionRequest:
    input_text: str
    system_prompt: Optional[str] = None
    model: str = ""
    max_tokens: int = 1024
    temperature: float = 0.2
    # Used for OpenAI-compatible + custom providers.
    endpoint_url: Optional[str] = None


class LLMProvider(Protocol):
    """Adapter protocol every concrete provider implements."""

    name: str

    async def complete(
        self, api_key: str, request: CompletionRequest, *, timeout: float = _DEFAULT_TIMEOUT_SECONDS
    ) -> CompletionResult: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


def _validate_endpoint(url: str, *, allow_http: bool = False) -> None:
    """Reject mixed-scheme, non-absolute, or private-IP endpoints.

    Customer-supplied endpoints get here for the ``custom`` + ``openai_compat``
    providers; preventing ``http://`` in prod is a must so we don't leak API
    keys in cleartext.  The IP check blocks SSRF against cloud metadata
    services and internal networks.

    Known local/in-boundary hostnames (localhost, host.docker.internal, etc.)
    are exempt from both checks — they are explicitly co-located endpoints and
    by definition cannot be used to reach external networks or cloud metadata.
    """
    if not url:
        raise LLMCallError(LLMCallError.KIND_BAD_REQUEST, "endpoint_url is required")

    # --- In-boundary exemption ---
    hostname = (urlparse(url).hostname or "").lower()
    if hostname in _LOCAL_HOSTS:
        return  # local/in-boundary sidecar — skip HTTPS and SSRF checks

    if not (url.startswith("https://") or (allow_http and url.startswith("http://"))):
        raise LLMCallError(
            LLMCallError.KIND_BAD_REQUEST,
            "LLM endpoint must use HTTPS (set allow_http=True for localhost dev only)",
        )

    # --- SSRF protection (delegated to the shared net_guard, single source of
    # truth). allow_private=False preserves the prior posture (block
    # private/loopback/reserved) and additionally blocks the cloud-metadata
    # hostnames, multicast, and unspecified addresses the hand-rolled check missed.
    from znyx_core.net_guard import assert_safe_egress_url, UnsafeEgressURL
    try:
        assert_safe_egress_url(url, allow_private=False)
    except UnsafeEgressURL as exc:
        raise LLMCallError(LLMCallError.KIND_BAD_REQUEST, str(exc)) from exc


def _sanitize_url(url: str) -> str:
    """Strip API key query params from URLs before including in error messages."""
    return re.sub(r'([?&])key=[^&]+', r'\1key=***', url)


async def _post_json(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    *,
    timeout: float,
) -> Dict[str, Any]:
    """Shared HTTP POST with consistent error mapping."""
    try:
        # follow_redirects=False: a redirect target bypasses the egress guard.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise LLMCallError(LLMCallError.KIND_TIMEOUT, f"LLM request timed out after {timeout:.1f}s") from exc
    except httpx.ConnectError as exc:
        raise LLMCallError(LLMCallError.KIND_NETWORK, f"Could not connect to {_sanitize_url(url)}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise LLMCallError(LLMCallError.KIND_NETWORK, f"HTTP error: {exc}") from exc

    if resp.status_code == 401 or resp.status_code == 403:
        raise LLMCallError(
            LLMCallError.KIND_AUTH, "Invalid API key or unauthorized", status_code=resp.status_code
        )
    if resp.status_code == 429:
        raise LLMCallError(
            LLMCallError.KIND_RATE_LIMIT, "Rate limit hit", status_code=429
        )
    if 400 <= resp.status_code < 500:
        # Log the full upstream error for debugging; return only a safe
        # summary to the client so provider internals aren't leaked.
        snippet = resp.text[:400] if resp.text else resp.reason_phrase
        logger.warning(
            "LLM provider %s error: %s — %s", resp.status_code, _sanitize_url(url), snippet,
        )
        raise LLMCallError(
            LLMCallError.KIND_BAD_REQUEST,
            f"LLM provider returned {resp.status_code}. Check your model name, API key, and request parameters.",
            status_code=resp.status_code,
        )
    if resp.status_code >= 500:
        raise LLMCallError(
            LLMCallError.KIND_SERVER, f"Upstream {resp.status_code}", status_code=resp.status_code
        )

    # Body too large → give up rather than buffer indefinitely.
    # Check both the declared Content-Length (if present) and the actual
    # body size, since a malicious upstream can omit or lie about the header.
    cl = resp.headers.get("content-length")
    if cl and int(cl) > _MAX_RESPONSE_BYTES:
        raise LLMCallError(
            LLMCallError.KIND_DECODE,
            f"Response exceeds {_MAX_RESPONSE_BYTES} byte cap",
        )
    if len(resp.content) > _MAX_RESPONSE_BYTES:
        raise LLMCallError(
            LLMCallError.KIND_DECODE,
            f"Response body exceeds {_MAX_RESPONSE_BYTES} byte cap",
        )

    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise LLMCallError(LLMCallError.KIND_DECODE, f"Invalid JSON from provider: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI (and OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """OpenAI Chat Completions API adapter.

    Also works with any OpenAI-compatible endpoint (Groq, Together,
    Fireworks, vLLM behind a proxy) — pass ``request.endpoint_url`` to
    override the default OpenAI host.
    """

    name = "openai"
    DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    async def complete(
        self,
        api_key: str,
        request: CompletionRequest,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> CompletionResult:
        url = request.endpoint_url or self.DEFAULT_ENDPOINT
        _validate_endpoint(url)

        messages: List[Dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.input_text})

        body = {
            "model": request.model or "gpt-4o-mini",
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        t0 = time.perf_counter()
        data = await _post_json(url, headers, body, timeout=timeout)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage = data.get("usage", {}) or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMCallError(
                LLMCallError.KIND_DECODE, f"OpenAI response missing expected fields: {exc}"
            ) from exc

        return CompletionResult(
            text=text,
            model=data.get("model", request.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw=data,
            latency_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicProvider:
    name = "anthropic"
    DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    async def complete(
        self,
        api_key: str,
        request: CompletionRequest,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> CompletionResult:
        url = request.endpoint_url or self.DEFAULT_ENDPOINT
        _validate_endpoint(url)

        body: Dict[str, Any] = {
            "model": request.model or "claude-3-5-haiku-20241022",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.input_text}],
        }
        if request.system_prompt:
            body["system"] = request.system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        data = await _post_json(url, headers, body, timeout=timeout)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        try:
            # Anthropic returns content as an array of typed blocks.
            blocks = data.get("content") or []
            text = "".join(
                block.get("text", "") for block in blocks if block.get("type") == "text"
            )
            usage = data.get("usage", {}) or {}
        except (KeyError, TypeError) as exc:
            raise LLMCallError(
                LLMCallError.KIND_DECODE, f"Anthropic response malformed: {exc}"
            ) from exc

        return CompletionResult(
            text=text,
            model=data.get("model", request.model),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            total_tokens=(
                (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                if usage.get("input_tokens") is not None
                else None
            ),
            raw=data,
            latency_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GoogleProvider:
    name = "google"

    async def complete(
        self,
        api_key: str,
        request: CompletionRequest,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> CompletionResult:
        # Gemini uses a model-in-path URL + API key as query param.
        model = request.model or "gemini-1.5-flash"
        default_endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        url = request.endpoint_url or default_endpoint
        _validate_endpoint(url)

        parts: List[Dict[str, Any]] = [{"text": request.input_text}]
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if request.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}

        # Use header-based auth to avoid leaking the API key in URL query
        # params (which appear in proxy logs, CDN logs, and access logs).
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        t0 = time.perf_counter()
        data = await _post_json(url, headers, body, timeout=timeout)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        try:
            candidate = (data.get("candidates") or [{}])[0]
            content_parts = candidate.get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in content_parts)
            usage = data.get("usageMetadata", {}) or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMCallError(
                LLMCallError.KIND_DECODE, f"Gemini response malformed: {exc}"
            ) from exc

        return CompletionResult(
            text=text,
            model=model,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            total_tokens=usage.get("totalTokenCount"),
            raw=data,
            latency_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Custom (raw OpenAI-shaped POST)
# ---------------------------------------------------------------------------


class CustomProvider:
    """Catch-all for self-hosted + internal gateway endpoints.

    Expects the endpoint to accept an OpenAI chat-completions-shaped body
    and return an OpenAI-shaped response. 95% of "internal LLM proxy"
    installations already do this because the OpenAI shape is the de-facto
    standard.
    """

    name = "custom"

    async def complete(
        self,
        api_key: str,
        request: CompletionRequest,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> CompletionResult:
        if not request.endpoint_url:
            raise LLMCallError(
                LLMCallError.KIND_BAD_REQUEST,
                "endpoint_url is required for the 'custom' provider",
            )
        # Delegate to OpenAI adapter — same wire shape.
        return await OpenAIProvider().complete(api_key, request, timeout=timeout)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_PROVIDERS: Dict[str, LLMProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "google": GoogleProvider(),
    "custom": CustomProvider(),
}


def get_provider(name: str) -> LLMProvider:
    """Resolve a provider by name. Raises ``LLMCallError`` on unknown."""
    provider = _PROVIDERS.get(name.lower())
    if provider is None:
        raise LLMCallError(
            LLMCallError.KIND_BAD_REQUEST, f"Unknown LLM provider: {name!r}"
        )
    return provider


def supported_providers() -> List[str]:
    return list(_PROVIDERS.keys())


# Default destination per provider/adapter — the host the call ACTUALLY hits when no
# explicit endpoint_url is given (the provider would otherwise default it internally). The
# F4 egress gate must see this so it audits + allowlists + residency-checks the real
# destination instead of recording "(unknown)". 'custom' has no default (it requires an
# endpoint_url), so it resolves to None and the gate correctly sees no host.
_DEFAULT_ENDPOINTS: Dict[str, str] = {
    "openai": OpenAIProvider.DEFAULT_ENDPOINT,
    "anthropic": AnthropicProvider.DEFAULT_ENDPOINT,
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
    # vendor moderation adapter (znyx_core.detectors.adapters.openai_moderation)
    "openai_moderation": "https://api.openai.com/v1/moderations",
}


def effective_endpoint(provider: Optional[str], endpoint_url: Optional[str]) -> Optional[str]:
    """The URL a call will actually reach: an explicit ``endpoint_url`` always wins;
    otherwise the provider's default host. Used by the egress gate so auditing/allowlisting
    reflect the true destination rather than ``None`` (which the provider would silently
    default to OpenAI/etc. AFTER the gate already passed)."""
    if endpoint_url:
        return endpoint_url
    if not provider:
        return None
    return _DEFAULT_ENDPOINTS.get(provider.lower())
