"""Calls to the provider: pooled connections, URLs, credentials and headers.

One keep-alive connection pool per provider (technique 38), so a call does not pay
for a new TLS handshake. The client's own credential for the gateway is never
forwarded; the provider credential comes from the provider's reference (or, for a
provider configured with ``api_key: client``, from the caller). Only the provider
headers a client legitimately sets (API version, beta flags) are passed on. The
organisation and project headers choose whose account is billed, so they pass only
when the caller sends its own provider key; Anthropic betas that let the provider
reach MCP servers or stored files (outside the gateway's tool policy and egress
gate) are dropped. Only the provider's informative response headers (rate limits,
request ids) are passed back, never the account's organisation or project.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
from typing import Any, Dict, Optional, Tuple

import httpx

from znyx_runtime.gateway.config import Provider

_FORWARD_REQUEST_HEADERS = {
    "openai": ("openai-beta",),
    "azure_openai": ("openai-beta",),
    "anthropic": ("anthropic-version", "anthropic-beta"),
}
# Only with the caller's own provider key: they pick the account that pays.
_CLIENT_KEY_HEADERS = {"openai": ("openai-organization", "openai-project")}
# Betas that make the provider itself call out (MCP servers) or read stored files.
_DROPPED_BETAS = ("mcp-client-*", "files-api-*")
_RETURN_RESPONSE_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-", "openai-")
_RETURN_RESPONSE_HEADERS = ("retry-after", "x-request-id", "request-id", "openai-processing-ms",
                            "azureml-model-session", "apim-request-id")
_HIDDEN_RESPONSE_HEADERS = ("openai-organization", "openai-project")
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class UpstreamError(Exception):
    """The provider could not be reached (or timed out); ``status`` is what the
    gateway answers with (502 or 504)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Upstream:
    def __init__(self, max_connections: int = 256):
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()
        self._max_connections = max_connections

    async def client(self, provider: Provider) -> httpx.AsyncClient:
        client = self._clients.get(provider.name)
        if client is not None:
            return client
        async with self._lock:
            client = self._clients.get(provider.name)
            if client is None:
                client = httpx.AsyncClient(
                    timeout=httpx.Timeout(provider.timeout_seconds, connect=10.0, pool=10.0),
                    limits=httpx.Limits(max_connections=self._max_connections,
                                        max_keepalive_connections=min(64, self._max_connections)),
                    follow_redirects=False,
                )
                self._clients[provider.name] = client
            return client

    async def close(self) -> None:
        for client in list(self._clients.values()):
            await client.aclose()
        self._clients.clear()

    # -- request construction ---------------------------------------------------

    @staticmethod
    def url(provider: Provider, *, deployment: Optional[str] = None,
            api_version: Optional[str] = None) -> str:
        if provider.format == "anthropic":
            return f"{provider.base_url}/v1/messages"
        if provider.format == "azure_openai":
            version = api_version or provider.api_version
            return (f"{provider.base_url}/openai/deployments/{deployment}/chat/completions"
                    f"?api-version={version}")
        return f"{provider.base_url}/chat/completions"

    @staticmethod
    def headers(provider: Provider, client_headers: Any, credential: Optional[str]) -> Dict[str, str]:
        out = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
        names = _FORWARD_REQUEST_HEADERS.get(provider.format, ())
        if provider.uses_client_key:
            names = names + _CLIENT_KEY_HEADERS.get(provider.format, ())
        for name in names:
            value = client_headers.get(name)
            if value:
                out[name] = value
        if "anthropic-beta" in out and not provider.uses_client_key:
            kept = [b.strip() for b in out["anthropic-beta"].split(",")
                    if b.strip() and not any(fnmatch.fnmatchcase(b.strip().lower(), d) for d in _DROPPED_BETAS)]
            if kept:
                out["anthropic-beta"] = ",".join(kept)
            else:
                out.pop("anthropic-beta")
        if provider.format == "anthropic":
            out.setdefault("anthropic-version", DEFAULT_ANTHROPIC_VERSION)
            if credential:
                out["x-api-key"] = credential
        elif provider.format == "azure_openai":
            if credential:
                out["api-key"] = credential
        elif credential:
            out["authorization"] = f"Bearer {credential}"
        for name, value in provider.headers:
            out[name.lower()] = value
        return out

    @staticmethod
    def returned_headers(response: httpx.Response) -> Dict[str, str]:
        out = {}
        for name, value in response.headers.items():
            lower = name.lower()
            if lower in _HIDDEN_RESPONSE_HEADERS:
                continue
            if lower in _RETURN_RESPONSE_HEADERS or lower.startswith(_RETURN_RESPONSE_PREFIXES):
                out[lower] = value
        return out

    # -- calls ----------------------------------------------------------------------

    async def send(self, provider: Provider, url: str, headers: Dict[str, str],
                   body: Dict[str, Any]) -> httpx.Response:
        client = await self.client(provider)
        try:
            return await client.post(url, headers=headers,
                                     content=json.dumps(body, separators=(",", ":")).encode())
        except httpx.TimeoutException as exc:
            raise UpstreamError(504, f"provider {provider.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(502, f"provider {provider.name} unreachable ({type(exc).__name__})") from exc

    async def open_stream(self, provider: Provider, url: str, headers: Dict[str, str],
                          body: Dict[str, Any]) -> Tuple[httpx.AsyncClient, httpx.Response]:
        client = await self.client(provider)
        request = client.build_request("POST", url, headers=headers,
                                       content=json.dumps(body, separators=(",", ":")).encode())
        try:
            response = await client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamError(504, f"provider {provider.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(502, f"provider {provider.name} unreachable ({type(exc).__name__})") from exc
        return client, response
