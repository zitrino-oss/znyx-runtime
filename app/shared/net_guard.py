"""SSRF egress guard for outbound URLs (webhooks, remote detectors, LLM gateways).

Framework-agnostic (raises ``UnsafeEgressURL``, not an HTTP error) so it can be
used from both the control plane and the runtime/shared layers.

Two postures:
  - ``allow_private=False`` (default) — for **customer-supplied** URLs (e.g.
    webhooks): block private/loopback/link-local/metadata. Strictest.
  - ``allow_private=True`` — for **operator-configured infrastructure** (e.g.
    self-hosted remote detectors or a private LLM gateway) that may legitimately
    live on an internal network. We still ALWAYS block the cloud-metadata
    service and link-local range — the credential-theft SSRF target that is
    essentially never a legitimate destination.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Hostnames that should never be a destination regardless of DNS.
_BLOCKED_HOSTNAMES = {"metadata.google.internal", "metadata"}

# Cloud instance-metadata endpoints (AWS/GCP/Azure IMDS, OpenStack, etc.).
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}


class UnsafeEgressURL(ValueError):
    """Raised when a URL is not a safe outbound destination."""


def assert_safe_egress_url(url: str, *, allow_private: bool = False) -> None:
    """Validate ``url`` is a safe outbound target, else raise ``UnsafeEgressURL``.

    Resolves the hostname and checks every resolved IP (defends against DNS that
    maps a public name to an internal address). See module docstring for the
    ``allow_private`` posture.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise UnsafeEgressURL("Invalid URL")

    if parsed.scheme not in ("http", "https"):
        raise UnsafeEgressURL("URL must use http or https")

    hostname = parsed.hostname
    if not hostname:
        raise UnsafeEgressURL("URL must include a hostname")

    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise UnsafeEgressURL("URL points to a blocked host")

    try:
        addrinfos = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror:
        raise UnsafeEgressURL("Could not resolve URL hostname")

    for *_unused, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        # Always blocked — the cloud-credential SSRF target + clearly
        # non-routable ranges. Applies even when private IPs are allowed.
        if (
            ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
            or str(ip) in _METADATA_IPS
        ):
            raise UnsafeEgressURL(f"URL resolves to a blocked address ({ip})")
        # Blocked only for untrusted (customer-supplied) URLs.
        if not allow_private and (ip.is_private or ip.is_loopback):
            raise UnsafeEgressURL(f"URL resolves to a private/internal address ({ip})")
