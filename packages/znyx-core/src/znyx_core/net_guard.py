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

DNS-rebinding note: ``assert_safe_egress_url`` only validates what DNS resolves
to *at check time*. If the caller then lets the HTTP client re-resolve the name
independently, an attacker controlling the name can return a public IP for the
check and a private/metadata IP for the real connection (TOCTOU). Callers that
POST attacker-influenced URLs should instead use ``resolve_egress_target`` and
connect to the returned, already-validated IP — see that function.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import List, NamedTuple
from urllib.parse import urlparse, urlunparse

# Hostnames that should never be a destination regardless of DNS.
_BLOCKED_HOSTNAMES = {"metadata.google.internal", "metadata"}

# Cloud instance-metadata endpoints (AWS/GCP/Azure IMDS, OpenStack, etc.).
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}


class UnsafeEgressURL(ValueError):
    """Raised when a URL is not a safe outbound destination."""


class EgressTarget(NamedTuple):
    """A validated, DNS-pinned outbound target.

    Connect to ``connect_url`` (the original URL with its hostname replaced by
    the single validated IP), while sending ``Host: host_header`` and using
    ``sni_hostname`` for TLS SNI + certificate verification. Because the IP is
    fixed here, the HTTP client never re-resolves the name, so there is no
    check-time/connect-time gap for DNS rebinding to exploit.
    """
    connect_url: str          # original URL with host replaced by the pinned IP
    host_header: str          # original host[:port] — for vhost routing
    sni_hostname: str         # original hostname — for TLS SNI + cert verification
    ip: str                   # the validated IP being connected to


def _validate_ip(ip: ipaddress._BaseAddress, *, allow_private: bool) -> None:
    """Raise ``UnsafeEgressURL`` if ``ip`` is not a permitted egress destination."""
    # Always blocked — the cloud-credential SSRF target + clearly non-routable
    # ranges. Applies even when private IPs are allowed. Loopback is EXCLUDED here:
    # IPv6 ``::1`` is otherwise flagged ``is_reserved`` and would be blocked, so a
    # co-located sidecar at ``localhost`` (→ ::1) fails. Loopback is instead governed
    # by the ``allow_private`` posture below, identically to IPv4 ``127.0.0.1``.
    if not ip.is_loopback and (
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


def _resolve_and_validate(url: str, *, allow_private: bool):
    """Parse ``url``, resolve its hostname, and validate EVERY resolved IP.

    Returns ``(parsed, hostname, ips)`` where ``ips`` is the list of validated
    ``ipaddress`` objects (order matches ``getaddrinfo``). Raises
    ``UnsafeEgressURL`` on a bad scheme/host or if any resolved IP is blocked.
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

    ips: List[ipaddress._BaseAddress] = []
    for *_unused, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        _validate_ip(ip, allow_private=allow_private)
        ips.append(ip)
    if not ips:
        raise UnsafeEgressURL("Could not resolve URL hostname")
    return parsed, hostname, ips


def assert_safe_egress_url(url: str, *, allow_private: bool = False) -> None:
    """Validate ``url`` is a safe outbound target, else raise ``UnsafeEgressURL``.

    Resolves the hostname and checks every resolved IP (defends against DNS that
    maps a public name to an internal address). See module docstring for the
    ``allow_private`` posture and the DNS-rebinding caveat.
    """
    _resolve_and_validate(url, allow_private=allow_private)


def resolve_egress_target(url: str, *, allow_private: bool = False) -> EgressTarget:
    """Validate ``url`` and return a DNS-pinned :class:`EgressTarget` to connect to.

    Like :func:`assert_safe_egress_url`, but resolves DNS exactly ONCE and hands
    back the concrete IP to dial, so the caller's HTTP client does not re-resolve
    the name (closing the DNS-rebinding TOCTOU). TLS SNI and certificate
    verification still use the original hostname via ``sni_hostname``.
    """
    parsed, hostname, ips = _resolve_and_validate(url, allow_private=allow_private)
    ip = ips[0]
    # Bracket IPv6 literals in the URL authority.
    ip_host = f"[{ip}]" if isinstance(ip, ipaddress.IPv6Address) else str(ip)
    netloc = f"{ip_host}:{parsed.port}" if parsed.port else ip_host
    connect_url = urlunparse(parsed._replace(netloc=netloc))
    host_header = f"{hostname}:{parsed.port}" if parsed.port else hostname
    return EgressTarget(
        connect_url=connect_url,
        host_header=host_header,
        sni_hostname=hostname,
        ip=str(ip),
    )
