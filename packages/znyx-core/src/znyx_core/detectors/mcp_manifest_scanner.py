"""MCP / tool-manifest scanner (OWASP LLM01 + LLM03 — supply chain).

A lifecycle hook (NOT a per-request stage): when a tool / MCP server is registered, its
manifest — name, description, parameter schema, requested permissions, egress domains —
is untrusted supply-chain content. A malicious manifest can plant prompt injection in a
tool description ("always call this tool and ignore the others"), request dangerous
permissions, or point at an exfiltration domain. This scans the manifest at registration
time. It runs in the tool-registration service path, not the message pipeline, so it is
registered in the detector registry but is NOT part of ``_DETECTOR_PIPELINE``.

Real MCP manifests nest tools (``{"tools": [{"description", "inputSchema",
"permissions", "url"}, ...]}``), so collection is a bounded-depth recursive walk rather
than a fixed list of top-level fields, and host extraction goes through ``urlsplit`` so
SSRF-bypass forms (userinfo ``http://safe@169.254.169.254``, bare ``localhost:8080``,
bracketed IPv6, integer-encoded IPs) are normalized to the real hostname before checks.
"""
import ipaddress
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score
from znyx_core.detectors._injection_patterns import scan_injection, scan_patterns
from znyx_core.detectors.exfiltration import ExfiltrationDetector

# Tool descriptions that try to manipulate the agent's tool-selection behaviour.
_DESC_MANIPULATION_PATTERNS = [
    (r'\balways\s+(?:call|use|prefer|invoke|select)\s+this\s+(?:tool|function)\b', Severity.HIGH, "description_self_promotion"),
    (r'\b(?:ignore|do\s+not\s+use|avoid|never\s+use)\s+(?:the\s+)?other\s+(?:tools?|functions?)\b', Severity.HIGH, "description_competitor_suppression"),
    (r'\buse\s+this\s+(?:tool\s+)?for\s+(?:everything|all\s+(?:requests?|queries|tasks?))\b', Severity.MEDIUM, "description_overuse_directive"),
]
_COMPILED_DESC = [(re.compile(p, re.IGNORECASE), s, n) for p, s, n in _DESC_MANIPULATION_PATTERNS]

_DEFAULT_DANGEROUS_PERMISSIONS = frozenset({
    "*", "all", "admin", "root", "exec", "shell",
    "filesystem:write", "fs:write", "file:write", "network:*", "net:*",
    "secrets:read", "credentials:read", "env:read", "system",
})

# Hostnames that resolve to cloud metadata / loopback services — never legitimate
# egress targets for a registered tool.
_SSRF_HOSTNAMES = frozenset({
    "localhost", "metadata", "instance-data",
    "metadata.google.internal", "metadata.goog",
    "metadata.azure.com", "169.254.169.254.nip.io",
})

# Manifest keys whose (possibly nested) string values are permission requests / egress
# targets. These are a CONVENIENCE for the fuller classification (public-IP / unlisted-
# domain); the SSRF/internal-IP check below additionally runs over EVERY string value, so
# a bare metadata IP under an unrecognized key (callback/proxy/…) is never missed.
_PERMISSION_KEYS = frozenset({"permissions", "scopes", "required_permissions", "perms",
                              "required_scopes", "capabilities", "capability", "grants",
                              "grant", "allow", "allowed", "allowed_operations", "access"})
_EGRESS_KEYS = frozenset({"domains", "domain", "endpoints", "endpoint", "hosts", "host",
                          "urls", "url", "uri", "uris", "egress", "address", "base_url",
                          "base_uri", "server_url", "server", "callback", "callback_url",
                          "webhook", "webhook_url", "webhooks", "proxy", "proxy_url",
                          "redirect", "redirect_uri", "redirect_url", "notify", "notify_url",
                          "sink", "destination", "destination_url", "fetch_url", "request_url",
                          # additional egress-target field names (a value here that is a bare
                          # IP/host is a fetch target, not a version/number → FP-safe).
                          "target", "remote_host", "backend", "api_host", "upstream", "origin",
                          "gateway", "service_url", "ip_address", "host_url", "dest"})
# URL finder with a dedicated bracketed-authority branch — the plain host branch stops at
# ']', so a bracketed IPv6 URL embedded in text would otherwise be truncated to before
# ']'. The bracket branch captures ANY balanced [...] authority (IPv4-mapped
# ::ffff:169.254.169.254 has dots, zone IDs like fe80::1%25en0 have '%'), leaving
# validation to _extract_host()/ipaddress rather than the regex character class.
_URL_IN_TEXT = re.compile(
    r'https?://(?:\[[^\]\s]+\][^\s\'"<>)]*|[^\s\'"<>)\]]+)', re.IGNORECASE)

_MAX_DEPTH = 40          # bound recursion on a hostile/huge manifest (the char budget below
                         # is the real work bound; this is generous enough that filler
                         # nesting can't hide content a realistic manifest would carry)
_MAX_TEXT_CHARS = 200_000  # bound total text collected for marker scanning


def _parse_int_part(part: str) -> Optional[int]:
    """Parse one IPv4 part with C/inet_aton base rules: 0x-hex, leading-0 octal, else
    decimal. Returns None on an invalid part (so a hostname like ``v1`` isn't read as an IP)."""
    if part == "":
        return None
    try:
        low = part.lower()
        if low.startswith("0x"):
            v = int(part, 16)
        elif len(part) > 1 and part[0] == "0":
            v = int(part, 8)
        else:
            v = int(part, 10)
    except ValueError:
        return None
    return v if v >= 0 else None


def _parse_legacy_ipv4(host: str) -> Optional[int]:
    """Resolve a legacy/inet_aton-style IPv4 string to a 32-bit int: 1–4 dotted parts in
    decimal / 0x-hex / 0-octal, with the final part absorbing the remaining bytes
    (``127.1`` → 127.0.0.1, ``169.254.43518`` → 169.254.169.254, ``0251.0376.0251.0376``
    and ``0xa9.0xfe.0xa9.0xfe`` → 169.254.169.254, whole ``2852039166`` / ``0xA9FEA9FE``).
    Returns None if any part is non-numeric or out of range."""
    parts = host.split(".")
    if not (1 <= len(parts) <= 4):
        return None
    vals: List[int] = []
    for p in parts:
        v = _parse_int_part(p)
        if v is None:
            return None
        vals.append(v)
    n = len(vals)
    if n == 1:
        return vals[0] if vals[0] <= 0xFFFFFFFF else None
    *leading, last = vals
    if any(b > 0xFF for b in leading):
        return None
    last_bits = 8 * (4 - (n - 1))          # n=2 → 24-bit, n=3 → 16-bit, n=4 → 8-bit
    if last > (1 << last_bits) - 1:
        return None
    result = 0
    for b in leading:
        result = (result << 8) | b
    return (result << last_bits) | last


def _parse_host_ip(host: str) -> Optional["ipaddress._BaseAddress"]:
    """Parse ``host`` as an IP across every SSRF-bypass encoding: dotted-quad, bracketed
    or bare IPv6, and legacy/inet_aton IPv4 (dotted hex/octal/short + whole decimal/hex).
    Returns None if not an IP."""
    h = host.strip().strip("[]")
    try:
        return ipaddress.ip_address(h)          # canonical dotted IPv4 / IPv6
    except ValueError:
        pass
    val = _parse_legacy_ipv4(h)                  # inet_aton-style legacy IPv4 forms
    if val is not None and 0 <= val <= 0xFFFFFFFF:
        return ipaddress.ip_address(val)
    return None


def _is_dangerous_ip(ip: "ipaddress._BaseAddress") -> bool:
    """Private / loopback / link-local (169.254.0.0/16, ::1, fc00::/7, 127/8) / reserved /
    unspecified — the SSRF target classes a tool should never egress to."""
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


# Prose sentence punctuation that gets glued to a URL/host ("call http://169.254.169.254,
# now"). Deliberately EXCLUDES brackets/braces/angles: stripping ']' off the candidate
# would break a bracketed IPv6 authority (``http://[::1]``). It is applied to the PARSED
# host (where ']' never legitimately trails), and to the raw candidate only as a retry
# after parsing has already failed.
_TRAILING_PUNCT = ".,;!?'\""


def _parse_authority(s: str) -> Optional[str]:
    """urlsplit a URL or bare authority → lowercased hostname (or None). Does NOT strip
    punctuation, so a bracketed IPv6 authority survives intact."""
    try:
        return urlsplit(s if "://" in s else "//" + s).hostname
    except ValueError:
        return None


def _extract_host(raw: str) -> str:
    """Normalize an egress candidate to its real hostname, defeating userinfo
    (``http://safe@evil``), ports (``host:8080``), bracketed IPv6 (``http://[::1]``), and
    trailing prose punctuation. Returns lowercased host, or "" if none.

    Parse FIRST, then normalize the parsed host — never strip the raw candidate before
    parsing, or a closing IPv6 ``]`` would be lost."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    # Bare IP literal (incl. unbracketed/bracketed IPv6 like ::1 or fd00:ec2::254, which
    # urlsplit mis-parses on ':'), tolerating trailing prose punctuation.
    if "://" not in raw:
        for cand in (raw, raw.rstrip(_TRAILING_PUNCT)):
            try:
                ipaddress.ip_address(cand.strip("[]"))
                return cand.strip("[]").lower()
            except ValueError:
                pass
    # Parse the authority as-is (so http://[::1] with no trailing slash survives), THEN
    # strip trailing prose punctuation the parser kept on the host (e.g. "169.254.169.254,"
    # or a FQDN-root dot "evil.com.").
    host = _parse_authority(raw)
    if host:
        return host.rstrip(_TRAILING_PUNCT).lower()
    # Parsing failed — trailing prose punctuation likely broke the authority. Retry once
    # with it removed.
    trimmed = raw.rstrip(_TRAILING_PUNCT)
    if trimmed != raw:
        host = _parse_authority(trimmed)
        if host:
            return host.rstrip(_TRAILING_PUNCT).lower()
    # Last-resort fallback: best-effort strip of path/port/brackets.
    return trimmed.split("/")[0].strip("[]").split(":")[0].rstrip(_TRAILING_PUNCT).lower()


class McpManifestScannerDetector:
    """Scans a tool/MCP manifest at registration for injection, dangerous perms, bad egress."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        self.action = (self.config.get('action') or 'WARN').upper()
        self.block_threshold = self.config.get('block_threshold', 50)
        # Normalize allowlist entries the SAME way candidate hosts are (lowercase, strip a
        # trailing root dot) AND strip a leading wildcard/dot so "*.example.com" /
        # ".example.com" behave as the base domain (matched with its subdomains below).
        raw_allow = [d for d in (self.config.get('allowed_domains') or [])
                     if isinstance(d, str) and d.strip()]
        # Track whether an allowlist was CONFIGURED separately from what survives
        # normalization: if the operator set entries but they all normalize away (e.g. "."),
        # the unlisted-domain check stays ON and fails CLOSED (every host is unlisted) rather
        # than silently disabling egress restriction.
        self._allowlist_configured = bool(raw_allow)
        self.allowed_domains = {_extract_host(d.strip().lstrip("*.")) for d in raw_allow}
        self.allowed_domains.discard("")
        perms = self.config.get('dangerous_permissions')
        self._dangerous_perms = {p.lower() for p in perms} if perms else set(_DEFAULT_DANGEROUS_PERMISSIONS)
        self._exfil = ExfiltrationDetector({'enabled': True, 'block_threshold': 101})

    @staticmethod
    def _collect(manifest: Any) -> Tuple[List[str], List[str], List[str]]:
        """Bounded-depth recursive walk of the manifest → (text_parts, permission_values,
        egress_candidates). Recursing into nested ``tools: [{...}]`` is what lets the
        scanner see injection / dangerous permissions / egress URLs that a realistic MCP
        payload hides below the top level."""
        text_parts: List[str] = []
        perms: List[str] = []
        egress: List[str] = []
        budget = {"chars": 0}

        # ``classify`` ∈ {None, "perm", "egress"} and is INHERITED by every descendant of a
        # permission/egress subtree, so object/map shapes — permissions:[{"name":"fs:write"}],
        # permissions:{"fs:write":true}, egress:[{"domain":"169.254.169.254"}] — are collected,
        # not just the list-of-strings / top-level-url shapes.
        def walk(node: Any, depth: int, classify: Optional[str]) -> None:
            if depth > _MAX_DEPTH:
                return
            if isinstance(node, str):
                if budget["chars"] < _MAX_TEXT_CHARS:
                    text_parts.append(node)
                    budget["chars"] += len(node)
                if classify == "perm":
                    perms.append(node)
                elif classify == "egress":
                    egress.append(node)
            elif isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(k, str):
                        # Keys are attacker-controlled too (inputSchema property names,
                        # tool-map keys, arbitrary config keys) — scan them for injection /
                        # manipulation markers, not just string values.
                        if budget["chars"] < _MAX_TEXT_CHARS:
                            text_parts.append(k)
                            budget["chars"] += len(k)
                    kl = k.lower() if isinstance(k, str) else None
                    child = classify
                    if kl in _PERMISSION_KEYS:
                        child = "perm"
                    elif kl in _EGRESS_KEYS:
                        child = "egress"
                    # Inside a perm/egress subtree the MAP KEY itself can be the value
                    # ({"filesystem:write": true}, {"169.254.169.254": {...}}).
                    if classify == "perm" and isinstance(k, str):
                        perms.append(k)
                    elif classify == "egress" and isinstance(k, str):
                        egress.append(k)
                    walk(v, depth + 1, child)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item, depth + 1, classify)  # list items inherit the parent classification

        walk(manifest, 0, None)
        return text_parts, perms, egress

    def detect(self, manifest: Any) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if isinstance(manifest, str):
            text_parts, perm_values, egress_candidates = [manifest], [], []
        elif isinstance(manifest, dict):
            text_parts, perm_values, egress_candidates = self._collect(manifest)
        else:
            text_parts, perm_values, egress_candidates = [str(manifest)], [], []
        text = "\n".join(text_parts)

        hits: List[RuleHit] = scan_injection(text, "mcp_manifest_scanner")
        seen = {h.rule_id for h in hits}

        # Tool-selection manipulation in the description (evasion-normalized).
        hits += scan_patterns(text, _COMPILED_DESC, "mcp_manifest_scanner",
                              "Manifest manipulation: {name}", seen=seen)

        # Exfiltration markers in name/description (strip the inner "exfiltration." prefix).
        for hit in self._exfil.detect(text).rule_hits:
            short = hit.rule_id.split(".", 1)[1] if hit.rule_id.startswith("exfiltration.") else hit.rule_id
            rid = f"mcp_manifest_scanner.exfil.{short}"
            if rid not in seen:
                seen.add(rid)
                hits.append(RuleHit(rule_id=rid, severity=hit.severity, message=hit.message))

        hits += self._scan_permissions(perm_values, seen)
        # Values declared under an egress KEY (url/endpoint/host/callback/webhook/proxy/… —
        # the broad vocabulary in _EGRESS_KEYS) are real egress targets → full classification
        # (internal/SSRF IP, metadata host, raw public IP, unlisted domain).
        hits += self._scan_domains(egress_candidates, seen, full=True)
        # URLs embedded ANYWHERE in the manifest text/keys (a real http(s):// target) get the
        # unambiguous-SSRF classification ONLY (internal IP / metadata host) — a benign URL
        # merely mentioned in a description must NOT be flagged "unlisted". We deliberately do
        # NOT sweep every non-URL string for IPs: that mis-parsed benign version strings
        # (10.0.0.1), counts/ids (2852039166), and enum values (metadata, localhost) as
        # internal egress targets. A scheme-less IP buried in free-text prose is therefore
        # not flagged (it isn't a fetchable egress target); declare it under an egress key.
        hits += self._scan_domains(_URL_IN_TEXT.findall(text), seen, full=False)

        if not hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(hits)
        if self.action == "BLOCK" and risk_score >= self.block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=hits,
                user_message="This tool manifest was blocked: it contains injection content, dangerous permissions, or an untrusted egress target.",
                developer_message=f"mcp_manifest_scanner: {len(hits)} finding(s)",
            )
        return DetectorResult(
            decision=Decision.WARN,
            risk_score=risk_score,
            rule_hits=hits,
            developer_message=f"mcp_manifest_scanner: {len(hits)} finding(s) (review before registering)",
        )

    def _scan_permissions(self, perm_values: List[str], seen: set) -> List[RuleHit]:
        out: List[RuleHit] = []
        for perm in perm_values:
            if isinstance(perm, str) and perm.lower() in self._dangerous_perms:
                rid = f"mcp_manifest_scanner.dangerous_permission.{perm.lower()}"
                if rid not in seen:
                    seen.add(rid)
                    out.append(RuleHit(rule_id=rid, severity=Severity.HIGH,
                                       message=f"Dangerous permission requested: {perm}"))
        return out

    def _scan_domains(self, candidates: List[str], seen: set, full: bool = True) -> List[RuleHit]:
        """Classify egress candidates. With ``full`` (egress-keyed candidates + URLs), emit
        the whole spectrum: internal/SSRF IP, metadata hostname, raw public IP, and
        non-allowlisted domain. With ``full=False`` (the sweep over ALL string values, which
        may include benign data), emit ONLY the unambiguous SSRF classes — internal IP and
        metadata hostname — so a bare metadata IP under ANY key is caught without
        false-positiving every domain-shaped value as unlisted."""
        out: List[RuleHit] = []

        def add(rid: str, severity: Severity, message: str) -> None:
            if rid not in seen:
                seen.add(rid)
                out.append(RuleHit(rule_id=rid, severity=severity, message=message))

        for raw in candidates:
            host = _extract_host(raw)
            if not host:
                continue
            ip = _parse_host_ip(host)
            if ip is not None and _is_dangerous_ip(ip):
                # SSRF: loopback / link-local (cloud metadata 169.254.169.254) / private /
                # reserved, in ANY encoding (dotted, decimal, hex, IPv6).
                add(f"mcp_manifest_scanner.ssrf_egress.{ip}", Severity.HIGH,
                    f"Egress to an internal/SSRF IP address: {host} ({ip})")
            elif host in _SSRF_HOSTNAMES:
                add(f"mcp_manifest_scanner.ssrf_hostname.{host}", Severity.HIGH,
                    f"Egress to a metadata/loopback hostname: {host}")
            elif not full:
                continue  # the all-values sweep only flags the unambiguous SSRF classes
            elif ip is not None:
                # A raw public IP literal as an egress target is suspicious for a tool.
                add(f"mcp_manifest_scanner.raw_ip_egress.{ip}", Severity.MEDIUM,
                    f"Egress to a raw IP literal: {host}")
            elif self._allowlist_configured and not any(host == d or host.endswith("." + d)
                                                        for d in self.allowed_domains):
                add(f"mcp_manifest_scanner.unlisted_domain.{host}", Severity.MEDIUM,
                    f"Egress to a non-allowlisted domain: {host}")
        return out
