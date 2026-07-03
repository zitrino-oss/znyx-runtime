"""
Malicious URL / Link Detector.

Scans LLM outputs for phishing URLs, suspicious domains, IP-based URLs,
data URIs, URL shorteners, homoglyph domains, and known-bad patterns.
"""
import re
import logging
import unicodedata
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.core.risk import calculate_risk_score

logger = logging.getLogger(__name__)

# ── URL extraction ─────────────────────────────────────────────────────────

_URL_RE = re.compile(
    r"(?:https?://|ftp://|//)"                   # scheme
    r"[^\s<>\"\'\)\]\},;]+"                       # URL body
    r"|"                                           # OR
    r"(?:data:)[^\s<>\"\'\)\]\},;]+",             # data URIs
    re.I,
)

# ── Known URL shorteners ──────────────────────────────────────────────────

_SHORTENERS: Set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
    "rb.gy", "bl.ink", "short.io", "v.gd", "clck.ru",
}

# ── Suspicious TLDs (high abuse rates) ────────────────────────────────────

_SUSPICIOUS_TLDS: Set[str] = {
    ".tk", ".ml", ".ga", ".cf", ".gq",  # Freenom (most abused)
    ".top", ".xyz", ".work", ".click", ".link", ".info",
    ".buzz", ".surf", ".rest", ".icu",
}

# ── Popular domains for homoglyph detection ───────────────────────────────

_POPULAR_DOMAINS: Set[str] = {
    "google.com", "facebook.com", "amazon.com", "apple.com", "microsoft.com",
    "paypal.com", "netflix.com", "instagram.com", "twitter.com", "linkedin.com",
    "github.com", "youtube.com", "yahoo.com", "dropbox.com", "chase.com",
    "bankofamerica.com", "wellsfargo.com", "citibank.com", "stripe.com",
    "openai.com", "anthropic.com",
}

# ── Homoglyph characters ─────────────────────────────────────────────────

_HOMOGLYPHS: Dict[str, str] = {
    # Cyrillic → Latin
    "а": "a", "А": "A",
    "в": "b", "В": "B",
    "с": "c", "С": "C",
    "ԁ": "d",
    "е": "e", "Е": "E",
    "ё": "e",
    "ғ": "f",
    "ɡ": "g",
    "һ": "h", "Н": "H",
    "і": "i", "І": "I",
    "ј": "j", "Ј": "J",
    "к": "k", "К": "K",
    "ӏ": "l",
    "м": "m", "М": "M",
    "н": "n",
    "о": "o", "О": "O",
    "р": "p", "Р": "P",
    "ԛ": "q",
    "г": "r",
    "ѕ": "s", "Ѕ": "S",
    "т": "t", "Т": "T",
    "у": "y",
    "ν": "v",  # Cyrillic/Greek nu
    "ԝ": "w",
    "х": "x", "Х": "X",
    "ᴢ": "z",

    # Greek → Latin
    "α": "a", "Α": "A",
    "β": "b",
    "ε": "e",
    "η": "n",
    "ι": "i",
    "κ": "k",
    "μ": "m",
    "ο": "o", "Ο": "O",
    "ρ": "p",
    "τ": "t", "Τ": "T",
    "υ": "u",
    "χ": "x",
    "ω": "w",
    "Β": "B",
    "Ε": "E",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ρ": "P",
    "Χ": "X",
    "Ζ": "Z",

    # Latin-like / modifier / extended
    "ɑ": "a",
    "ƅ": "b",
    "ϲ": "c",  # Greek lunate sigma
    "ɛ": "e",
    "ƒ": "f",
    "ɦ": "h",
    "ɩ": "i",
    "ʝ": "j",
    "ƙ": "k",
    "ℓ": "l",
    "ɱ": "m",
    "ɴ": "n",
    "ɵ": "o",
    "ƿ": "p",
    "ʀ": "r",
    "ʂ": "s",
    "ƭ": "t",
    "ʋ": "v",
    "ɯ": "w",
    "ʏ": "y",
    "ʐ": "z",

    # Fullwidth → ASCII
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e",
    "ｆ": "f", "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j",
    "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n", "ｏ": "o",
    "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t",
    "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y",
    "ｚ": "z",
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E",
    "Ｆ": "F", "Ｇ": "G", "Ｈ": "H", "Ｉ": "I", "Ｊ": "J",
    "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N", "Ｏ": "O",
    "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T",
    "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y",
    "Ｚ": "Z",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",

    # Roman numerals & other symbols
    "ⅰ": "i", "ⅱ": "ii", "ⅲ": "iii",
    "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III",

    # Dotless i / special Latin
    "ı": "i",
    "ȷ": "j",
    "ŀ": "l",
    "ɫ": "l",
    "ꝛ": "r",
    "ꜱ": "s",
}


def _normalize_homoglyphs(domain: str) -> str:
    """Replace homoglyph characters with their ASCII equivalents.

    Applies Unicode NFKC normalization first (which collapses many
    compatibility characters like fullwidth forms), then maps remaining
    confusables via the explicit table.
    """
    # NFKC normalization handles fullwidth, ligatures, compatibility forms
    domain = unicodedata.normalize("NFKC", domain)
    result = []
    for char in domain:
        result.append(_HOMOGLYPHS.get(char, char))
    return "".join(result)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


class MaliciousURLDetector:
    """Detects potentially malicious URLs in text."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.action = config.get("action", "WARN")
        self.block_ip_urls = config.get("block_ip_urls", True)
        self.block_shorteners = config.get("block_shorteners", True)
        self.check_data_uris = config.get("check_data_uris", True)
        self.max_subdomain_depth = config.get("max_subdomain_depth", 3)

        # Custom lists
        self.domain_blocklist: Set[str] = set(config.get("domain_blocklist", []))
        self.domain_allowlist: Set[str] = set(config.get("domain_allowlist", []))
        self.suspicious_tlds: Set[str] = set(config.get("suspicious_tlds", _SUSPICIOUS_TLDS))
        self.popular_domains: Set[str] = set(config.get("popular_domains", _POPULAR_DOMAINS))

    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text."""
        return _URL_RE.findall(text)

    def _check_url(self, url: str) -> List[RuleHit]:
        """Check a single URL for suspicious characteristics."""
        hits: List[RuleHit] = []

        # Data URI check
        if url.lower().startswith("data:"):
            if self.check_data_uris:
                hits.append(RuleHit(
                    rule_id="url.data_uri",
                    severity=Severity.HIGH,
                    message=f"Data URI detected (potential XSS vector): {url[:80]}",
                ))
            return hits

        # Parse URL
        try:
            parsed = urlparse(url if "://" in url else f"https:{url}")
            hostname = parsed.hostname or ""
        except Exception:
            return hits

        if not hostname:
            return hits

        hostname_lower = hostname.lower()

        # Allowlist check - if domain is explicitly allowed, skip all other checks
        for allowed in self.domain_allowlist:
            if hostname_lower == allowed or hostname_lower.endswith(f".{allowed}"):
                return []

        # Blocklist check
        for blocked in self.domain_blocklist:
            if hostname_lower == blocked or hostname_lower.endswith(f".{blocked}"):
                hits.append(RuleHit(
                    rule_id="url.blocklisted",
                    severity=Severity.HIGH,
                    message=f"Blocklisted domain: {hostname}",
                ))
                return hits

        # IP-based URL check
        if self.block_ip_urls:
            ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
            if ip_pattern.match(hostname):
                hits.append(RuleHit(
                    rule_id="url.ip_based",
                    severity=Severity.MEDIUM,
                    message=f"IP-based URL (often used in phishing): {hostname}",
                ))

        # URL shortener check
        if self.block_shorteners:
            for shortener in _SHORTENERS:
                if hostname_lower == shortener or hostname_lower.endswith(f".{shortener}"):
                    hits.append(RuleHit(
                        rule_id="url.shortener",
                        severity=Severity.MEDIUM,
                        message=f"URL shortener (destination unknown): {hostname}",
                    ))
                    break

        # Suspicious TLD check
        for tld in self.suspicious_tlds:
            if hostname_lower.endswith(tld):
                hits.append(RuleHit(
                    rule_id="url.suspicious_tld",
                    severity=Severity.LOW,
                    message=f"Suspicious TLD (high abuse rate): {hostname}",
                ))
                break

        # Punycode / IDN check
        if hostname_lower.startswith("xn--") or any(
            part.startswith("xn--") for part in hostname_lower.split(".")
        ):
            hits.append(RuleHit(
                rule_id="url.punycode",
                severity=Severity.MEDIUM,
                message=f"Punycode/IDN domain (potential homoglyph attack): {hostname}",
            ))

        # Homoglyph detection - normalize and compare to popular domains
        normalized = _normalize_homoglyphs(hostname_lower)
        # Also extract the base domain (last two labels) for comparison
        normalized_base = ".".join(normalized.split(".")[-2:])
        has_confusables = normalized != hostname_lower

        if has_confusables:
            # Contains homoglyph characters - check both full hostname and
            # base domain against popular domains
            for popular in self.popular_domains:
                if (normalized == popular
                        or normalized_base == popular
                        or _levenshtein_distance(normalized_base, popular) <= 1):
                    hits.append(RuleHit(
                        rule_id="url.homoglyph",
                        severity=Severity.HIGH,
                        message=f"Homoglyph domain resembling '{popular}': {hostname}",
                    ))
                    break
        else:
            # Also check Levenshtein distance for typosquatting (e.g., paypa1.com)
            base_domain = ".".join(hostname_lower.split(".")[-2:])
            for popular in self.popular_domains:
                if base_domain != popular and _levenshtein_distance(base_domain, popular) == 1:
                    hits.append(RuleHit(
                        rule_id="url.homoglyph",
                        severity=Severity.HIGH,
                        message=f"Typosquatting domain resembling '{popular}': {hostname}",
                    ))
                    break

        # Excessive subdomains
        parts = hostname_lower.split(".")
        if len(parts) > self.max_subdomain_depth + 2:  # +2 for domain.tld
            hits.append(RuleHit(
                rule_id="url.excessive_subdomains",
                severity=Severity.MEDIUM,
                message=f"Excessive subdomains ({len(parts)} levels): {hostname}",
            ))

        return hits

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        urls = self._extract_urls(text)
        if not urls:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        all_hits: List[RuleHit] = []
        for url in urls:
            all_hits.extend(self._check_url(url))

        if not all_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Risk score
        risk_score = calculate_risk_score(all_hits)

        # Determine decision
        if self.action == "REDACT":
            # Redact URLs from text
            sanitized = text
            for url in urls:
                sanitized = sanitized.replace(url, "[URL REDACTED]")
            decision = Decision.REDACT
        elif self.action == "BLOCK":
            sanitized = None
            decision = Decision.BLOCK
        else:
            sanitized = None
            decision = Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=all_hits,
            sanitized_text=sanitized if self.action == "REDACT" else None,
            user_message="The response contains potentially suspicious links.",
            developer_message=f"malicious_url: {len(all_hits)} suspicious URL patterns in {len(urls)} URLs",
        )
