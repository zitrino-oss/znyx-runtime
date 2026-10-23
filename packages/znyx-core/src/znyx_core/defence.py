"""The shared-defence wire format. Never accepts arbitrary text or executable patterns."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone

VERSION = 1
CATEGORIES = frozenset({"prompt_injection", "jailbreak", "exfiltration", "malicious_tool"})
MAX_SIGNATURES = 10000
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(b"znyx-defence-nfkc-v1\0" + normalized.encode("utf-8")).hexdigest()


def clean_signature(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"version", "digest", "category"}:
        raise ValueError("A signature contains only version, digest and category")
    if type(value["version"]) is not int or value["version"] != VERSION:
        raise ValueError("Unsupported signature version")
    if not isinstance(value["digest"], str) or not _DIGEST.fullmatch(value["digest"]):
        raise ValueError("digest must be a lowercase SHA-256 fingerprint")
    if not isinstance(value["category"], str) or value["category"] not in CATEGORIES:
        raise ValueError("Unknown attack category")
    return dict(value)


def signature(text: str, category: str) -> dict:
    return clean_signature({"version": VERSION, "digest": fingerprint(text), "category": category})


def match(text: str, feed: object) -> bool:
    if not isinstance(feed, dict) or feed.get("enabled") is not True:
        return False
    # Expired/corrupt feeds cannot keep a withdrawn signature active indefinitely.
    try:
        until = datetime.fromisoformat(feed["expires_at"].replace("Z", "+00:00"))
        if until.tzinfo is None or until <= datetime.now(timezone.utc):
            return False
        entries = feed["signatures"]
        if not isinstance(entries, list) or len(entries) > MAX_SIGNATURES:
            return False
        digests = {clean_signature(entry)["digest"] for entry in entries}
    except (KeyError, TypeError, ValueError):
        return False
    return fingerprint(text) in digests
