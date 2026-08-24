"""Keyed shingle fingerprinting (OWASP LLM08 system-prompt leakage).

Privacy-preserving fingerprints: a system prompt is reduced to a set of **keyed**
hashes of its overlapping token n-grams ("shingles"). The raw prompt is never stored —
only ``HMAC-SHA256(per_org_key, shingle)`` digests. Detection HMACs the candidate
output's shingles with the SAME per-org key and counts how many match.

A per-org key (pepper) means digests are not comparable across orgs and resist offline
dictionary attack; the ``min_shingle_tokens`` floor (≥8) means trivially-guessable short
fragments are never fingerprinted. This module is dependency-free (stdlib only) so both
the control plane (builds fingerprints) and the runtime detector (matches) share it.
"""
from __future__ import annotations

import hmac
import re
from hashlib import sha256
from typing import List, Set

DEFAULT_MIN_SHINGLE_TOKENS = 8
_DIGEST_HEX = 32  # truncated 128-bit digest — compact for JSONB, ample collision margin
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _shingles(tokens: List[str], n: int) -> List[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def shingle_hashes(text: str, key: bytes, min_shingle_tokens: int = DEFAULT_MIN_SHINGLE_TOKENS) -> Set[str]:
    """Keyed digests of every ``min_shingle_tokens``-token shingle in ``text``.
    Returns an empty set when the text is too short to fingerprint."""
    out: Set[str] = set()
    for sh in _shingles(tokenize(text), min_shingle_tokens):
        out.add(hmac.new(key, sh.encode("utf-8"), sha256).hexdigest()[:_DIGEST_HEX])
    return out


def overlap_count(text: str, key: bytes, min_shingle_tokens: int, known: Set[str]) -> int:
    """How many of ``text``'s keyed shingle digests are in the ``known`` set."""
    if not known:
        return 0
    return len(shingle_hashes(text, key, min_shingle_tokens) & set(known))
