"""Token counts per provider family, for savings the provider cannot report (text the
gateway removed before sending is never counted by the provider).

With ``ZNYX_TOKENIZER=tiktoken`` and the optional ``tiktoken`` package installed (its
encoding files present: set ``TIKTOKEN_CACHE_DIR`` on an air-gapped host), OpenAI-format
text is counted exactly. Otherwise, and always for Anthropic (whose tokenizer is not
public), the count is an estimate from characters per token for that family. A saving
counted this way is flagged ``estimated`` in the ledger.
"""
from __future__ import annotations

import functools
import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = {"openai": 4.0, "azure_openai": 4.0, "anthropic": 3.5}
DEFAULT_CHARS_PER_TOKEN = 4.0
_O200K = ("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5", "o1", "o3", "o4", "chatgpt-4o")


@functools.lru_cache(maxsize=4)
def _encoding(name: str):
    try:
        import tiktoken
        return tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - not installed, or its files are missing
        logger.warning("tokens: tiktoken encoding %s unavailable (%s); estimating instead", name, exc)
        return None


def _encoder(family: str, model: Optional[str]):
    if family not in ("openai", "azure_openai") or os.environ.get("ZNYX_TOKENIZER", "").lower() != "tiktoken":
        return None
    name = (model or "").split("/")[-1].lower()
    return _encoding("o200k_base" if name.startswith(_O200K) else "cl100k_base")


def count(text: str, family: str = "openai", model: Optional[str] = None) -> int:
    """Tokens in ``text`` for a provider family (openai | azure_openai | anthropic)."""
    if not text:
        return 0
    encoder = _encoder(family, model)
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    return math.ceil(len(text) / CHARS_PER_TOKEN.get(family, DEFAULT_CHARS_PER_TOKEN))


def exact(family: str, model: Optional[str] = None) -> bool:
    """Whether ``count`` is exact for this family (a real tokenizer), not an estimate."""
    return _encoder(family, model) is not None
