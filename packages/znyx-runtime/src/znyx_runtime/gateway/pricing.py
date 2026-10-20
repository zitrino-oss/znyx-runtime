"""Cost per call from a per-model price table.

The table shipped here is a dated default: list prices in US dollars per million
tokens as published by the providers. Contracts differ and prices change, so the
gateway configuration (``prices:`` in gateway.yaml, or the bundle in managed mode)
overrides any entry and adds models. A model the table does not know gets no cost
(``None``), never a guess; usage is still recorded.

Matching: the exact model id, then the id without a date or version suffix
(``gpt-4o-2024-08-06`` -> ``gpt-4o``, ``claude-3-5-haiku-20241022`` ->
``claude-3-5-haiku``), then the longest table entry the id starts with.

Cached input is priced at ``cached_input_per_mtok`` where the table has it.
Anthropic cache writes are counted as ordinary input (they cost 25% more), so cost
for prompts that write the cache is slightly understated.
"""
from __future__ import annotations

import re
from typing import Dict, Mapping, Optional

from znyx_runtime.gateway.formats import Usage

PRICES_AS_OF = "2025-11"

DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-5": {"input_per_mtok": 1.25, "cached_input_per_mtok": 0.125, "output_per_mtok": 10.00},
    "gpt-5-mini": {"input_per_mtok": 0.25, "cached_input_per_mtok": 0.025, "output_per_mtok": 2.00},
    "gpt-5-nano": {"input_per_mtok": 0.05, "cached_input_per_mtok": 0.005, "output_per_mtok": 0.40},
    "gpt-4.1": {"input_per_mtok": 2.00, "cached_input_per_mtok": 0.50, "output_per_mtok": 8.00},
    "gpt-4.1-mini": {"input_per_mtok": 0.40, "cached_input_per_mtok": 0.10, "output_per_mtok": 1.60},
    "gpt-4.1-nano": {"input_per_mtok": 0.10, "cached_input_per_mtok": 0.025, "output_per_mtok": 0.40},
    "gpt-4o": {"input_per_mtok": 2.50, "cached_input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-4o-mini": {"input_per_mtok": 0.15, "cached_input_per_mtok": 0.075, "output_per_mtok": 0.60},
    "o3": {"input_per_mtok": 2.00, "cached_input_per_mtok": 0.50, "output_per_mtok": 8.00},
    "o4-mini": {"input_per_mtok": 1.10, "cached_input_per_mtok": 0.275, "output_per_mtok": 4.40},
    "o3-mini": {"input_per_mtok": 1.10, "cached_input_per_mtok": 0.55, "output_per_mtok": 4.40},
    # Anthropic
    "claude-opus-4-5": {"input_per_mtok": 5.00, "cached_input_per_mtok": 0.50, "output_per_mtok": 25.00},
    "claude-opus-4-1": {"input_per_mtok": 15.00, "cached_input_per_mtok": 1.50, "output_per_mtok": 75.00},
    "claude-opus-4": {"input_per_mtok": 15.00, "cached_input_per_mtok": 1.50, "output_per_mtok": 75.00},
    "claude-sonnet-4-5": {"input_per_mtok": 3.00, "cached_input_per_mtok": 0.30, "output_per_mtok": 15.00},
    "claude-sonnet-4": {"input_per_mtok": 3.00, "cached_input_per_mtok": 0.30, "output_per_mtok": 15.00},
    "claude-3-7-sonnet": {"input_per_mtok": 3.00, "cached_input_per_mtok": 0.30, "output_per_mtok": 15.00},
    "claude-3-5-sonnet": {"input_per_mtok": 3.00, "cached_input_per_mtok": 0.30, "output_per_mtok": 15.00},
    "claude-haiku-4-5": {"input_per_mtok": 1.00, "cached_input_per_mtok": 0.10, "output_per_mtok": 5.00},
    "claude-3-5-haiku": {"input_per_mtok": 0.80, "cached_input_per_mtok": 0.08, "output_per_mtok": 4.00},
}

_SUFFIX = re.compile(r"(-\d{4}-\d{2}-\d{2}|-\d{8}|-latest|@[\w.-]+)$")


class PriceTable:
    def __init__(self, overrides: Optional[Mapping[str, Mapping[str, float]]] = None):
        self.table: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
        for model, entry in (overrides or {}).items():
            self.table[model] = dict(entry)

    def lookup(self, model: Optional[str]) -> Optional[Dict[str, float]]:
        if not model:
            return None
        name = model.split("/")[-1].lower()        # "openai/gpt-4o" style prefixes
        if name in self.table:
            return self.table[name]
        stripped = _SUFFIX.sub("", name)
        if stripped in self.table:
            return self.table[stripped]
        best = None
        for candidate in self.table:
            if stripped.startswith(candidate) and (best is None or len(candidate) > len(best)):
                # "gpt-4o-mini" must not be priced as "gpt-4o": the next character after
                # the matched prefix has to end the name or start a version-like suffix.
                rest = stripped[len(candidate):]
                if rest == "" or rest[0] in "-.:@" and not rest[1:2].isalpha():
                    best = candidate
        return self.table[best] if best else None

    def cost(self, model: Optional[str], usage: Usage) -> Optional[float]:
        price = self.lookup(model)
        if price is None or usage.input_tokens is None and usage.output_tokens is None:
            return None
        cached = usage.cached_input_tokens or 0
        fresh = max(0, (usage.input_tokens or 0) - cached)
        cached_rate = price.get("cached_input_per_mtok", price["input_per_mtok"])
        total = (fresh * price["input_per_mtok"] + cached * cached_rate
                 + (usage.output_tokens or 0) * price["output_per_mtok"]) / 1_000_000
        return round(total, 8)
