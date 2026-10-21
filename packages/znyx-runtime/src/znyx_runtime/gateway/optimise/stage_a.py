"""Stage A techniques: on by default, never change what an answer says.

- ``minify`` (4): a text part that is entirely JSON is sent without the whitespace
  between its tokens. Lossless: strings, numbers, key order and repeated keys are kept
  exactly as written.
- ``dedupe`` (3): a large text repeated within the request (the same document or tool
  result sent twice) keeps its first copy; later copies become a short note naming the
  first. Only texts the input guard reads take part, the system prompt and the model's
  own answers are never changed, and the user text this call adds is never replaced.
- ``prompt_cache`` (1): cache markers on the stable prefix (tools, system) and on the
  conversation so far, on Anthropic's own API, when the app set none and the same prefix
  was seen recently (a marker on a prefix that is not reused costs 25% more for nothing);
  ``prompt_cache_key`` on OpenAI's own API. What was seen is remembered in this process
  only: in the shared store it would push out guardrail state (conversation history,
  session budgets, flood counters).

Minify and dedupe run before the input guard, so the guard reads exactly what is sent.
Cache markers change no text and are added after it.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from znyx_runtime.gateway import formats as fmts

MINIFY_MIN_CHARS = 200
MINIFY_MAX_CHARS = 256_000              # per part; compacting is about 30 ms per MB
MINIFY_REQUEST_CHARS = 1_000_000        # per request: the event loop is not held longer
CACHE_SHAPE_MAX_CHARS = 8_000_000       # a larger conversation is not hashed for markers
DEDUPE_MIN_CHARS = 500
CACHE_SEEN_TTL_SECONDS = 300            # Anthropic's default cache lives five minutes
_JSON_TOKEN = re.compile(r'("(?:[^"\\]|\\.)*")|[ \t\n\r]+', re.S)
# Blocks Anthropic accepts a cache marker on (thinking blocks take none).
_MARKABLE = ("text", "image", "document", "tool_use", "tool_result", "search_result")
MARKER = {"type": "ephemeral"}


# -- minify -----------------------------------------------------------------------

def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def minify_json(text: str) -> Optional[str]:
    """The compact form of ``text`` when all of it is JSON and compacting saves
    something worth sending differently for; else None."""
    if not (MINIFY_MIN_CHARS <= len(text) <= MINIFY_MAX_CHARS):
        return None
    stripped = text.strip(" \t\n\r")          # JSON whitespace only: other spaces are not JSON
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        json.loads(stripped, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None
    compact = _JSON_TOKEN.sub(lambda m: m.group(1) or "", stripped)
    if len(text) - len(compact) < max(20, len(text) // 20):
        return None
    return compact


def minify(fmt, body: Dict[str, Any], count: Callable[[str], int]) -> Tuple[int, int]:
    """Compact every JSON text part (tool results, documents, pasted data) except the
    model's own earlier answers. Returns (parts compacted, tokens saved by ``count``)."""
    changed = saved = budget = 0
    for part in fmt.input_parts(body, ("system", "developer", "user", "tool", "function")):
        if part.kind not in ("text", "tool_result") or len(part.text) < MINIFY_MIN_CHARS:
            continue
        if in_document(body, part.path):
            continue
        budget += len(part.text)
        if budget > MINIFY_REQUEST_CHARS:
            break
        compact = minify_json(part.text)
        if compact is None:
            continue
        fmt.set_text(body, part, compact)
        changed += 1
        saved += count(part.text) - count(compact)
    return changed, saved


def in_document(body: Dict[str, Any], path: Tuple[Any, ...]) -> bool:
    """Whether a part lives inside an Anthropic document or search result: their text is
    what citations point into (character offsets, quoted text), so it is never changed."""
    node: Any = body
    for step in path[:-1]:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return False
        if isinstance(node, dict) and node.get("type") in ("document", "search_result"):
            return True
    return False


# -- dedupe -----------------------------------------------------------------------

def _where(part: fmts.Part) -> str:
    """Where the first copy is, in words that stay true whatever the gateway adds to the
    request later (a canary system message shifts every message number)."""
    if part.message is None or part.role in ("system", "developer"):
        return "the system prompt"
    if part.kind == "tool_result":
        return f"an earlier {(part.tool_name or 'tool')[:60]} result"
    return "an earlier message"


def dedupe_note(first: str) -> str:
    return f"[Repeat removed by the gateway: this content is identical to {first} above.]"


def dedupe(fmt, body: Dict[str, Any], check_roles: Tuple[str, ...],
           count: Callable[[str], int]) -> Tuple[int, int]:
    """Replace later copies of a large repeated text by a note. Returns (copies
    replaced, tokens saved by ``count``)."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    last_assistant = max((i for i, m in enumerate(messages)
                          if isinstance(m, dict) and m.get("role") == "assistant"), default=-1)
    first_copy: Dict[str, str] = {}
    replaced = saved = 0
    for part in fmt.input_parts(body, check_roles):
        if (part.kind not in ("text", "tool_result") or part.role == "assistant"
                or len(part.text) < DEDUPE_MIN_CHARS or in_document(body, part.path)):
            continue                       # the model's own answers and documents are never changed
        digest = hashlib.sha256(part.text.encode("utf-8", "surrogatepass")).hexdigest()
        if digest not in first_copy:
            first_copy[digest] = _where(part)
            continue
        if part.role in ("system", "developer") or part.message is None:
            continue                       # the stable prefix is never changed
        if part.role == "user" and part.kind == "text" and part.message > last_assistant:
            continue                       # the user text this call adds is sent whole
        note = dedupe_note(first_copy[digest])
        if len(note) >= len(part.text):
            continue
        try:
            fmt.set_text(body, part, note)
        except (ValueError, TypeError):
            continue
        replaced += 1
        saved += count(part.text) - count(note)
    return replaced, saved


# -- prompt cache -----------------------------------------------------------------

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha(*pieces: str) -> str:
    h = hashlib.sha256()
    for piece in pieces:
        h.update(piece.encode("utf-8", "surrogatepass"))
        h.update(b"\x1f")
    return h.hexdigest()


def _has_marker(body: Dict[str, Any]) -> bool:
    """Whether the app placed cache markers itself, at any depth (then caching is the
    app's own: Anthropic refuses more than four markers)."""
    stack: List[Any] = [body.get("tools"), body.get("system"), body.get("messages")]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "cache_control" in item:
                return True
            stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
        elif isinstance(item, list):
            stack.extend(v for v in item if isinstance(v, (dict, list)))
    return False


class CacheShape:
    """What prompt caching needs to know about the request as the client sent it
    (before the gateway changed anything): the hash of its stable prefix, of the whole
    conversation, and of the conversation as the previous call sent it (everything
    before the last assistant turn), with rough sizes."""

    def __init__(self, api: str, body: Dict[str, Any], scope: str, model: str):
        tools = body.get("tools")
        if api == "anthropic":
            prefix = [tools, body.get("system")]
        else:
            messages = body.get("messages") if isinstance(body.get("messages"), list) else []
            lead = []
            for m in messages:
                if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                    lead.append(m)
                else:
                    break
            prefix = [tools, lead]
        prefix_text = _canonical(prefix)
        self.prefix_chars = len(prefix_text) if any(p for p in prefix) else 0
        self.prefix_hash = _sha(scope, model, prefix_text)
        self.app_managed = api == "anthropic" and _has_marker(body)
        self.total_chars = self.prefix_chars
        self.full_hash = self.previous_hash = None
        if api != "anthropic":
            return
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        chain = [self.prefix_hash]
        for message in messages:
            text = _canonical(message)
            self.total_chars += len(text)
            if self.total_chars > CACHE_SHAPE_MAX_CHARS:
                return                     # mostly images or files: not worth hashing for markers
            chain.append(_sha(chain[-1], text))
        self.full_hash = chain[-1]
        last_assistant = max((i for i, m in enumerate(messages)
                              if isinstance(m, dict) and m.get("role") == "assistant"), default=-1)
        if last_assistant >= 1:
            self.previous_hash = chain[last_assistant]


def min_cache_tokens(model: str) -> int:
    """The shortest prefix Anthropic caches for a model (shorter ones are not cached
    and cost nothing extra; this only saves a marker that cannot help)."""
    name = (model or "").lower()
    if "haiku-4" in name or "opus-4-5" in name:
        return 4096
    if "haiku" in name:
        return 2048
    return 1024


class SeenSet:
    """Hashes seen in the last five minutes, in this process (bounded, least recently
    seen dropped first). Each worker learns its own; a prefix repeats often enough that
    every worker soon has it."""

    def __init__(self, size: int = 20_000, ttl: float = CACHE_SEEN_TTL_SECONDS):
        self._items: "OrderedDict[str, float]" = OrderedDict()
        self._size, self._ttl = size, ttl

    def seen(self, key: str, record: bool = True) -> bool:
        """Whether ``key`` was seen within the TTL; ``record`` marks it seen now."""
        now = time.monotonic()
        last = self._items.get(key)
        hit = last is not None and now - last < self._ttl
        if record:
            self._items[key] = now
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return hit

    def __len__(self) -> int:
        return len(self._items)


def anthropic_host(provider) -> bool:
    """Cache markers go only to Anthropic's own API: a compatible server may refuse them."""
    return (urlsplit(provider.base_url).hostname or "").lower() == "api.anthropic.com"


def _mark_last_block(content: Any) -> Optional[Any]:
    """``content`` (a string or a list of blocks) with a cache marker on its last
    block, or None when that block cannot carry one."""
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": dict(MARKER)}] if content else None
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict) and last.get("type") in _MARKABLE:
            last["cache_control"] = dict(MARKER)
            return content
    return None


def anthropic_markers(body: Dict[str, Any], shape: CacheShape, seen: SeenSet, caller: str,
                      model: str) -> int:
    """Add up to two cache markers (the stable prefix; the conversation so far).
    Returns how many were added."""
    if shape.app_managed:
        return 0
    minimum = min_cache_tokens(model) * 3.5      # characters
    added = 0
    if shape.prefix_chars >= minimum and seen.seen(f"{caller}:p:{shape.prefix_hash[:40]}"):
        system = body.get("system")
        marked = _mark_last_block(system) if system else None
        if marked is not None:
            body["system"] = marked
            added += 1
        elif isinstance(body.get("tools"), list) and body["tools"] and isinstance(body["tools"][-1], dict):
            body["tools"][-1]["cache_control"] = dict(MARKER)
            added += 1
    if shape.full_hash is not None and shape.total_chars >= minimum:
        continuing = (shape.previous_hash is not None
                      and seen.seen(f"{caller}:c:{shape.previous_hash[:40]}", record=False))
        seen.seen(f"{caller}:c:{shape.full_hash[:40]}")
        messages = body.get("messages")
        if continuing and isinstance(messages, list) and messages and isinstance(messages[-1], dict):
            marked = _mark_last_block(messages[-1].get("content"))
            if marked is not None:
                messages[-1]["content"] = marked
                added += 1
    return added


def openai_cache_key(body: Dict[str, Any], shape: CacheShape, provider, caller: str) -> bool:
    """Set ``prompt_cache_key`` (requests with one prefix are routed to the same cache)
    on OpenAI's own API, when the app did not and the prefix is long enough to be
    cached. Other OpenAI-format servers may refuse the field, so they are left alone."""
    if provider.format != "openai" or "prompt_cache_key" in body:
        return False
    if (urlsplit(provider.base_url).hostname or "").lower() != "api.openai.com":
        return False
    if shape.prefix_chars < 1024 * 4:
        return False
    body["prompt_cache_key"] = "znyx-" + _sha(caller, shape.prefix_hash)[:32]
    return True


def cache_saving(usage: fmts.Usage, price: Dict[str, float]) -> float:
    """What the provider's cache saved on this call (its own figures): reads at the
    cached rate instead of the input rate, minus the premium paid to write the cache."""
    read = usage.cached_input_tokens or 0
    written = usage.cache_write_tokens or 0
    rate = price["input_per_mtok"]
    cached_rate = price.get("cached_input_per_mtok", rate)
    write_rate = price.get("cache_write_per_mtok", rate)
    return round((read * (rate - cached_rate) - written * (write_rate - rate)) / 1_000_000, 8)
