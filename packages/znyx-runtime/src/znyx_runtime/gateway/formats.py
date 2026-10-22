"""The provider API formats the gateway speaks: OpenAI Chat Completions (also Azure
OpenAI and most self-hosted servers) and Anthropic Messages.

Each format knows where the text to check sits in a request and a response, how to
write a guardrail's redaction back in place, what a block looks like in that API
(so an unchanged client SDK understands it), how to read token usage, and how to
read and write its streaming events. Everything the gateway does not change is
forwarded untouched: unknown fields, parameters and future API additions pass
through.

Every place a request or a response carries text is read, not only the obvious one:
text parts and refusals, documents and search results (also inside tool results),
tool-call arguments in the history, audio transcripts, reasoning and thinking text,
citations. Content the gateway cannot read (images, audio, files) is reported as an
``opaque`` part so the gateway can count it or refuse it.

Blocks follow the conventions SDKs already handle: a blocked request is the
provider's own error shape (HTTP 400, code ``content_filter`` for OpenAI, as Azure
OpenAI returns it); a withheld response keeps the response shape with
``finish_reason: content_filter`` (OpenAI) or ``stop_reason: refusal`` (Anthropic).
"""
from __future__ import annotations

import codecs
import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

Path = Tuple[Any, ...]

# Parts that are read as text by the input guard. "history_tool_call": arguments of a
# tool call already in the conversation (the model's, or forged by the client).
TEXT_KINDS = ("text", "tool_result", "history_tool_call")


@dataclass
class Part:
    """One piece of text the guardrails look at, and where it lives."""
    path: Path
    role: str
    text: str
    # text | tool_result | tool_call | history_tool_call | audio | citation | thinking |
    # opaque (content the gateway cannot read; ``text`` is its type)
    kind: str = "text"
    tool_name: Optional[str] = None
    choice: int = 0
    message: Optional[int] = None          # index in ``messages`` (None: system prompt)
    # A text block in a list of blocks (``path[:-1]`` is the block, ``path[:-2]`` the
    # list): the blocks of one list are read together, and can be merged.
    block: bool = False

    @property
    def container(self) -> Path:
        """The parts of one container are read together (a message's text parts, one
        tool result's blocks), so text split across parts is seen as one text."""
        return self.path[:-2] if self.block else self.path


@dataclass
class Usage:
    input_tokens: Optional[int] = None           # all input, cache reads and writes included
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None    # read from a prompt cache
    cache_write_tokens: Optional[int] = None     # written to a prompt cache (Anthropic)

    def add(self, other: "Usage") -> None:
        for name in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens"):
            value = getattr(other, name)
            if value is not None:
                setattr(self, name, value)


def _get(doc: Any, path: Path) -> Any:
    for step in path:
        doc = doc[step]
    return doc


def _set(doc: Any, path: Path, value: Any) -> None:
    target = _get(doc, path[:-1])
    target[path[-1]] = value


def remove_blocks(doc: Any, parts: List[Part]) -> None:
    """Remove the list blocks these parts live in. Removed last-to-first so the paths
    of the others stay valid."""
    def order(path: Path):
        return tuple((0, p) if isinstance(p, int) else (1, str(p)) for p in path)

    for path in sorted({p.path[:-1] for p in parts if p.block}, key=order, reverse=True):
        del _get(doc, path[:-1])[path[-1]]


def sse(data: Any, event: Optional[str] = None) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode("utf-8")


def iter_sse(lines: Iterator[str]) -> Iterator[Tuple[Optional[str], str]]:
    """``(event, data)`` from SSE text lines (multi-line data joined)."""
    event, data = None, []
    for line in lines:
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "event":
            event = value
        elif name == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


class SseDecoder:
    """Incremental SSE decoding of upstream bytes. A character split across network
    reads is decoded once it is complete; ``close`` returns an event the provider
    did not end with a blank line."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        self._event: Optional[str] = None
        self._data: List[str] = []

    def feed(self, chunk: bytes) -> List[Tuple[Optional[str], str]]:
        self._buf += self._utf8.decode(chunk)
        return self._lines()

    def _lines(self) -> List[Tuple[Optional[str], str]]:
        out: List[Tuple[Optional[str], str]] = []
        while True:
            idx = self._buf.find("\n")
            if idx < 0:
                break
            line, self._buf = self._buf[:idx].rstrip("\r"), self._buf[idx + 1:]
            if line == "":
                if self._data:
                    out.append((self._event, "\n".join(self._data)))
                self._event, self._data = None, []
            elif line.startswith(":"):
                continue
            else:
                name, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if name == "event":
                    self._event = value
                elif name == "data":
                    self._data.append(value)
        return out

    def close(self) -> List[Tuple[Optional[str], str]]:
        self._buf += self._utf8.decode(b"", final=True)
        if self._buf:
            self._buf += "\n"
        out = self._lines()
        if self._data:
            out.append((self._event, "\n".join(self._data)))
            self._event, self._data = None, []
        return out


def _loads_finite(raw: bytes) -> Any:
    """JSON as a provider sends it; NaN and Infinity are refused (they cannot be sent
    on to the client as JSON)."""
    def reject(constant: str) -> Any:
        raise ValueError(f"non-finite number {constant}")
    return json.loads(raw, parse_constant=reject)


parse_response = _loads_finite


# ---------------------------------------------------------------------------
# OpenAI Chat Completions (and Azure OpenAI)
# ---------------------------------------------------------------------------

_OPENAI_ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
                       403: "permission_error", 404: "not_found_error", 408: "timeout_error",
                       413: "invalid_request_error", 429: "rate_limit_error"}


class OpenAIChat:
    name = "openai"

    @staticmethod
    def model(body: Dict[str, Any]) -> Optional[str]:
        model = body.get("model")
        return model if isinstance(model, str) else None

    @staticmethod
    def is_stream(body: Dict[str, Any]) -> bool:
        return body.get("stream") is True

    @staticmethod
    def input_parts(body: Dict[str, Any], check_roles: Tuple[str, ...]) -> List[Part]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return []
        tool_names: Dict[str, str] = {}
        parts: List[Part] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    tool_names[str(call.get("id"))] = str(call["function"].get("name") or "tool")
            if role not in check_roles:
                continue
            kind = "tool_result" if role in ("tool", "function") else "text"
            tool = (tool_names.get(str(msg.get("tool_call_id")), str(msg.get("name") or "tool"))
                    if kind == "tool_result" else None)
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(Part(("messages", i, "content"), role, content, kind, tool, message=i))
            elif isinstance(content, list):
                for j, piece in enumerate(content):
                    if not isinstance(piece, dict):
                        continue
                    ptype = piece.get("type")
                    if ptype == "text" and isinstance(piece.get("text"), str):
                        parts.append(Part(("messages", i, "content", j, "text"), role, piece["text"],
                                          kind, tool, message=i, block=True))
                    elif ptype == "refusal" and isinstance(piece.get("refusal"), str):
                        parts.append(Part(("messages", i, "content", j, "refusal"), role,
                                          piece["refusal"], kind, tool, message=i, block=True))
                    else:
                        parts.append(Part(("messages", i, "content", j), role, str(ptype or "unknown"),
                                          "opaque", message=i))
            if role == "assistant":
                if isinstance(msg.get("refusal"), str):
                    parts.append(Part(("messages", i, "refusal"), role, msg["refusal"], message=i))
                for k, call in enumerate(msg.get("tool_calls") or []):
                    fn = call.get("function") if isinstance(call, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        parts.append(Part(("messages", i, "tool_calls", k, "function", "arguments"), role,
                                          fn["arguments"], "history_tool_call",
                                          str(fn.get("name") or "tool"), message=i))
                fc = msg.get("function_call")
                if isinstance(fc, dict) and isinstance(fc.get("arguments"), str):
                    parts.append(Part(("messages", i, "function_call", "arguments"), role, fc["arguments"],
                                      "history_tool_call", str(fc.get("name") or "tool"), message=i))
        return parts

    @staticmethod
    def set_text(doc: Dict[str, Any], part: Part, text: Any) -> None:
        _set(doc, part.path, text)

    @staticmethod
    def hidden_context(body: Dict[str, Any]) -> List[str]:
        """The system and developer instructions this request carries."""
        out: List[str] = []
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict) or msg.get("role") not in ("system", "developer"):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                out.append(content)
            elif isinstance(content, list):
                out.extend(p["text"] for p in content if isinstance(p, dict)
                           and p.get("type") == "text" and isinstance(p.get("text"), str))
        return out

    @staticmethod
    def add_system_text(body: Dict[str, Any], text: str) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return
        first = messages[0] if messages else None
        if isinstance(first, dict) and first.get("role") in ("system", "developer"):
            if isinstance(first.get("content"), str):
                first["content"] = first["content"] + "\n\n" + text
                return
            if isinstance(first.get("content"), list):
                first["content"].append({"type": "text", "text": text})
                return
        messages.insert(0, {"role": "system", "content": text})

    @staticmethod
    def output_parts(response: Dict[str, Any]) -> List[Part]:
        parts: List[Part] = []
        for c, choice in enumerate(response.get("choices") or []):
            msg = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(msg, dict):
                continue
            index = choice.get("index", c)
            base = ("choices", c, "message")
            if isinstance(msg.get("content"), str):
                parts.append(Part(base + ("content",), "assistant", msg["content"], choice=index))
            elif isinstance(msg.get("content"), list):
                for j, piece in enumerate(msg["content"]):
                    if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                        parts.append(Part(base + ("content", j, "text"), "assistant", piece["text"],
                                          choice=index))
            for field in ("refusal", "reasoning_content", "reasoning"):
                if isinstance(msg.get(field), str):
                    parts.append(Part(base + (field,), "assistant", msg[field], choice=index))
            audio = msg.get("audio")
            if isinstance(audio, dict) and isinstance(audio.get("transcript"), str):
                parts.append(Part(base + ("audio", "transcript"), "assistant", audio["transcript"],
                                  "audio", choice=index))
            for k, call in enumerate(msg.get("tool_calls") or []):
                fn = call.get("function") if isinstance(call, dict) else None
                custom = call.get("custom") if isinstance(call, dict) else None
                if isinstance(fn, dict):
                    parts.append(Part(base + ("tool_calls", k, "function", "arguments"), "assistant",
                                      str(fn.get("arguments") or ""), "tool_call",
                                      str(fn.get("name") or "tool"), choice=index))
                elif isinstance(custom, dict):
                    # A custom tool (free-text input instead of JSON arguments): checked the same way.
                    parts.append(Part(base + ("tool_calls", k, "custom", "input"), "assistant",
                                      str(custom.get("input") or ""), "tool_call",
                                      str(custom.get("name") or "tool"), choice=index))
            fc = msg.get("function_call")
            if isinstance(fc, dict):
                parts.append(Part(base + ("function_call", "arguments"), "assistant",
                                  str(fc.get("arguments") or ""), "tool_call",
                                  str(fc.get("name") or "tool"), choice=index))
        return parts

    @staticmethod
    def _choices(response: Dict[str, Any], choice: int):
        for c, ch in enumerate(response.get("choices") or []):
            if isinstance(ch, dict) and ch.get("index", c) == choice:
                yield ch

    @classmethod
    def withhold_choice(cls, response: Dict[str, Any], choice: int, message: str) -> None:
        for ch in cls._choices(response, choice):
            msg = ch.get("message")
            if not isinstance(msg, dict):
                msg = ch["message"] = {"role": "assistant"}
            msg["content"] = message
            for name in ("tool_calls", "function_call", "audio", "refusal", "reasoning_content",
                         "reasoning", "annotations"):
                msg.pop(name, None)
            ch["logprobs"] = None
            ch["finish_reason"] = "content_filter"

    @classmethod
    def rewritten_choice(cls, response: Dict[str, Any], choice: int) -> None:
        """A choice whose text the guardrails changed: its log probabilities spell out
        the original tokens, so they go."""
        for ch in cls._choices(response, choice):
            if ch.get("logprobs") is not None:
                ch["logprobs"] = None

    @staticmethod
    def error(status: int, message: str, code: str = "content_filter",
              error_type: Optional[str] = None, extra: Optional[Dict[str, Any]] = None
              ) -> Tuple[int, Dict[str, Any]]:
        body: Dict[str, Any] = {"message": message,
                                "type": error_type or _OPENAI_ERROR_TYPES.get(status, "api_error"),
                                "param": None, "code": code}
        if extra:
            body["znyx"] = extra
        return status, {"error": body}

    @staticmethod
    def usage(response: Dict[str, Any]) -> Usage:
        u = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(u, dict):
            return Usage()
        details = u.get("prompt_tokens_details") or {}
        return Usage(u.get("prompt_tokens"), u.get("completion_tokens"),
                     details.get("cached_tokens") if isinstance(details, dict) else None)

    # -- streaming ----------------------------------------------------------

    @staticmethod
    def ensure_stream_usage(body: Dict[str, Any]) -> bool:
        """Ask the provider for usage in the stream (needed for budgets and cost).
        Returns True when the gateway added the option, so the extra usage chunk is
        not forwarded to a client that did not ask for it."""
        options = body.get("stream_options")
        if isinstance(options, dict) and options.get("include_usage") is True:
            return False
        body["stream_options"] = {**(options if isinstance(options, dict) else {}),
                                  "include_usage": True}
        return True


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------

_ANTHROPIC_ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
                          403: "permission_error", 404: "not_found_error",
                          413: "request_too_large", 429: "rate_limit_error",
                          500: "api_error", 502: "api_error", 503: "overloaded_error",
                          504: "api_error", 529: "overloaded_error"}

# Blocks the provider made and signs or encrypts (they cannot be edited, and a client
# cannot forge them): not read on the way in.
_PROVIDER_BLOCKS = ("thinking", "redacted_thinking", "server_tool_use", "web_search_tool_result",
                    "web_fetch_tool_result", "code_execution_tool_result",
                    "bash_code_execution_tool_result", "text_editor_code_execution_tool_result",
                    "mcp_tool_use", "mcp_tool_result", "container_upload")


def _anthropic_content(blocks: Any, base: Path, role: str, kind: str, tool: Optional[str],
                       message: Optional[int], parts: List[Part]) -> None:
    """The text inside a tool result, a document or a search result."""
    if isinstance(blocks, str):
        parts.append(Part(base, role, blocks, kind, tool, message=message))
        return
    if not isinstance(blocks, list):
        return
    for k, sub in enumerate(blocks):
        if not isinstance(sub, dict):
            continue
        here = base + (k,)
        stype = sub.get("type")
        if stype == "text" and isinstance(sub.get("text"), str):
            parts.append(Part(here + ("text",), role, sub["text"], kind, tool, message=message,
                              block=True))
        elif stype in ("document", "search_result"):
            _anthropic_document(sub, here, role, kind, tool, message, parts)
        else:
            parts.append(Part(here, role, str(stype or "unknown"), "opaque", message=message))


def _anthropic_document(block: Dict[str, Any], here: Path, role: str, kind: str,
                        tool: Optional[str], message: Optional[int], parts: List[Part]) -> None:
    for field in ("title", "context"):
        if isinstance(block.get(field), str) and block[field]:
            parts.append(Part(here + (field,), role, block[field], kind, tool, message=message))
    if block.get("type") == "search_result":
        _anthropic_content(block.get("content"), here + ("content",), role, kind, tool, message, parts)
        return
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    if source.get("type") == "text" and isinstance(source.get("data"), str):
        parts.append(Part(here + ("source", "data"), role, source["data"], kind, tool, message=message))
    elif source.get("type") == "content":
        _anthropic_content(source.get("content"), here + ("source", "content"), role, kind, tool,
                           message, parts)
    else:
        parts.append(Part(here, role, f"document:{source.get('type') or 'unknown'}", "opaque",
                          message=message))


class AnthropicMessages:
    name = "anthropic"

    @staticmethod
    def model(body: Dict[str, Any]) -> Optional[str]:
        model = body.get("model")
        return model if isinstance(model, str) else None

    @staticmethod
    def is_stream(body: Dict[str, Any]) -> bool:
        return body.get("stream") is True

    @staticmethod
    def input_parts(body: Dict[str, Any], check_roles: Tuple[str, ...]) -> List[Part]:
        parts: List[Part] = []
        if "system" in check_roles:
            system = body.get("system")
            if isinstance(system, str):
                parts.append(Part(("system",), "system", system))
            elif isinstance(system, list):
                for j, block in enumerate(system):
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(Part(("system", j, "text"), "system", str(block.get("text", "")),
                                          block=True))
        messages = body.get("messages")
        if not isinstance(messages, list):
            return parts
        tool_names: Dict[str, str] = {}
        tool_role_checked = "tool" in check_roles
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_names[str(block.get("id"))] = str(block.get("name") or "tool")
            if isinstance(content, str):
                if role in check_roles:
                    parts.append(Part(("messages", i, "content"), role, content, message=i))
                continue
            if not isinstance(content, list):
                continue
            checked = role in check_roles
            for j, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                base = ("messages", i, "content", j)
                btype = block.get("type")
                if btype == "tool_result":
                    if tool_role_checked:
                        name = tool_names.get(str(block.get("tool_use_id")), "tool")
                        _anthropic_content(block.get("content"), base + ("content",), "tool",
                                           "tool_result", name, i, parts)
                elif not checked or btype in _PROVIDER_BLOCKS:
                    continue
                elif btype == "text" and isinstance(block.get("text"), str):
                    parts.append(Part(base + ("text",), role, block["text"], message=i, block=True))
                elif btype in ("document", "search_result"):
                    _anthropic_document(block, base, role, "text", None, i, parts)
                elif btype == "tool_use":
                    parts.append(Part(base + ("input",), role, json.dumps(block.get("input") or {}),
                                      "history_tool_call", str(block.get("name") or "tool"), message=i))
                else:
                    parts.append(Part(base, role, str(btype or "unknown"), "opaque", message=i))
        return parts

    @staticmethod
    def set_text(doc: Dict[str, Any], part: Part, text: Any) -> None:
        """Write a checked text back. A tool call's ``input`` is an object: a rewritten
        one must still parse as one (ValueError otherwise)."""
        if part.kind == "history_tool_call" and isinstance(text, str):
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("tool input is not an object")
            text = value
        _set(doc, part.path, text)

    @staticmethod
    def hidden_context(body: Dict[str, Any]) -> List[str]:
        system = body.get("system")
        if isinstance(system, str):
            return [system]
        if isinstance(system, list):
            return [b["text"] for b in system if isinstance(b, dict) and b.get("type") == "text"
                    and isinstance(b.get("text"), str)]
        return []

    @staticmethod
    def add_system_text(body: Dict[str, Any], text: str) -> None:
        system = body.get("system")
        if isinstance(system, str) and system:
            body["system"] = system + "\n\n" + text
        elif isinstance(system, list):
            system.append({"type": "text", "text": text})
        else:
            body["system"] = text

    @staticmethod
    def output_parts(response: Dict[str, Any]) -> List[Part]:
        parts: List[Part] = []
        for j, block in enumerate(response.get("content") or []):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append(Part(("content", j, "text"), "assistant", block["text"]))
                for k, cite in enumerate(block.get("citations") or []):
                    if isinstance(cite, dict) and isinstance(cite.get("cited_text"), str):
                        parts.append(Part(("content", j, "citations", k, "cited_text"), "assistant",
                                          cite["cited_text"], "citation"))
            elif btype == "tool_use":
                parts.append(Part(("content", j, "input"), "assistant",
                                  json.dumps(block.get("input") or {}), "tool_call",
                                  str(block.get("name") or "tool")))
            elif btype == "thinking" and isinstance(block.get("thinking"), str):
                parts.append(Part(("content", j, "thinking"), "assistant", block["thinking"], "thinking"))
        return parts

    @staticmethod
    def withhold_choice(response: Dict[str, Any], choice: int, message: str) -> None:
        response["content"] = [{"type": "text", "text": message}]
        response["stop_reason"] = "refusal"
        response["stop_sequence"] = None

    @staticmethod
    def rewritten_choice(response: Dict[str, Any], choice: int) -> None:
        return None

    @staticmethod
    def error(status: int, message: str, code: str = "content_filter",
              error_type: Optional[str] = None, extra: Optional[Dict[str, Any]] = None
              ) -> Tuple[int, Dict[str, Any]]:
        body: Dict[str, Any] = {"type": error_type or _ANTHROPIC_ERROR_TYPES.get(status, "api_error"),
                                "message": message}
        if extra:
            body["znyx"] = {"code": code, **extra}
        return status, {"type": "error", "error": body}

    @staticmethod
    def usage(response: Dict[str, Any]) -> Usage:
        u = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(u, dict):
            return Usage()
        cached = u.get("cache_read_input_tokens")
        created = u.get("cache_creation_input_tokens")
        base = u.get("input_tokens")
        total_in = None if base is None else base + (cached or 0) + (created or 0)
        return Usage(total_in, u.get("output_tokens"), cached, created)


FORMATS = {"openai": OpenAIChat, "azure": OpenAIChat, "anthropic": AnthropicMessages}


def deep_copy(doc: Any) -> Any:
    return copy.deepcopy(doc)
