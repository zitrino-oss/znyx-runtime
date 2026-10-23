"""Guarded streaming for gateway calls.

The provider's stream is read as it arrives; text is released to the client only
once the streaming evaluator has checked the window that covers it (the Phase 0
evaluator: exact redactions, nothing released ahead of its check, at most one
window of added time to first token). Released text is re-emitted in the provider's
own event format, so the client's SDK assembles the response as usual. Refusals and
reasoning text are guarded the same way.

What cannot be released piece by piece is held and checked whole: tool calls (also
legacy ``function_call``), audio (its transcript is checked, then the audio is sent
in one go), Anthropic thinking blocks (signed, so they pass unchanged or not at
all) and citations. A block ends that choice with ``finish_reason: content_filter``
(OpenAI) or the message with ``stop_reason: refusal`` (Anthropic), and the provider
stream is closed so the rest of the answer is not generated and paid for.

Whatever ends a stream (its last event, a block, an error, the client leaving) the
provider stream is closed, usage is charged to the key (estimated from what was
generated when the provider's own count never arrived) and the gateway's
concurrency slot is freed: a stream holds its slot until it ends.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import anyio
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from znyx_runtime.gateway import formats as fmts

logger = logging.getLogger(__name__)


@dataclass
class _StreamState:
    usage: fmts.Usage = field(default_factory=fmts.Usage)
    guards: List["_TextGuard"] = field(default_factory=list)
    generated_chars: int = 0               # text the provider produced (for usage estimates)
    failed: bool = False                   # the provider reported an error mid-stream


class _GuardedStream(StreamingResponse):
    """The client's stream; ``cleanup`` runs however it ends."""
    owns_slot = True

    def __init__(self, content, *, cleanup, **kwargs):
        super().__init__(content, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._cleanup()


class _TextGuard:
    """The output guard for one stream of text (an OpenAI choice's content, refusal or
    reasoning, or an Anthropic text block)."""

    def __init__(self, gw, scope: Dict[str, str], policy: Dict[str, Any], record, label: str,
                 context: Optional[Dict[str, Any]] = None):
        from znyx_core.engine.streaming import StreamingEvaluator
        from znyx_runtime.api.routes import runtime_judge_ctx

        self.gw, self.policy, self.record = gw, policy, record
        from znyx_runtime.gateway.defence import active
        self.defence_text = ""
        self.defence_buffered = active(gw._controls().get("shared_defence") or {})
        self.request = gw._request(scope, record, f"stream-{label}", "", "assistant", context)
        self.ev = StreamingEvaluator(
            policy=policy, context="output", request=self.request,
            orchestrator=gw.evaluator.orchestrator, judge_ctx=runtime_judge_ctx(policy, self.request),
            remediation=gw.evaluator.remediation, mode="buffered" if self.defence_buffered else "incremental")
        self.t0 = time.perf_counter()
        self.recorded = False

    @property
    def blocked(self) -> bool:
        return self.ev.is_blocked

    async def _run(self, fn, *args) -> List[str]:
        from znyx_core.engine.evaluator import DetectorDeadlineExceeded, EvaluatorOverloadedError
        t0 = time.perf_counter()
        try:
            events = await self.gw.evaluator.run_bounded(fn, *args)
        except EvaluatorOverloadedError:
            events = self.ev.fail_closed("Evaluator saturated (fail-closed)")
        except DetectorDeadlineExceeded:
            events = self.ev.fail_closed("Evaluation deadline exceeded (fail-closed)")
        self.record.guard_ms += int((time.perf_counter() - t0) * 1000)
        return [e["data"]["text"] for e in events if e.get("event") == "chunk" and e["data"].get("text")]

    async def push(self, text: str) -> List[str]:
        if self.defence_buffered:
            self.defence_text += text
        return await self._run(self.ev.push, text)

    async def finish(self) -> List[str]:
        if self.defence_buffered:
            from znyx_core.defence import match
            if match(self.defence_text, self.gw._controls().get("shared_defence") or {}):
                self.record.rule_ids.append("shared_defence.attack_fingerprint")
                return await self._run(self.ev.fail_closed, "Reviewed attack fingerprint")
        return await self._run(self.ev.flush)

    def close(self) -> None:
        """One evaluation in telemetry per guarded stream, as /v1/evaluate/stream does."""
        if self.recorded:
            return
        self.recorded = True
        from znyx_runtime.gateway.pipeline import _worst
        decision = self.ev.final_decision
        self.record.output_decision = _worst(self.record.output_decision, decision.value)
        self.record.rule_ids.extend(h.rule_id for h in self.ev.rule_hits)
        try:
            self.gw.evaluator.record_stream(
                self.request, decision=decision, risk_score=self.ev.max_risk,
                rule_hits=self.ev.rule_hits, latency_ms=int((time.perf_counter() - self.t0) * 1000),
                context="output", policy_version=self.policy.get("policy_version"))
        except Exception:  # noqa: BLE001
            logger.debug("gateway: stream telemetry failed", exc_info=True)


async def _check_held(gw, text: str, kind: str, name: Optional[str], scope, policy, record,
                      label: str, context: Optional[Dict[str, Any]] = None):
    """One check of something held back whole (a tool call, a transcript, a thinking
    block, a citation)."""
    from znyx_runtime.gateway.pipeline import _worst
    t0 = time.perf_counter()
    verdict = await gw._check_text(text, kind, "assistant", name, "output", scope, record, policy,
                                   policy, label, context)
    record.guard_ms += int((time.perf_counter() - t0) * 1000)
    record.output_decision = _worst(record.output_decision, verdict.decision)
    record.rule_ids.extend(verdict.rule_ids)
    return verdict


def _clean(verdict, text: str) -> bool:
    """Passed on unchanged: nothing to block and nothing to change."""
    return (verdict.decision != "BLOCK" and not verdict.wants_change
            and (verdict.sanitized_text is None or verdict.sanitized_text == text))


async def stream_call(gw, request: Request, api: str, fmt, provider, url: str, headers: Dict[str, str],
                      body: Dict[str, Any], key, scope: Dict[str, str], policy: Dict[str, Any],
                      record, context: Optional[Dict[str, Any]] = None, release=None) -> Response:
    # Usage is needed for budgets and cost. Azure OpenAI rejects stream_options on older
    # API versions, so it is only added for OpenAI-format providers.
    added_usage = fmt.ensure_stream_usage(body) if provider.format == "openai" else False
    # The call was charged to rate limits and budgets on the way in.
    from znyx_runtime.gateway.pipeline import quiet_policy
    policy = quiet_policy(policy)
    t_up = time.perf_counter()
    record._upstream_started = True
    client, upstream = await gw.upstream.open_stream(provider, url, headers, body)
    if upstream.status_code != 200:
        content = await upstream.aread()
        await upstream.aclose()
        record.upstream_ms = int((time.perf_counter() - t_up) * 1000)
        record.status, record.outcome, record.reason = upstream.status_code, "error", "provider_error"
        gw._charge(key, record)
        gw._record(record)
        returned = gw.upstream.returned_headers(upstream)
        returned.update(gw.trace_headers(record))
        return Response(content=content, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type", "application/json"),
                        headers=returned)
    returned = gw.upstream.returned_headers(upstream)
    returned.update({**gw.trace_headers(record), "cache-control": "no-cache", "x-accel-buffering": "no"})
    state = _StreamState()
    events = (_openai(gw, upstream, fmt, scope, policy, record, added_usage, t_up, context, state)
              if api in ("openai", "azure") else _anthropic(gw, upstream, scope, policy, record, t_up,
                                                             context, state))
    done = False

    async def cleanup() -> None:
        nonlocal done
        if done:
            return
        done = True
        try:
            for guard in state.guards:
                guard.close()
            gw._apply_usage(record, state.usage)
            gw.estimate_missing_usage(record, state.generated_chars)
            record.status = record.status or 200
            gw._charge(key, record)
            gw._record(record)
        except Exception:  # noqa: BLE001 - recording must not keep the slot or the stream open
            logger.exception("gateway: stream cleanup failed for %s", record.request_id)
        finally:
            if release is not None:
                release()
            for closer in (events.aclose, upstream.aclose):
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    pass

    return _GuardedStream(events, cleanup=cleanup, media_type="text/event-stream", headers=returned)


async def _pump(upstream, decoder: fmts.SseDecoder, process) -> AsyncIterator[bytes]:
    """Feed the provider's bytes through ``process(event, data) -> (out, stop)``, the
    last event included even when the provider did not end it with a blank line."""
    async for raw in upstream.aiter_bytes():
        for name, data in decoder.feed(raw):
            out, stop = await process(name, data)
            for item in out:
                yield item
            if stop:
                return
    for name, data in decoder.close():
        out, stop = await process(name, data)
        for item in out:
            yield item
        if stop:
            return


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------

async def _openai(gw, upstream, fmt, scope, policy, record, added_usage: bool, t_up: float,
                  context: Optional[Dict[str, Any]], state: _StreamState) -> AsyncIterator[bytes]:
    envelope: Dict[str, Any] = {}
    # Guarded text per choice and field. Reasoning is not checked for hidden-context
    # leaks (the model restates its instructions there), only by the other checks.
    texts: Dict[Tuple[int, str], _TextGuard] = {}
    tools: Dict[int, Dict[int, Dict[str, Any]]] = {}
    functions: Dict[int, Dict[str, str]] = {}
    audio: Dict[int, Dict[str, Any]] = {}
    seen: set = set()
    finished: set = set()
    usage_chunk: Optional[Dict[str, Any]] = None
    text_fields = ("content", "refusal", "reasoning_content")

    def chunk(index: int, delta: Dict[str, Any], finish: Optional[str] = None) -> bytes:
        return fmts.sse({**envelope, "choices": [{"index": index, "delta": delta, "logprobs": None,
                                                   "finish_reason": finish}]})

    def released(texts_out: List[str], index: int, name: str) -> List[bytes]:
        if texts_out and record.ttft_ms is None:
            record.ttft_ms = int((time.perf_counter() - record.started) * 1000)
        return [chunk(index, {name: t}) for t in texts_out]

    def guard_for(index: int, name: str) -> _TextGuard:
        guard = texts.get((index, name))
        if guard is None:
            guard = texts[(index, name)] = _TextGuard(
                gw, scope, policy, record, f"{name[:1]}{index}",
                None if name == "reasoning_content" else context)
            state.guards.append(guard)
        return guard

    def withhold(index: int) -> List[bytes]:
        record.outcome = "blocked_output"
        return [chunk(index, {}, "content_filter")]

    async def finish_choice(index: int, reason: str) -> List[bytes]:
        out: List[bytes] = []
        finished.add(index)
        for name in text_fields:
            guard = texts.get((index, name))
            if guard is not None:
                out += released(await guard.finish(), index, name)
                guard.close()
                if guard.blocked:
                    return out + withhold(index)
        held = audio.get(index)
        if held is not None:
            if gw.config.settings.media_inspection:
                from znyx_runtime.gateway import media
                from znyx_runtime.gateway.pipeline import Denied
                try:
                    raw = bytearray()
                    for delta in held["deltas"]:
                        if delta.get("data"):
                            decoded, _ = media.decode(delta["data"])
                            raw.extend(decoded)
                            if len(raw) > media.MAX_BYTES:
                                raise Denied(413, "Audio exceeds the inspection limit", "media_too_large")
                    if raw:
                        await media.check(gw, bytes(raw), "audio/wav", scope, policy, record, "output")
                except Denied:
                    record.note("audio_withheld")
                    return out + withhold(index)
            verdict = await _check_held(gw, held["transcript"], "text", None, scope, policy, record,
                                        f"a{index}", context)
            if not _clean(verdict, held["transcript"]):
                return out + withhold(index)
            out += [chunk(index, {"audio": d}) for d in held["deltas"]]
        calls = tools.get(index) or {}
        if calls:
            complete = []
            for k in sorted(calls):
                call = calls[k]
                verdict = await _check_held(gw, call["arguments"], "tool_call", call["name"], scope,
                                            policy, record, f"t{index}.{k}")
                if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_tool_args is None):
                    return out + withhold(index)
                arguments = call["arguments"]
                if verdict.sanitized_tool_args is not None:
                    arguments = json.dumps(verdict.sanitized_tool_args)
                    record.note("output_rewritten")
                if call.get("type") == "custom":
                    complete.append({"index": k, "id": call["id"], "type": "custom",
                                     "custom": {"name": call["name"], "input": arguments}})
                else:
                    complete.append({"index": k, "id": call["id"], "type": call.get("type") or "function",
                                     "function": {"name": call["name"], "arguments": arguments}})
            out.append(chunk(index, {"tool_calls": complete}))
        fn = functions.get(index)
        if fn is not None:
            verdict = await _check_held(gw, fn["arguments"], "tool_call", fn["name"], scope, policy,
                                        record, f"f{index}")
            if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_tool_args is None):
                return out + withhold(index)
            arguments = fn["arguments"]
            if verdict.sanitized_tool_args is not None:
                arguments = json.dumps(verdict.sanitized_tool_args)
                record.note("output_rewritten")
            out.append(chunk(index, {"function_call": {"name": fn["name"], "arguments": arguments}}))
        return out + [chunk(index, {}, reason)]

    async def process(_event: Optional[str], data: str) -> Tuple[List[bytes], bool]:
        nonlocal envelope, usage_chunk
        out: List[bytes] = []
        if data.strip() == "[DONE]":
            return out, False
        try:
            obj = json.loads(data)
        except ValueError:
            return out, False
        if not isinstance(obj, dict):
            return out, False
        if obj.get("error") is not None and not obj.get("choices"):
            # The provider failed mid-stream: passed on as sent, so the SDK raises rather
            # than reading a cut-off answer as complete.
            record.outcome, record.reason = "error", "provider_stream_error"
            state.failed = True
            return [fmts.sse({"error": obj["error"]})], True
        if not envelope:
            envelope = {k: obj[k] for k in ("id", "object", "created", "model",
                                            "system_fingerprint", "service_tier") if k in obj}
            if record.upstream_ms is None:
                record.upstream_ms = int((time.perf_counter() - t_up) * 1000)
        if isinstance(obj.get("model"), str):
            record.response_model = obj["model"]
        if isinstance(obj.get("usage"), dict):
            state.usage.add(fmt.usage(obj))
            if not added_usage:
                usage_chunk = obj
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            index = ch.get("index", 0)
            if index in finished:
                continue
            seen.add(index)
            delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
            if "role" in delta:
                out.append(chunk(index, {"role": delta["role"], "content": ""}))
            blocked = False
            for name in text_fields:
                piece = delta.get(name)
                if isinstance(piece, str) and piece:
                    state.generated_chars += len(piece)
                    guard = guard_for(index, name)
                    out += released(await guard.push(piece), index, name)
                    if guard.blocked:
                        out += await finish_choice(index, "content_filter")
                        blocked = True
                        break
            if blocked:
                continue
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                slot = tools.setdefault(index, {}).setdefault(
                    tc.get("index", 0), {"id": None, "type": None, "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                slot["type"] = tc.get("type") or slot["type"]
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                custom = tc.get("custom") if isinstance(tc.get("custom"), dict) else {}
                if custom:
                    slot["type"] = "custom"           # free-text input instead of JSON arguments
                piece = str(fn.get("arguments") or custom.get("input") or "")
                slot["name"] += str(fn.get("name") or custom.get("name") or "")
                slot["arguments"] += piece
                state.generated_chars += len(piece)
            fc = delta.get("function_call")
            if isinstance(fc, dict):
                slot = functions.setdefault(index, {"name": "", "arguments": ""})
                slot["name"] += str(fc.get("name") or "")
                slot["arguments"] += str(fc.get("arguments") or "")
                state.generated_chars += len(str(fc.get("arguments") or ""))
            au = delta.get("audio")
            if isinstance(au, dict):
                held = audio.setdefault(index, {"deltas": [], "transcript": ""})
                held["deltas"].append(au)
                if isinstance(au.get("transcript"), str):
                    held["transcript"] += au["transcript"]
                    state.generated_chars += len(au["transcript"])
            reason = ch.get("finish_reason")
            if reason:
                out += await finish_choice(index, reason)
        # Every choice has ended and one was blocked: stop the provider generating.
        return out, bool(seen and seen <= finished and record.outcome == "blocked_output")

    async for item in _pump(upstream, fmts.SseDecoder(), process):
        yield item
    if state.failed:
        return
    for index in sorted(seen - finished):
        for item in await finish_choice(index, "stop"):
            yield item
    if usage_chunk is not None:
        yield fmts.sse({**usage_chunk})
    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------

async def _anthropic(gw, upstream, scope, policy, record, t_up: float,
                     context: Optional[Dict[str, Any]], state: _StreamState) -> AsyncIterator[bytes]:
    guards: Dict[int, _TextGuard] = {}
    tools: Dict[int, Dict[str, Any]] = {}
    thinking: Dict[int, Dict[str, Any]] = {}
    passthrough: set = set()
    usage = state.usage
    # Consecutive text blocks are one logical answer. Keep the evaluator's holdback
    # across their boundaries and emit one text block, so a secret cannot be split
    # into separately flushed blocks. Remap later indices for SDKs that index a list.
    indices: Dict[int, int] = {}
    next_index = 0
    pending_text: Optional[int] = None
    answer_guard: Optional[_TextGuard] = None

    def ev(name: str, data: Dict[str, Any]) -> bytes:
        if "index" in data:
            data = {**data, "index": indices.get(data["index"], data["index"])}
        return fmts.sse(data, event=name)

    def text_delta(index: int, text: str) -> bytes:
        if record.ttft_ms is None:
            record.ttft_ms = int((time.perf_counter() - record.started) * 1000)
        return ev("content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "text_delta", "text": text}})

    def refusal(index: Optional[int]) -> List[bytes]:
        record.outcome = "blocked_output"
        out = []
        if index is not None:
            out.append(ev("content_block_stop", {"type": "content_block_stop", "index": index}))
        out.append(ev("message_delta", {"type": "message_delta",
                                        "delta": {"stop_reason": "refusal", "stop_sequence": None},
                                        "usage": {"output_tokens": usage.output_tokens or 0}}))
        out.append(ev("message_stop", {"type": "message_stop"}))
        return out

    async def push(index: int, text: str) -> Tuple[List[bytes], bool]:
        guard = guards[index]
        state.generated_chars += len(text)
        out = [text_delta(index, t) for t in await guard.push(text)]
        if guard.blocked:
            guard.close()
            return out + refusal(index), True
        return out, False

    async def finish_text() -> Tuple[List[bytes], bool]:
        nonlocal pending_text
        if pending_text is None:
            return [], False
        index, pending_text = pending_text, None
        guard = guards[index]
        out = [text_delta(index, t) for t in await guard.finish()]
        if guard.blocked:
            return out + refusal(index), True
        return out + [ev("content_block_stop", {"type": "content_block_stop", "index": index})], False

    async def process(name: Optional[str], data: str) -> Tuple[List[bytes], bool]:
        nonlocal pending_text, next_index, answer_guard
        try:
            obj = json.loads(data)
        except ValueError:
            return [], False
        if not isinstance(obj, dict):
            return [], False
        kind = obj.get("type") or name
        if kind == "message_start":
            message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            record.response_model = message.get("model") or record.response_model
            usage.add(fmts.AnthropicMessages.usage(message))
            if record.upstream_ms is None:
                record.upstream_ms = int((time.perf_counter() - t_up) * 1000)
            return [ev("message_start", obj)], False
        if kind == "content_block_start":
            index = obj.get("index", 0)
            block = obj.get("content_block") if isinstance(obj.get("content_block"), dict) else {}
            btype = block.get("type")
            prefix: List[bytes] = []
            if btype != "text":
                prefix, stopped = await finish_text()
                if stopped:
                    return prefix, True
            merging = btype == "text" and pending_text is not None
            if merging:
                indices[index] = indices[pending_text]
            else:
                indices[index] = next_index
                next_index += 1
            if btype == "text":
                if merging:
                    guards[index] = guards[pending_text]
                    out = []
                else:
                    pending_text = index
                    if answer_guard is None:
                        answer_guard = _TextGuard(gw, scope, policy, record, f"b{index}", context)
                        state.guards.append(answer_guard)
                    # A tool/thinking block ends the emitted text block, but the
                    # next text keeps the preceding detector context.
                    guards[index] = answer_guard
                    out = [ev("content_block_start", {**obj, "content_block": {**block, "text": ""}})]
                initial = block.get("text")
                if isinstance(initial, str) and initial:
                    more, stop = await push(index, initial)
                    return out + more, stop
                return out, False
            if btype == "tool_use":
                tools[index] = {"id": block.get("id"), "name": block.get("name") or "tool", "json": ""}
            elif btype == "thinking":
                # Held whole: a signed block passes unchanged or not at all.
                thinking[index] = {"events": [ev("content_block_start", obj)],
                                   "text": str(block.get("thinking") or "")}
            else:
                passthrough.add(index)
                return prefix + [ev("content_block_start", obj)], False
            return prefix, False
        if kind == "content_block_delta":
            index = obj.get("index", 0)
            delta = obj.get("delta") if isinstance(obj.get("delta"), dict) else {}
            dtype = delta.get("type")
            if index in thinking:
                held = thinking[index]
                held["events"].append(ev("content_block_delta", obj))
                if dtype == "thinking_delta":
                    held["text"] += str(delta.get("thinking") or "")
                    state.generated_chars += len(str(delta.get("thinking") or ""))
                return [], False
            if index in passthrough:
                # Text is only ever released through a guard.
                return ([] if dtype == "text_delta" else [ev("content_block_delta", obj)]), False
            if index in guards:
                if dtype == "text_delta":
                    return await push(index, str(delta.get("text") or ""))
                if dtype == "citations_delta":
                    citation = delta.get("citation") if isinstance(delta.get("citation"), dict) else {}
                    cited = citation.get("cited_text")
                    if isinstance(cited, str) and cited:
                        verdict = await _check_held(gw, cited, "text", None, scope, policy, record,
                                                    f"cite{index}")
                        if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_text is None):
                            guards[index].close()
                            return refusal(index), True
                        if verdict.sanitized_text is not None and verdict.sanitized_text != cited:
                            obj = copy.deepcopy(obj)
                            obj["delta"]["citation"]["cited_text"] = verdict.sanitized_text
                    return [ev("content_block_delta", obj)], False
                return [], False
            if index in tools and dtype == "input_json_delta":
                piece = str(delta.get("partial_json") or "")
                tools[index]["json"] += piece
                state.generated_chars += len(piece)
            return [], False
        if kind == "content_block_stop":
            index = obj.get("index", 0)
            if index in thinking:
                held = thinking.pop(index)
                verdict = await _check_held(gw, held["text"], "text", None, scope, policy, record,
                                            f"think{index}")
                if not _clean(verdict, held["text"]):
                    return refusal(None), True
                return held["events"] + [ev("content_block_stop", obj)], False
            if index in passthrough:
                return [ev("content_block_stop", obj)], False
            if index in guards:
                return [], False               # flushed at the next non-text block or message end
            if index in tools:
                call = tools.pop(index)
                verdict = await _check_held(gw, call["json"], "tool_call", call["name"], scope, policy,
                                            record, f"t{index}")
                if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_tool_args is None):
                    return refusal(None), True
                payload = call["json"] or "{}"
                if verdict.sanitized_tool_args is not None:
                    payload = json.dumps(verdict.sanitized_tool_args)
                    record.note("output_rewritten")
                return [ev("content_block_start", {"type": "content_block_start", "index": index,
                                                   "content_block": {"type": "tool_use", "id": call["id"],
                                                                     "name": call["name"], "input": {}}}),
                        ev("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "input_json_delta",
                                                             "partial_json": payload}}),
                        ev("content_block_stop", {"type": "content_block_stop", "index": index})], False
            return [], False
        if kind == "message_delta":
            prefix, stopped = await finish_text()
            if stopped:
                return prefix, True
            if isinstance(obj.get("usage"), dict):
                out_tokens = obj["usage"].get("output_tokens")
                if out_tokens is not None:
                    usage.output_tokens = out_tokens
            return prefix + [ev("message_delta", obj)], False
        if kind == "message_stop":
            prefix, stopped = await finish_text()
            return prefix if stopped else prefix + [ev("message_stop", obj)], True
        if kind == "ping":
            return [ev("ping", obj)], False
        if kind == "error":
            record.outcome, record.reason = "error", "provider_stream_error"
            state.failed = True
            return [ev("error", obj)], True
        return [], False

    async for item in _pump(upstream, fmts.SseDecoder(), process):
        yield item
