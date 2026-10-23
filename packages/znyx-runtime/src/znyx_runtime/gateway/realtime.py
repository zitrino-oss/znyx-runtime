"""Guarded Realtime WebSocket: manual audio turns, bounded complete-output inspection."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import uuid
import wave
from urllib.parse import urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect

from znyx_runtime.gateway import formats, media
from znyx_runtime.gateway.channels import checked_text, context, output_context, response_input
from znyx_runtime.gateway.defence import feed_state
from znyx_runtime.gateway.pipeline import CallRecord, Denied, quiet_policy

router = APIRouter(include_in_schema=False)
MAX_FRAME = 1024 * 1024
MAX_TURN = 16 * 1024 * 1024


def usage_cost(price, usage, record):
    if price is None:
        return None
    if any(type(usage.get(field)) is not int or usage[field] < 0 for field in ("input_tokens", "output_tokens")):
        return None
    details = usage.get("input_token_details") or usage.get("input_tokens_details") or {}
    output = usage.get("output_token_details") or usage.get("output_tokens_details") or {}
    audio_in, audio_out = details.get("audio_tokens", 0), output.get("audio_tokens", 0)
    cached = details.get("cached_tokens", 0)
    audio_cached = (details.get("cached_tokens_details") or {}).get("audio_tokens", 0)
    if type(cached) is not int or not 0 <= cached <= usage["input_tokens"]:
        return None
    if audio_in > usage["input_tokens"] or audio_out > usage["output_tokens"] or audio_cached > min(cached, audio_in):
        return None
    for name, count in (("audio_input_tokens", audio_in), ("audio_output_tokens", audio_out), ("cached_audio_input_tokens", audio_cached)):
        if type(count) is not int or count < 0:
            return None
        if count:
            record.note(name, count)
    if audio_in and "audio_input_per_mtok" not in price or audio_out and "audio_output_per_mtok" not in price:
        return None
    text_in = max(0, (usage.get("input_tokens") or 0) - audio_in)
    text_out = max(0, (usage.get("output_tokens") or 0) - audio_out)
    text_cached = max(0, cached - audio_cached)
    return round((max(0, text_in - text_cached) * price["input_per_mtok"]
        + text_cached * price.get("cached_input_per_mtok", price["input_per_mtok"])
        + text_out * price["output_per_mtok"]
        + max(0, audio_in - audio_cached) * price.get("audio_input_per_mtok", 0)
        + audio_cached * price.get("cached_audio_input_per_mtok", price.get("audio_input_per_mtok", 0))
        + audio_out * price.get("audio_output_per_mtok", 0)) / 1_000_000, 8)


def pcm_wav(data):
    if len(data) % 2:
        raise Denied(400, "PCM16 audio has an incomplete sample", "invalid_audio")
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(data)
    return out.getvalue()


def enforce_audio(config, session=True):
    """The gateway commits audio only after inspection; server VAD cannot send it early."""
    for legacy in ("input_audio_format", "output_audio_format"):
        if config.pop(legacy, "pcm16") != "pcm16":
            raise Denied(400, "Guarded realtime uses pcm16", "unsupported_audio_format")
    config.pop("turn_detection", None)
    audio = config.setdefault("audio", {})
    if session:
        if config.get("type", "realtime") != "realtime":
            raise Denied(400, "Use the guarded transcription endpoint for transcription-only calls", "unsupported_session_type")
        config["type"] = "realtime"
        audio.setdefault("input", {})["turn_detection"] = None
        # Inspection already runs offline ASR. Provider ASR would be an independently
        # billable call at commit time, before response admission.
        audio["input"]["transcription"] = None
    elif "input" in audio:
        raise Denied(400, "Response audio configuration only selects output", "invalid_audio_configuration")
    for direction in (("input", "output") if session else ("output",)):
        fmt = audio.setdefault(direction, {}).get("format") or {"type": "audio/pcm", "rate": 24000}
        if fmt.get("type") != "audio/pcm" or fmt.get("rate", 24000) != 24000:
            raise Denied(400, "Guarded realtime uses mono PCM16 at 24 kHz", "unsupported_audio_format")
        audio[direction]["format"] = {"type": "audio/pcm", "rate": 24000}


class Session:
    def __init__(self, gw, request, key, credential, provider, scope, policy, record, upstream):
        self.gw, self.request, self.key, self.credential = gw, request, key, credential
        self.provider, self.scope, self.policy, self.record, self.upstream = provider, scope, policy, record, upstream
        self.audio = bytearray()
        self.events = []
        self.event_bytes = 0
        self.active = False
        self.history = []
        self.tools = []
        self.instructions = ""
        self.items = []
        self.audio_seconds = 0.0
        self.context_tokens = 0
        self.output_modalities = ["audio"]
        self.last_history = 0
        self.last_audio_seconds = 0.0
        self.feed_state = feed_state(gw._controls().get("shared_defence") or {})

    async def refresh(self):
        self.key, _ = self.gw._identify(self.request)
        current, self.scope, self.policy = await context(self.gw, self.request, self.key, self.credential,
            {"model": self.record.model}, self.record, rate=False)
        if current != self.provider:
            raise Denied(403, "Realtime provider configuration changed; reconnect", "provider_changed")
        if feed_state(self.gw._controls().get("shared_defence") or {}) != self.feed_state:
            raise Denied(409, "Shared defence changed; reconnect to recheck the session context", "defence_changed")

    async def send(self, event):
        await self.upstream.send(json.dumps(event, separators=(",", ":")))

    async def configure(self, config, session=True):
        await self.refresh()
        if config.get("model", self.record.model) != self.record.model:
            raise Denied(403, "Reconnect to change the session model", "model_changed")
        probe = {"input": config.get("input", ""), "tools": config.get("tools", self.tools)}
        if not session and config.get("conversation") not in (None, "auto", "none"):
            raise Denied(400, "Stored conversation references cannot be inspected", "unseen_history")
        if "instructions" in config:
            probe["instructions"] = config["instructions"]
        await response_input(self.gw, probe, self.scope, self.policy, self.record)
        if "input" in config:
            config["input"] = probe["input"]
            await media.inspect_body(self.gw, config["input"], self.scope, self.policy, self.record)
            self.history.append(json.dumps(config["input"]))
        self.tools = probe["tools"]
        config["tools"] = self.tools
        if "instructions" in probe:
            config["instructions"] = probe["instructions"]
            self.instructions = probe["instructions"]
            self.history.append(probe["instructions"])
        if session:
            self.output_modalities = config.get("output_modalities", self.output_modalities)
        enforce_audio(config, session=session)

    async def client_event(self, event):
        if not isinstance(event, dict):
            raise Denied(400, "Realtime events must be JSON objects", "invalid_event")
        kind = event.get("type")
        await self.refresh()
        if kind == "session.update":
            await self.configure(event.setdefault("session", {}))
        elif kind == "input_audio_buffer.append":
            data, _ = media.decode(event.get("audio"))
            if len(self.audio) + len(data) > media.MAX_BYTES:
                raise Denied(413, "Audio turn exceeds 8 MiB", "audio_too_large")
            self.audio.extend(data)
            return  # no audio crosses the boundary before its complete inspection
        elif kind == "input_audio_buffer.clear":
            self.audio.clear()
        elif kind == "input_audio_buffer.commit":
            if not self.audio:
                raise Denied(400, "No pending audio to commit", "invalid_event")
            raw = bytes(self.audio)
            transcript = await media.check(self.gw, pcm_wav(raw), "audio/wav", self.scope, self.policy, self.record)
            self.history.append(transcript)
            self.audio_seconds += len(raw) / 48000
            self.audio.clear()
            await self.gw._permit_egress(self.provider, self.policy, len(raw), self.scope, self.record)
            await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(raw).decode()})
        elif kind == "conversation.item.create":
            item = event.get("item")
            probe = {"input": [item]}
            await response_input(self.gw, probe, self.scope, self.policy, self.record)
            await media.inspect_body(self.gw, item, self.scope, self.policy, self.record)
            self.history.append(json.dumps(item))
            self.items.append(item)
        elif kind == "response.create":
            if self.active or self.audio:
                raise Denied(409, "Finish the current response and commit pending audio before requesting a response", "turn_in_progress")
            config = event.setdefault("response", {})
            await self.configure(config, session=False)
            self.gw._check_limits(self.key, self.record.model)
            # One turn has one record and one reservation. Recheck all previously offered
            # tools under today's delegation and stop rules before generating a call.
            media_sources = getattr(self.record, "_media_sources", [])
            self.record = CallRecord(request_id="gw-" + uuid.uuid4().hex, key_id=self.key.id,
                tenant_id=self.key.tenant_id, app_id=self.key.app_id, agent_id=self.key.agent_id,
                member=self.key.member, api="realtime", model=self.record.model,
                provider=self.provider.name, provider_format=self.provider.format, stream=True)
            self.record._media_sources = media_sources
            self.record._reauthorize = lambda: self.gw.current_identity(self.request)
            self.gw._trace(self.record, self.request)
            await self.refresh()
            reserve = {"model": self.record.model, "messages": self.history,
                       "max_tokens": config.get("max_output_tokens") or 1000}
            if type(reserve["max_tokens"]) is not int or reserve["max_tokens"] <= 0:
                raise Denied(400, "Use a finite positive max_output_tokens", "invalid_output_cap")
            rates = self.gw.prices_for(self.record.member).lookup(self.record.model)
            audio_output = "audio" in config.get("output_modalities", self.output_modalities)
            if self.key.limits.spend_usd_per_day and (self.audio_seconds or audio_output):
                if not rates or "audio_input_per_mtok" not in rates or "audio_output_per_mtok" not in rates:
                    raise Denied(403, "Configure realtime audio input/output token rates for spend-limited voice", "price_unknown")
            quote = ({"input_per_mtok": max(rates["input_per_mtok"], rates.get("audio_input_per_mtok", 0)),
                      "output_per_mtok": max(rates["output_per_mtok"], rates.get("audio_output_per_mtok", 0))}
                     if rates else None)
            self.record.estimated_input_tokens = (self.context_tokens + int((self.audio_seconds - self.last_audio_seconds) * 100)
                + sum(len(text) for text in self.history[self.last_history:]) // 3)
            self.gw._reserve_call(self.key, self.record, reserve, self.provider, price_override=quote)
            config["max_output_tokens"] = reserve["max_tokens"]
            self.active = True
            self.record._upstream_started = True
        elif kind not in ("response.cancel", "conversation.item.delete", "conversation.item.truncate"):
            raise Denied(400, "Unsupported realtime event", "unsupported_event")
        if sum(len(item) for item in self.history) > MAX_TURN:
            raise Denied(413, "Realtime session context exceeds 16 MiB; reconnect", "session_too_large")
        await self.gw._permit_egress(self.provider, self.policy, len(json.dumps(event)), self.scope, self.record)
        await self.send(event)

    async def checked_output(self):
        from znyx_runtime.gateway.optimise.verify import contract_of
        if contract_of(self.policy):
            raise Denied(400, "A complete output contract now applies; reconnect using Responses", "realtime_contract_not_supported")
        def inspectable(value):
            if isinstance(value, list):
                return [inspectable(item) for item in value]
            if isinstance(value, dict):
                binary_delta = value.get("type") in ("response.output_audio.delta", "response.audio.delta")
                return {key: inspectable(item) for key, item in value.items()
                        if key not in ("audio", "data") and not (binary_delta and key == "delta")}
            return value if isinstance(value, str) else None
        # Final/done events and metadata can contain content absent from deltas. Check
        # every released event as well as the concatenated text and actual audio below.
        await checked_text(self.gw, json.dumps(inspectable(self.events)), "output", self.scope,
                           self.policy, self.record, editable=False)
        text, audio, arguments, tools = {}, {}, {}, {}
        for event in self.events:
            kind = event.get("type", "")
            slot = (event.get("output_index", 0), event.get("content_index", 0))
            if kind in ("response.output_audio.delta", "response.audio.delta"):
                data, _ = media.decode(event.get("delta"))
                audio.setdefault(slot, bytearray()).extend(data)
                if len(audio[slot]) > media.MAX_BYTES:
                    raise Denied(502, "Generated audio exceeds 8 MiB", "audio_too_large")
            elif kind in ("response.output_text.delta", "response.text.delta", "response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                text[slot] = text.get(slot, "") + str(event.get("delta") or "")
            elif kind == "response.function_call_arguments.delta":
                index = event.get("output_index", 0)
                arguments[index] = arguments.get(index, "") + str(event.get("delta") or "")
            elif kind in ("response.output_item.added", "response.output_item.done"):
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    tools[event.get("output_index", 0)] = item
        combined = "".join(text.values())
        context = output_context(self.gw, {"instructions": self.instructions, "tools": self.tools, "input": self.items}, self.record)
        await checked_text(self.gw, combined, "output", self.scope, self.policy, self.record, editable=False, context=context)
        for raw in audio.values():
            await media.check(self.gw, pcm_wav(bytes(raw)), "audio/wav", self.scope, self.policy, self.record, "output")
        # Check final content too: providers may put text/arguments only in done events.
        response = self.events[-1].get("response") or {}
        for index, item in enumerate(response.get("output") or []):
            if item.get("type") == "function_call":
                tools[index] = item
            for part in item.get("content") or []:
                for field in ("text", "transcript"):
                    if isinstance(part.get(field), str):
                        await checked_text(self.gw, part[field], "output", self.scope, self.policy, self.record, editable=False, context=context)
        for index, item in tools.items():
            raw = arguments.get(index, str(item.get("arguments") or ""))
            verdict = await self.gw._check_text(raw, "tool_call", "assistant", item.get("name"), "output",
                self.scope, self.record, self.policy, quiet_policy(self.policy), "realtime-tool")
            if verdict.decision == "BLOCK" or verdict.wants_change or verdict.sanitized_tool_args not in (None, json.loads(raw or "{}")):
                raise Denied(400, "Realtime tool call withheld", "content_filter", outcome="blocked_output")
            if raw != str(item.get("arguments") or raw):
                raise Denied(502, "Inconsistent tool arguments", "invalid_provider_response")
        self.history.append(combined)

    async def server_event(self, event):
        kind = event.get("type", "")
        if kind == "error":
            raise Denied(502, "Realtime provider refused an event", "upstream_error", outcome="error")
        if kind.startswith("response.") or self.active:
            if not self.active:
                raise Denied(502, "Provider generated an unsolicited response", "unsolicited_response")
            self.event_bytes += len(json.dumps(event))
            if self.event_bytes > MAX_TURN:
                raise Denied(502, "Realtime response exceeds 16 MiB", "response_too_large")
            self.events.append(event)
            if kind != "response.done":
                return []
            usage = (event.get("response") or {}).get("usage") or {}
            self.gw._apply_usage(self.record, formats.Usage(usage.get("input_tokens"), usage.get("output_tokens")))
            self.context_tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
            cost = usage_cost(self.gw.prices_for(self.record.member).lookup(self.record.model), usage, self.record)
            self.record.cost_usd, self.record.price_known = cost, cost is not None
            if cost is None and getattr(self.record, "_reservation", None) is not None:
                self.record.cost_usd = self.record._reservation[2]  # retain the conservative quote when usage cannot be priced
                self.record.usage_estimated = True
                if self.record.input_tokens is None or self.record.output_tokens is None:
                    self.record.input_tokens = self.record.estimated_input_tokens
                    self.record.output_tokens = max(0, self.record._reservation[1] - self.record.input_tokens)
            try:
                await self.refresh()
                await self.checked_output()
                self.last_history, self.last_audio_seconds = len(self.history), self.audio_seconds
                self.record.status = 200
                return list(self.events)
            except Denied as exc:
                self.record.status, self.record.outcome, self.record.reason = exc.status, exc.outcome, exc.code
                raise
            finally:
                self.gw._charge(self.key, self.record)
                self.gw._record(self.record)
                self.events.clear()
                self.event_bytes = 0
                self.active = False
        # Provider notifications can echo content; never relay unchecked diagnostics.
        await checked_text(self.gw, json.dumps(event), "output", self.scope, self.policy, self.record, editable=False)
        return [event]

    def close(self):
        if not self.active:
            return
        reserved = getattr(self.record, "_reservation", None)
        if reserved is not None:
            self.record.input_tokens = self.record.estimated_input_tokens
            self.record.output_tokens = max(0, reserved[1] - self.record.input_tokens)
            self.record.cost_usd = reserved[2]
            self.record.usage_estimated = True
        self.record.status, self.record.outcome, self.record.reason = 499, "error", "realtime_incomplete"
        self.gw._charge(self.key, self.record)
        self.gw._record(self.record)
        self.active = False


@router.websocket("/v1/realtime")
async def realtime(socket: WebSocket):
    gw = getattr(socket.app.state, "gateway", None)
    if gw is None or not gw.config.enabled:
        await socket.close(code=1013)
        return
    acquired, session = False, None
    try:
        key, credential = gw._identify(socket)
        if gw._slots.locked():
            raise Denied(429, "Gateway is busy", "rate_limit_exceeded")
        await gw._slots.acquire()
        acquired = True
        record = CallRecord(request_id="gw-" + uuid.uuid4().hex, key_id=key.id, tenant_id=key.tenant_id,
            app_id=key.app_id, agent_id=key.agent_id, member=key.member, api="realtime")
        model = socket.query_params.get("model")
        provider, scope, policy = await context(gw, socket, key, credential, {"model": model}, record)
        from znyx_runtime.gateway.optimise.verify import contract_of
        if contract_of(policy):
            raise Denied(400, "Use Responses or Chat for a complete output contract", "realtime_contract_not_supported")
        url = urlsplit(provider.base_url.rstrip("/") + "/realtime")
        url = urlunsplit(("wss" if url.scheme == "https" else "ws", url.netloc, url.path, urlencode({"model": model}), ""))
        await gw._permit_egress(provider, policy, 0, scope, record)
        headers = gw.upstream.headers(provider, socket.headers, credential if provider.uses_client_key else provider.credential())
        headers.pop("content-type", None)
        headers.pop("accept", None)
        protocols = socket.scope.get("subprotocols") or []
        await socket.accept(subprotocol="realtime" if "realtime" in protocols else None)
        async with connect(url, additional_headers=headers, max_size=MAX_FRAME, max_queue=16,
                           open_timeout=15, close_timeout=5, ping_interval=20) as upstream:
            session = Session(gw, socket, key, credential, provider, scope, policy, record, upstream)
            await session.send({"type": "session.update", "session": {"type": "realtime", "audio": {
                "input": {"turn_detection": None, "transcription": None, "format": {"type": "audio/pcm", "rate": 24000}},
                "output": {"format": {"type": "audio/pcm", "rate": 24000}}}}})
            async def from_client():
                while True:
                    raw = await asyncio.wait_for(socket.receive_text(), timeout=300)
                    if len(raw.encode()) > MAX_FRAME:
                        raise Denied(413, "Realtime frame exceeds 1 MiB", "frame_too_large")
                    await session.client_event(formats.parse_response(raw))
            async def from_server():
                async for raw in upstream:
                    for event in await session.server_event(formats.parse_response(raw)):
                        await socket.send_json(event)
            tasks = [asyncio.create_task(from_client()), asyncio.create_task(from_server())]
            try:
                done, _ = await asyncio.wait(tasks, timeout=3600, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if socket.client_state.name == "CONNECTED":
            await socket.send_json({"type": "error", "error": {"type": "invalid_request_error",
                "code": exc.code if isinstance(exc, Denied) else "realtime_unavailable",
                "message": exc.message if isinstance(exc, Denied) else "Realtime session could not continue"}})
            await socket.close(code=1008)
        elif socket.client_state.name == "CONNECTING":
            await socket.close(code=1008)
    finally:
        if session:
            session.close()
        if acquired:
            gw._slots.release()
