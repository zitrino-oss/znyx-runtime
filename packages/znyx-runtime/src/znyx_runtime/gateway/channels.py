"""Guarded non-chat channels. Complete outputs are inspected before any bytes are released."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import uuid

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from znyx_runtime.gateway import formats, media
from znyx_runtime.gateway.pipeline import CallRecord, Denied, quiet_policy
from znyx_runtime.gateway.upstream import UpstreamError

router = APIRouter(include_in_schema=False)
CHANNELS = ("embeddings", "responses", "images/generations", "images/edits",
            "audio/speech", "audio/transcriptions", "audio/translations")
MAX_OUTPUT = 16 * 1024 * 1024


def media_cost(channel, price, body, record, image_count=None):
    if not price:
        return None
    if channel.startswith("images/") and "per_image" in price:
        count = image_count if image_count is not None else int(body.get("n") or 1)
        if not 1 <= count <= 10:
            raise Denied(400, "Image count must be between 1 and 10", "invalid_image_count")
        return round(count * price["per_image"], 8)
    if channel == "audio/speech" and "per_million_characters" in price:
        return round(len(body.get("input") or "") * price["per_million_characters"] / 1_000_000, 8)
    if channel in ("audio/transcriptions", "audio/translations") and "per_audio_minute" in price:
        seconds = getattr(record, "_media_input_seconds", None)
        if seconds is not None:
            return round(seconds / 60 * price["per_audio_minute"], 8)
    return None


def reserve_media(gw, key, channel, body, record):
    if key.limits.tokens_per_day:
        raise Denied(403, "This endpoint is billed in media units; use a spend-limited key", "unsupported_budget_meter")
    price = gw.prices_for(record.member).lookup(record.model)
    cost = media_cost(channel, price, body, record)
    record.price_known = cost is not None
    if key.limits.spend_usd_per_day:
        if cost is None:
            raise Denied(403, "Configure this model's media-unit price before using a spend-limited key", "price_unknown")
        from znyx_core.state_store import StateUnavailable, get_state_store
        day = gw._day_key(key)
        try:
            if not get_state_store().reserve(day, {"tokens": 0, "cost": cost}, {"cost": key.limits.spend_usd_per_day}, 2 * 86400):
                raise Denied(429, "Daily spend is spent or reserved", "insufficient_quota")
        except StateUnavailable:
            raise Denied(503, "Shared budgets are unavailable", "limits_unavailable") from None
        record._reservation = (day, 0, cost)
    return price, cost


def subtitles(payload, fmt):
    import math
    def stamp(value):
        seconds = float(value)
        if not math.isfinite(seconds) or not 0 <= seconds <= 86400:
            raise Denied(502, "Invalid transcript timing", "invalid_provider_response")
        ms = round(seconds * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}" + ("," if fmt == "srt" else ".") + f"{ms % 1000:03}"
    segments = payload.get("segments") or [{"start": 0, "end": payload.get("duration") or 1, "text": payload["text"]}]
    chunks = ["WEBVTT\n"] if fmt == "vtt" else []
    for index, segment in enumerate(segments, 1):
        text = str(segment.get("text") or "").replace("-->", "→")
        chunks.append(f"{index}\n{stamp(segment.get('start', 0))} --> {stamp(segment.get('end', 0))}\n{text}\n")
    return "\n".join(chunks)


def embedding_text(value, model):
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value or not all(type(token) is int and token >= 0 for token in value):
        raise Denied(400, "Embedding input must be text or a valid token array", "uninspectable_embedding_input")
    try:
        import tiktoken
        # No guessed fallback encoding: the decoded text must be what this model sees.
        return tiktoken.encoding_for_model(model).decode(value, errors="strict")
    except (ImportError, KeyError, ValueError, UnicodeError, OverflowError):
        raise Denied(400, "Install the model tokenizer or send embedding input as text", "uninspectable_embedding_input") from None


async def context(gw, request, key, credential, body, record, deployment=None, rate=True):
    family = "azure" if deployment else "openai"
    try:
        provider = gw.config.provider_for(key, family, request.headers.get("x-znyx-provider"))
    except LookupError as exc:
        raise Denied(403, str(exc), "provider_not_available") from None
    if provider.blocked:
        raise Denied(403, provider.blocked, "provider_not_allowed_here")
    model = body.get("model") or deployment
    if not isinstance(model, str) or not model:
        raise Denied(400, "The request names no model", "model_required")
    if not key.model_allowed(deployment or model):
        raise Denied(403, "Model is not approved for this key", "model_not_allowed")
    if provider.uses_client_key and not credential:
        raise Denied(401, "Supply your provider credential with X-Znyx-Key", "provider_key_required")
    record.model, record.provider, record.provider_format = model, provider.name, provider.format
    record.user = gw._user_of(request, body)
    if rate:
        gw._check_limits(key, model)
    member = gw.bundle_manager.member_by_key(key.member or None)
    delegation = gw.delegation_context(request, key, member)
    gw.apply_delegation(record, delegation)
    record._stops = gw._check_controls(key, member, model, provider, deployment,
                                      person=delegation.person if delegation else None)
    scope = gw._scope(key, request, member)
    policy = gw.bundle_manager.get_policy(tenant_id=scope["tenant_id"], app_id=scope["app_id"],
        agent_id=scope["agent_id"], env=scope["env"], member=member)
    return provider, scope, policy


async def checked_text(gw, text, stage, scope, policy, record, kind="text", tool=None, editable=True, context=None):
    verdict = await gw._check_text(text, kind, "user" if stage == "input" else "assistant", tool,
        stage, scope, record, policy, quiet_policy(policy), "channel", context)
    record.rule_ids.extend(verdict.rule_ids)
    if stage == "input":
        record.input_decision = verdict.decision
    else:
        record.output_decision = verdict.decision
    changed = verdict.wants_change or verdict.sanitized_text not in (None, text)
    if verdict.decision == "BLOCK" or (changed and (not editable or verdict.sanitized_text is None)):
        raise Denied(400, "Content was withheld by policy", "content_filter", outcome="blocked_" + stage)
    return verdict.sanitized_text if verdict.sanitized_text is not None else text


async def bounded_response(client, url, headers, *, body=None, data=None, files=None):
    kwargs = {"json": body} if body is not None else {"data": data, "files": files}
    async with client.stream("POST", url, headers=headers, **kwargs) as response:
        out = bytearray()
        async for chunk in response.aiter_bytes():
            if len(out) + len(chunk) > MAX_OUTPUT:
                raise Denied(502, "Provider response exceeds 16 MiB", "provider_response_too_large", outcome="error")
            out.extend(chunk)
        return httpx.Response(response.status_code, headers=response.headers, content=bytes(out))


async def multipart(request, max_bytes):
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > max_bytes:
            raise Denied(413, "Request exceeds the configured limit", "request_too_large")
        raw.extend(chunk)
    request._body = bytes(raw)
    data, files = {}, []
    async with request.form(max_files=32, max_fields=100, max_part_size=max_bytes) as form:
        for name, value in form.multi_items():
            if hasattr(value, "read"):
                content = await value.read(media.MAX_BYTES + 1)
                if len(content) > media.MAX_BYTES:
                    raise Denied(413, "Uploaded media exceeds 8 MiB", "media_too_large")
                files.append((name, (value.filename or "media", content, value.content_type or "application/octet-stream")))
            elif name in data:
                raise Denied(400, "Duplicate form fields are refused", "invalid_request")
            else:
                data[name] = str(value)
    return data, files


async def response_input(gw, body, scope, policy, record):
    if any(body.get(name) for name in ("previous_response_id", "conversation", "background")):
        raise Denied(400, "Send complete inline history; stored/background responses cannot be guarded", "unseen_history")
    body["store"] = False
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") not in ("function", "custom"):
            raise Denied(400, "Provider-hosted tools require a separately governed connector", "provider_tool_not_allowed")
        name = tool.get("name") or "tool"
        delegated = getattr(record, "_delegated_tools", None)
        from znyx_core.identity import tool_granted
        if delegated is not None and not tool_granted(delegated, name):
            raise Denied(403, "Tool is outside this person's delegation", "tool_not_delegated")
        if record._stops.quarantined or record._stops.tool_stop(name):
            raise Denied(403, "The offered tool is stopped", "tool_stopped")
        # Apply the existing tools policy to the offered definition too.
        offered = {"tools": [{"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}]}
        gw._filter_tools("openai", offered, policy, record)
        if not offered["tools"]:
            raise Denied(403, "Tool is not allowed by policy", "tool_not_allowed")
    if isinstance(body.get("instructions"), str):
        body["instructions"] = await checked_text(gw, body["instructions"], "input", scope, policy, record)
    value = body.get("input")
    if isinstance(value, str):
        body["input"] = await checked_text(gw, value, "input", scope, policy, record)
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                raise Denied(400, "Responses input items must be objects", "invalid_request")
            if item.get("type") in ("function_call", "custom_tool_call"):
                field = "arguments" if item["type"] == "function_call" else "input"
                item[field] = await checked_text(gw, str(item.get(field) or ""), "input", scope, policy, record)
            elif item.get("type") in ("function_call_output", "custom_tool_call_output"):
                value = item.get("output")
                if isinstance(value, str):
                    item["output"] = await checked_text(gw, value, "input", scope, policy, record, kind="tool_result")
                elif isinstance(value, list):
                    await response_parts(gw, value, scope, policy, record, kind="tool_result")
                else:
                    raise Denied(400, "Tool output must contain inline text or content parts", "invalid_request")
            else:
                if item.get("type", "message") != "message" or item.get("role") not in ("user", "assistant", "system", "developer"):
                    raise Denied(400, "Unknown or stored input items cannot be inspected", "unseen_history")
                content = item.get("content")
                if isinstance(content, str):
                    item["content"] = await checked_text(gw, content, "input", scope, policy, record)
                elif isinstance(content, list):
                    await response_parts(gw, content, scope, policy, record)
                else:
                    raise Denied(400, "Messages require inline content", "invalid_request")
    else:
        raise Denied(400, "Responses requires inline input", "invalid_request")


async def response_parts(gw, parts, scope, policy, record, kind="text"):
    allowed = {"input_text", "output_text", "input_image", "input_file", "input_audio"}
    if any(not isinstance(part, dict) or part.get("type") not in allowed for part in parts):
        raise Denied(400, "Unknown content parts cannot be inspected", "uninspectable_content")
    texts = [part for part in parts if part["type"] in ("input_text", "output_text")]
    if any(not isinstance(part.get("text"), str) for part in texts):
        raise Denied(400, "Text parts must contain text", "invalid_request")
    if texts:
        text = await checked_text(gw, "".join(part["text"] for part in texts), "input", scope, policy, record, kind=kind)
        for index, part in enumerate(texts):
            part["text"] = text if index == 0 else ""


def output_context(gw, body, record):
    messages = [{"role": "system", "content": body.get("instructions") or ""}]
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") in ("function_call_output", "custom_tool_call_output"):
            value = item.get("output")
            messages.append({"role": "tool", "content": value if isinstance(value, str) else json.dumps(value)})
    synthetic = {"messages": messages, "tools": [{"type": "function", "function": tool} for tool in body.get("tools") or []]}
    context = gw._output_context(formats.OpenAIChat, synthetic)
    if getattr(record, "_media_sources", None):
        sources = context.setdefault("grounding_sources", [])
        sources.extend({"text": text, "id": f"media-{i}"} for i, text in enumerate(record._media_sources))
        context["source_context"] = "\n\n".join(source["text"] for source in sources)
    return context


async def response_output(gw, payload, scope, policy, record, context=None):
    output = payload.get("output")
    if not isinstance(output, list):
        raise Denied(502, "Malformed Responses result", "invalid_provider_response", outcome="error")
    texts = []
    for item in output:
        if not isinstance(item, dict):
            raise Denied(502, "Malformed Responses output item", "invalid_provider_response")
        kind = item.get("type")
        if kind == "message":
            parts = item.get("content") or []
            if any(not isinstance(part, dict) or part.get("type") not in ("output_text", "refusal") for part in parts):
                raise Denied(502, "Unknown Responses content cannot be inspected", "unsupported_provider_output")
            for part in parts:
                if part.get("annotations"):
                    await checked_text(gw, json.dumps(part["annotations"]), "output", scope, policy, record, editable=False)
            texts.extend(parts)
        elif kind in ("function_call", "custom_tool_call"):
            field = "arguments" if kind == "function_call" else "input"
            raw = str(item.get(field) or "")
            verdict = await gw._check_text(raw, "tool_call", "assistant", item.get("name"), "output",
                scope, record, policy, quiet_policy(policy), "response-tool")
            record.rule_ids.extend(verdict.rule_ids)
            if verdict.decision == "BLOCK" or (verdict.wants_change and verdict.sanitized_tool_args is None):
                raise Denied(400, "Tool call withheld by policy", "content_filter", outcome="blocked_output")
            if verdict.sanitized_tool_args is not None:
                item[field] = json.dumps(verdict.sanitized_tool_args)
        elif kind == "reasoning":
            if item.get("encrypted_content") or item.get("content"):
                raise Denied(502, "Opaque reasoning content cannot be inspected", "unsupported_provider_output")
            for part in item.get("summary") or []:
                if isinstance(part.get("text"), str):
                    part["text"] = await checked_text(gw, part["text"], "output", scope, policy, record)
        else:
            raise Denied(502, "Unknown Responses output cannot be inspected", "unsupported_provider_output")
    if texts:
        raw = "".join(p.get("text") or p.get("refusal") or "" for p in texts)
        text = await checked_text(gw, raw, "output", scope, policy, record, context=context)
        for index, part in enumerate(texts):
            field = "refusal" if part.get("type") == "refusal" else "text"
            part[field] = text if index == 0 else ""
            if text != raw:
                part.pop("annotations", None)  # offsets no longer refer to the released text
        from znyx_runtime.gateway.optimise.verify import contract_of, check_contract
        contract = contract_of(policy)
        if contract and str(contract.on_fail or "").lower() == "block" and await asyncio.to_thread(check_contract, text, contract):
            raise Denied(400, "Response failed the output contract", "content_filter", outcome="blocked_output")
    return payload


def response_events(payload):
    """Rebuild a buffered Responses stream only from the checked result."""
    sequence = 0
    def event(kind, **fields):
        nonlocal sequence
        data = {"type": kind, "sequence_number": sequence, **fields}
        sequence += 1
        return formats.sse(data, kind)
    yield event("response.created", response={**payload, "status": "in_progress", "output": []})
    for index, item in enumerate(payload["output"]):
        empty = {**item, "status": "in_progress"}
        if item.get("type") == "message":
            empty["content"] = []
        elif item.get("type") in ("function_call", "custom_tool_call"):
            empty["arguments" if item["type"] == "function_call" else "input"] = ""
        yield event("response.output_item.added", output_index=index, item=empty)
        if item.get("type") == "message":
            for ci, part in enumerate(item.get("content") or []):
                common = {"output_index": index, "item_id": item.get("id"), "content_index": ci}
                field = "refusal" if part.get("type") == "refusal" else "text"
                yield event("response.content_part.added", **common, part={**part, field: ""})
                kind = "response.refusal" if field == "refusal" else "response.output_text"
                yield event(kind + ".delta", **common, delta=part.get(field, ""))
                yield event(kind + ".done", **common, **{field: part.get(field, "")})
                yield event("response.content_part.done", **common, part=part)
        elif item.get("type") == "function_call":
            common = {"output_index": index, "item_id": item.get("id")}
            yield event("response.function_call_arguments.delta", **common, delta=item.get("arguments", ""))
            yield event("response.function_call_arguments.done", **common, arguments=item.get("arguments", ""))
        yield event("response.output_item.done", output_index=index, item=item)
    yield event("response.completed", response=payload)


async def handle(request, channel, deployment=None):
    gw = getattr(request.app.state, "gateway", None)
    if gw is None or not gw.config.enabled:
        return JSONResponse({"error": {"message": "Gateway is not configured"}}, status_code=503)
    record = CallRecord(request_id="gw-" + uuid.uuid4().hex, key_id="", tenant_id="", app_id="", agent_id="", api=channel)
    gw._trace(record, request)
    key = None
    acquired = False
    try:
        key, credential = gw._identify(request)
        record.key_id, record.tenant_id, record.app_id, record.agent_id, record.member = (
            key.id, key.tenant_id, key.app_id, key.agent_id, key.member)
        record._reauthorize = lambda: gw.current_identity(request)
        if gw._slots.locked():
            raise Denied(429, "Gateway is busy", "rate_limit_exceeded")
        await gw._slots.acquire()
        acquired = True
        files = []
        if "multipart/form-data" in request.headers.get("content-type", ""):
            body, files = await multipart(request, gw.config.settings.max_request_bytes)
        else:
            body = await gw._read_body(request)
        provider, scope, policy = await context(gw, request, key, credential, body, record, deployment)
        if channel.startswith("images/"):
            count = body.get("n", 1)
            if isinstance(count, str) and count.isdigit():
                count = int(count)
            if type(count) is not int or not 1 <= count <= 10:
                raise Denied(400, "Image count must be between 1 and 10", "invalid_image_count")
        if channel == "audio/speech" and (not isinstance(body.get("input"), str) or not 1 <= len(body["input"]) <= 4096):
            raise Denied(400, "Speech input must contain 1..4096 text characters", "invalid_request")
        record.stream = body.get("stream") is True
        if channel == "embeddings":
            inputs = body.get("input")
            if isinstance(inputs, str) or (isinstance(inputs, list) and inputs and all(type(i) is int for i in inputs)):
                body["input"] = await checked_text(gw, embedding_text(inputs, body["model"]), "input", scope, policy, record)
            elif isinstance(inputs, list) and inputs:
                body["input"] = [await checked_text(gw, embedding_text(item, body["model"]), "input", scope, policy, record) for item in inputs]
            else:
                raise Denied(400, "Send embedding input as text so it can be inspected", "uninspectable_embedding_input")
        elif channel == "responses":
            await response_input(gw, body, scope, policy, record)
        else:
            for field in ("input", "prompt", "instructions"):
                if isinstance(body.get(field), str):
                    body[field] = await checked_text(gw, body[field], "input", scope, policy, record)
        await media.inspect_body(gw, body, scope, policy, record)
        for _, (_, data, mime) in files:
            await media.check(gw, data, mime, scope, policy, record)
        # OCR/ASR may take minutes. Recheck the identity and controls before releasing
        # inspected bytes to a provider; a policy change requires a fresh inspection.
        current_key, current_credential = gw._identify(request)
        current_provider, current_scope, current_policy = await context(
            gw, request, current_key, current_credential, body, record, deployment, rate=False)
        if (current_key != key or current_provider != provider or current_scope != scope or current_policy != policy):
            raise Denied(409, "Gateway policy changed during inspection; retry the request", "policy_changed")
        if channel == "images/generations":
            # Legacy models otherwise return opaque, time-limited URLs.
            if str(body["model"]).startswith("dall-e"):
                body["response_format"] = "b64_json"
        if channel in ("audio/transcriptions", "audio/translations"):
            if not files:
                raise Denied(400, "An audio file is required", "invalid_request")
            requested_format = body.get("response_format", "json")
            body["response_format"] = ("verbose_json" if requested_format in ("srt", "vtt", "verbose_json")
                                       else "diarized_json" if requested_format == "diarized_json" else "json")
        else:
            requested_format = None
        stream_flag = body.pop("stream", False)
        streamed = stream_flag is True or stream_flag == "true"
        record.stream = streamed
        if streamed and channel.startswith("images/"):
            body.pop("partial_images", None)
        if body.get("stream_format") == "sse":
            raise Denied(400, "Use binary speech output; it is buffered for inspection", "unsupported_stream_format")
        # Auxiliary channels do not run chat optimisation transforms. Reuse atomic budget
        # admission, with only token-metered endpoints allowed under token/spend caps.
        media_price, media_charge = None, None
        if channel not in ("embeddings", "responses"):
            media_price, media_charge = reserve_media(gw, key, channel, body, record)
        if channel in ("embeddings", "responses"):
            reserve = dict(body)
            if channel == "embeddings":
                reserve["max_tokens"] = 1
            else:
                reserve["max_tokens"] = body.get("max_output_tokens") or 1000
                if key.limits.tokens_per_day or key.limits.spend_usd_per_day:
                    body["max_output_tokens"] = reserve["max_tokens"]
            gw._reserve_call(key, record, reserve, provider)
        url = provider.base_url.rstrip("/") + "/" + channel
        if deployment:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", deployment):
                raise Denied(400, "Invalid deployment", "invalid_deployment")
            version = request.query_params.get("api-version") or provider.api_version
            if not version or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:-preview)?", version):
                raise Denied(400, "A valid api-version is required", "invalid_api_version")
            url = provider.base_url.rstrip("/") + f"/openai/deployments/{deployment}/{channel}?api-version={version}"
        await gw._permit_egress(provider, policy, len(json.dumps(body)) + sum(len(item[1][1]) for item in files), scope, record)
        headers = gw.upstream.headers(provider, request.headers, credential if provider.uses_client_key else provider.credential())
        client = await gw.upstream.client(provider)
        record._upstream_started = True
        if files:
            headers.pop("content-type", None)
            response = await bounded_response(client, url, headers, data=body, files=files)
        else:
            response = await bounded_response(client, url, headers, body=body)
        if response.status_code != 200:
            # Provider diagnostics may themselves contain echoed inputs or secrets.
            raise Denied(response.status_code if 400 <= response.status_code < 600 else 502,
                         "Provider refused the request", "upstream_error", outcome="error")
        record.cost_usd = media_charge
        if channel == "audio/speech":
            mime = response.headers.get("content-type", "audio/mpeg").split(";")[0]
            data = response.content
            if body.get("response_format") == "pcm":
                from znyx_runtime.gateway.realtime import pcm_wav
                data, mime = pcm_wav(data), "audio/wav"
            elif not mime.startswith("audio/"):
                mime = "audio/" + str(body.get("response_format") or "mpeg")
            await media.check(gw, data, mime, scope, policy, record, "output")
            output = Response(response.content, media_type="audio/pcm" if body.get("response_format") == "pcm" else mime)
        else:
            try:
                payload = formats.parse_response(response.content)
            except ValueError:
                raise Denied(502, "Malformed provider output", "invalid_provider_response", outcome="error") from None
            if not isinstance(payload, dict):
                raise Denied(502, "Provider output is not an object", "invalid_provider_response", outcome="error")
            usage = payload.get("usage") or {}
            gw._apply_usage(record, formats.Usage(usage.get("input_tokens", usage.get("prompt_tokens")), usage.get("output_tokens", 0)))
            if channel not in ("embeddings", "responses"):
                record.cost_usd = media_cost(channel, media_price, body, record,
                    len(payload.get("data") or []) if channel.startswith("images/") else None)
                record.price_known = record.cost_usd is not None
            if channel == "responses":
                payload = await response_output(gw, payload, scope, policy, record, output_context(gw, body, record))
            elif channel.startswith("images/"):
                for image in payload.get("data") or []:
                    if image.get("b64_json"):
                        data, _ = media.decode(image["b64_json"])
                        mime = "image/png"
                    elif image.get("url"):
                        data, mime = await media.fetch(gw, image["url"], scope, policy, record)
                        image["url"] = "data:" + mime + ";base64," + base64.b64encode(data).decode()
                    else:
                        raise Denied(502, "Provider returned an uninspectable image URL", "unresolved_media")
                    await media.check(gw, data, mime, scope, policy, record, "output")
                    if image.get("b64_json") and image.get("url"):
                        image.pop("url", None)
                    if isinstance(image.get("revised_prompt"), str):
                        image["revised_prompt"] = await checked_text(gw, image["revised_prompt"], "output", scope, policy, record)
            elif channel.startswith("audio/"):
                original = str(payload.get("text") or "")
                payload["text"] = await checked_text(gw, original, "output", scope, policy, record)
                if payload["text"] != original:
                    payload.pop("segments", None)
                    payload.pop("words", None)
                # Segments/word timings carry duplicate transcripts: withhold if a rewrite
                # cannot be mapped exactly, rather than leaking the original alongside it.
                for part in payload.get("segments") or []:
                    if isinstance(part.get("text"), str):
                        part["text"] = await checked_text(gw, part["text"], "output", scope, policy, record)
                for part in payload.get("words") or []:
                    if isinstance(part.get("word"), str):
                        part["word"] = await checked_text(gw, part["word"], "output", scope, policy, record)
            if channel == "responses" and streamed:
                output = Response(b"".join(response_events(payload)), media_type="text/event-stream")
            elif channel.startswith("images/") and streamed:
                kind = "image_generation.completed" if channel == "images/generations" else "image_edit.completed"
                events = []
                for image in payload.get("data") or []:
                    b64 = image.get("b64_json")
                    if not b64:
                        data, _ = media.decode(image["url"])
                        b64 = base64.b64encode(data).decode()
                    event = {"type": kind, "b64_json": b64, "created_at": payload.get("created", 0),
                        "output_format": payload.get("output_format", body.get("output_format", "png")),
                        "size": payload.get("size", body.get("size", "auto")),
                        "quality": payload.get("quality", body.get("quality", "auto")),
                        "background": payload.get("background", body.get("background", "auto")),
                        "usage": payload.get("usage") or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
                    events.append(formats.sse(event, kind))
                output = Response(b"".join(events), media_type="text/event-stream")
            elif channel.startswith("audio/") and requested_format == "text":
                output = Response(payload["text"], media_type="text/plain")
            elif channel.startswith("audio/") and requested_format in ("srt", "vtt"):
                output = Response(subtitles(payload, requested_format), media_type="text/vtt" if requested_format == "vtt" else "application/x-subrip")
            elif channel.startswith("audio/") and streamed:
                output = Response(formats.sse({"type": "transcript.text.delta", "delta": payload["text"]}, "transcript.text.delta")
                    + formats.sse({"type": "transcript.text.done", "text": payload["text"]}, "transcript.text.done"), media_type="text/event-stream")
            else:
                output = JSONResponse(payload)
        record.status = 200
        gw._charge(key, record)
        gw._record(record)
        output.headers.update(gw.trace_headers(record))
        return output
    except Denied as exc:
        return gw._finish(record, formats.OpenAIChat, exc)
    except (httpx.TimeoutException, TimeoutError):
        return gw._finish(record, formats.OpenAIChat, Denied(504, "Channel provider timed out", "upstream_timeout", outcome="error"))
    except (httpx.HTTPError, UpstreamError):
        return gw._finish(record, formats.OpenAIChat, Denied(502, "Channel provider is unavailable", "upstream_error", outcome="error"))
    except Exception:
        return gw._finish(record, formats.OpenAIChat, Denied(500, "Channel request could not be handled", "internal_error", outcome="error"))
    finally:
        if key and getattr(record, "_reservation", None) is not None:
            if getattr(record, "_upstream_started", False):
                gw.estimate_missing_usage(record, 0)
            gw._charge(key, record)
        if acquired:
            gw._slots.release()


def _endpoint(channel):
    async def endpoint(request: Request):
        return await handle(request, channel)
    return endpoint


def _azure_endpoint(channel):
    async def endpoint(deployment: str, request: Request):
        return await handle(request, channel, deployment)
    return endpoint


for _channel in CHANNELS:
    router.add_api_route("/v1/" + _channel, _endpoint(_channel), methods=["POST"])
    router.add_api_route("/openai/deployments/{deployment}/" + _channel, _azure_endpoint(_channel), methods=["POST"])
