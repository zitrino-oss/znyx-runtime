"""Media bytes inspected by the customer's offline inference service, never client OCR claims."""
from __future__ import annotations

import base64
import os
from urllib.parse import urlsplit

import httpx

from znyx_runtime.gateway.config import Provider
from znyx_runtime.gateway.pipeline import Denied, quiet_policy

MAX_BYTES = 8 * 1024 * 1024


def decode(value: str) -> tuple[bytes, str]:
    mime = "application/octet-stream"
    if not isinstance(value, str):
        raise Denied(400, "Media must be base64 or a data URL", "invalid_media")
    if value.startswith("data:"):
        head, sep, value = value.partition(",")
        if not sep or not head.endswith(";base64"):
            raise Denied(400, "Use a base64 data URL", "invalid_media")
        mime = head[5:-7]
    if len(value) > 4 * ((MAX_BYTES + 2) // 3):
        raise Denied(413, "Media exceeds 8 MiB", "media_too_large")
    try:
        result = base64.b64decode(value, validate=True)
    except ValueError:
        raise Denied(400, "Invalid media base64", "invalid_media") from None
    if not result or len(result) > MAX_BYTES:
        raise Denied(400, "Empty or oversized media", "invalid_media")
    return result, mime


async def extract(gw, data: bytes, mime: str, scope, policy, record) -> str:
    url = (os.getenv("ZNYX_MEDIA_EXTRACT_URL") or os.getenv("ZNYX_INFERENCE_URL") or "").rstrip("/")
    if not url:
        raise Denied(503, "Configure the in-network media extractor to inspect this content", "media_extractor_unavailable")
    from znyx_core.engine.egress import in_network
    if not in_network(urlsplit(url).hostname):
        raise Denied(503, "The media extractor must be an operator-approved in-network endpoint", "media_extractor_not_in_network")
    if not data or len(data) > MAX_BYTES:
        raise Denied(413, "Media exceeds the inspection limit", "media_too_large")
    provider = Provider(name="media-extractor", format="openai", base_url=url,
                        in_boundary=True, timeout_seconds=160)
    await gw._permit_egress(provider, policy, len(data), scope, record)
    try:
        async with httpx.AsyncClient(timeout=160, follow_redirects=False, trust_env=False) as client:
            async with client.stream("POST", url + "/v1/media/extract", json={
                    "data": base64.b64encode(data).decode(), "mime_type": mime}) as response:
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > 6 * 1024 * 1024:
                        raise ValueError("oversized extraction response")
                    raw.extend(chunk)
                import json
                payload = json.loads(raw)
        if response.status_code != 200 or payload.get("complete") is not True or not isinstance(payload.get("text"), str):
            raise ValueError("incomplete extraction")
        if len(payload["text"]) > 1024 * 1024:
            raise ValueError("oversized extraction")
        record._last_media_duration = payload.get("duration_seconds")
        record.note("media_inspected")
        return payload["text"]
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        raise Denied(503, "Media could not be completely inspected", "media_inspection_failed") from None


async def check(gw, data, mime, scope, policy, record, stage="input"):
    text = await extract(gw, data, mime, scope, policy, record)
    duration = getattr(record, "_last_media_duration", None)
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and 0 <= duration <= 120:
        field = "_media_" + stage + "_seconds"
        setattr(record, field, getattr(record, field, 0) + duration)
        record.note("audio_" + stage + "_ms", round(duration * 1000))
    verdict = await gw._check_text(text, "text", "user" if stage == "input" else "assistant", None,
        stage, scope, record, policy, quiet_policy(policy), "media", {"attachments": [
            {"type": mime.split("/", 1)[0], "extracted_text": text}], "modalities": [mime.split("/", 1)[0]]})
    record.rule_ids.extend(verdict.rule_ids)
    if stage == "input":
        record.input_decision = verdict.decision
    else:
        record.output_decision = verdict.decision
    if verdict.decision == "BLOCK" or verdict.wants_change or verdict.sanitized_text not in (None, text):
        # Changing the transcript does not change the bytes a vision/audio model sees.
        raise Denied(400, "Media was withheld by policy", "content_filter", outcome="blocked_" + stage)
    if stage == "input":
        sources = getattr(record, "_media_sources", [])
        if len(sources) < 32:
            sources.append(text[:200000])
        record._media_sources = sources
    return text


async def fetch(gw, url, scope, policy, record):
    """Only operator-approved HTTPS origins. No redirects, private addresses or client credentials."""
    import asyncio
    import ipaddress
    import socket
    parsed = urlsplit(url)
    allowed = {name.strip().lower() for name in os.getenv("ZNYX_MEDIA_URL_ALLOWLIST", "").split(",") if name.strip()}
    try:
        valid = (parsed.scheme == "https" and parsed.hostname in allowed and parsed.port in (None, 443)
                 and not parsed.username and not parsed.password and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise Denied(400, "Remote media origin is not approved; supply inline bytes or configure the operator allowlist", "unresolved_media")
    addresses = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, 443, 0, socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses):
        raise Denied(400, "Remote media must resolve exclusively to public addresses", "unsafe_media_url")
    destination = Provider(name="media-download", format="openai", base_url=url,
        region=os.getenv("ZNYX_MEDIA_URL_REGION") or None, in_boundary=False)
    await gw._permit_egress(destination, policy, 0, scope, record)
    # Pin the checked address for this connection; Host/SNI retain certificate and
    # virtual-host verification. A second DNS lookup must not rebind to metadata/private IPs.
    address = addresses[0][4][0]
    pinned = parsed._replace(netloc="[" + address + "]" if ":" in address else address).geturl()
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
            async with client.stream("GET", pinned, headers={"host": parsed.netloc},
                                     extensions={"sni_hostname": parsed.hostname}) as response:
                if response.status_code != 200:
                    raise ValueError("Media download failed")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(data) + len(chunk) > MAX_BYTES:
                        raise ValueError("Media exceeds the inspection limit")
                    data.extend(chunk)
                mime = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        return bytes(data), mime
    except (httpx.HTTPError, ValueError):
        raise Denied(502, "Remote media could not be retrieved completely", "media_download_failed") from None


def attachments(value):
    """Actual inline media from Chat, Responses and Anthropic. Never trust alt/transcript fields."""
    if isinstance(value, list):
        for part in value:
            yield from attachments(part)
    elif isinstance(value, dict):
        kind = value.get("type")
        if kind in ("image_url", "input_image"):
            image = value.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str) or not url.startswith("data:"):
                if not isinstance(url, str):
                    raise Denied(400, "An image URL or inline bytes are required", "unresolved_media")
                def inline_image(data, mime, block=value):
                    uri = "data:" + mime + ";base64," + base64.b64encode(data).decode()
                    if isinstance(block.get("image_url"), dict):
                        block["image_url"]["url"] = uri
                    else:
                        block["image_url"] = uri
                yield url, None, inline_image
            else:
                yield decode(url)
        elif kind in ("image", "document"):
            source = value.get("source") or {}
            if source.get("type") == "text" and isinstance(source.get("data"), str):
                yield source["data"].encode(), "text/plain"
            elif source.get("type") == "base64":
                data, _ = decode(source.get("data"))
                yield data, source.get("media_type") or "application/octet-stream"
            else:
                if source.get("type") == "url" and isinstance(source.get("url"), str):
                    def inline_source(data, mime, block=value):
                        block["source"] = {"type": "base64", "media_type": mime,
                                           "data": base64.b64encode(data).decode()}
                    yield source["url"], None, inline_source
                else:
                    raise Denied(400, "Supply inline image/document bytes or an approved URL", "unresolved_media")
        elif kind == "input_audio":
            audio = value.get("input_audio") or value
            data, _ = decode(audio.get("data") or audio.get("audio"))
            yield data, "audio/" + str(audio.get("format") or "wav")
        elif kind in ("file", "input_file"):
            file = value.get("file") or value
            if not file.get("file_data"):
                raise Denied(400, "Stored file references cannot be inspected; supply file_data", "unresolved_media")
            data, mime = decode(file["file_data"])
            yield data, "application/pdf" if mime == "application/octet-stream" else mime
        else:
            for name, nested in value.items():
                if name not in ("metadata", "tools", "parameters", "input_schema"):
                    yield from attachments(nested)


async def inspect_body(gw, body, scope, policy, record, stage="input"):
    count = 0
    for attachment in attachments(body):
        data, mime = attachment[:2]
        count += 1
        if count > 32:
            raise Denied(413, "At most 32 attachments may be inspected", "too_many_attachments")
        if mime is None:
            data, mime = await fetch(gw, data, scope, policy, record)
            attachment[2](data, mime)  # the provider sees the checked bytes, never a mutable URL
        await check(gw, data, mime, scope, policy, record, stage)
    return count
