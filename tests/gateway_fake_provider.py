"""A fake LLM provider for gateway tests: OpenAI, Azure OpenAI and Anthropic shapes,
streaming and not, recording exactly what the gateway forwarded.

What it answers is chosen by the last user message:
    SAY:<text>                     answer with <text>
    SAYREV:<text>                  answer with <text> reversed
    TOOL:<name>:<json args>        call a tool
    FAIL:<status>                  answer with the provider's error and that status
    ECHO (or anything else)        answer "ok"; the forwarded body is in `received`

A test can also set ``scripted`` to a function of the forwarded body that returns the
exact answer: a JSON object, ``(status, object)``, raw bytes or text, or a list of raw
SSE pieces to stream. ``reset()`` clears it.
"""
import asyncio
import json
import time
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

received: List[Dict[str, Any]] = []
scripted = None
STREAM_PIECE = 7          # characters per streamed delta
STREAM_DELAY = 0.0
RESPONSE_DELAY = 0.0      # seconds before a non-streamed answer (calls in flight together)


def _last_user_text(body: Dict[str, Any]) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        for part in content or []:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
    return ""


def _script(body: Dict[str, Any]):
    text = _last_user_text(body)
    if text.startswith("SAYREV:"):           # the answer, reversed in the prompt so a secret
        return "text", text[7:][::-1]         # in the answer is not also in the prompt
    if text.startswith("SAY:"):
        return "text", text[4:]
    if text.startswith("TOOL:"):
        _, name, args = text.split(":", 2)
        return "tool", (name, args)
    if text.startswith("FAIL:"):
        return "fail", int(text[5:])
    return "text", "ok"


def _scripted(body: Dict[str, Any]):
    out = scripted(body)
    if isinstance(out, list):
        async def gen():
            for piece in out:
                yield piece if isinstance(piece, bytes) else piece.encode("utf-8")
        return StreamingResponse(gen(), media_type="text/event-stream")
    status, payload = out if isinstance(out, tuple) else (200, out)
    if isinstance(payload, (bytes, str)):
        return Response(payload, status_code=status, media_type="application/json")
    return JSONResponse(payload, status_code=status)


def _pieces(text: str):
    for i in range(0, len(text), STREAM_PIECE):
        yield text[i:i + STREAM_PIECE]


def build_app() -> FastAPI:
    app = FastAPI()

    async def openai(request: Request, deployment: str = None):
        body = await request.json()
        received.append({"api": "azure" if deployment else "openai", "body": body,
                         "headers": dict(request.headers), "deployment": deployment,
                         "query": dict(request.query_params)})
        if RESPONSE_DELAY and not body.get("stream"):
            await asyncio.sleep(RESPONSE_DELAY)
        if scripted is not None:
            return _scripted(body)
        kind, value = _script(body)
        model = body.get("model") or deployment or "gpt-4o-mini"
        if kind == "fail":
            return JSONResponse({"error": {"message": "provider says no", "type": "rate_limit_error",
                                           "code": "rate_limit_exceeded"}}, status_code=value,
                                headers={"retry-after": "7", "x-request-id": "req_fake"})
        usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                 "prompt_tokens_details": {"cached_tokens": 20}}
        if not body.get("stream"):
            message: Dict[str, Any] = {"role": "assistant", "content": None}
            finish = "stop"
            if kind == "tool":
                message["tool_calls"] = [{"id": "call_1", "type": "function",
                                          "function": {"name": value[0], "arguments": value[1]}}]
                finish = "tool_calls"
            else:
                message["content"] = value
            return JSONResponse({"id": "chatcmpl-fake", "object": "chat.completion", "created": 1,
                                 "model": model, "system_fingerprint": "fp_fake", "extra_field": 42,
                                 "choices": [{"index": 0, "message": message, "logprobs": None,
                                              "finish_reason": finish}], "usage": usage},
                                headers={"x-request-id": "req_fake", "x-ratelimit-remaining-requests": "99"})

        async def gen():
            env = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 1, "model": model}
            yield f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            if kind == "tool":
                name, args = value
                yield f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': ''}}]}, 'finish_reason': None}]})}\n\n"
                for piece in _pieces(args):
                    yield f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': piece}}]}, 'finish_reason': None}]})}\n\n"
                finish = "tool_calls"
            else:
                for piece in _pieces(value):
                    if STREAM_DELAY:
                        await asyncio.sleep(STREAM_DELAY)
                    yield f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"
                finish = "stop"
            yield f"data: {json.dumps({**env, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})}\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                yield f"data: {json.dumps({**env, 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await openai(request)

    @app.post("/openai/deployments/{deployment}/chat/completions")
    async def azure(deployment: str, request: Request):
        return await openai(request, deployment)

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        received.append({"api": "anthropic", "body": body, "headers": dict(request.headers)})
        if RESPONSE_DELAY and not body.get("stream"):
            await asyncio.sleep(RESPONSE_DELAY)
        if scripted is not None:
            return _scripted(body)
        kind, value = _script(body)
        model = body.get("model") or "claude-3-5-haiku-20241022"
        if kind == "fail":
            return JSONResponse({"type": "error", "error": {"type": "rate_limit_error",
                                                            "message": "provider says no"}},
                                status_code=value, headers={"retry-after": "7"})
        usage = {"input_tokens": 100, "output_tokens": 25, "cache_read_input_tokens": 10,
                 "cache_creation_input_tokens": 0}
        if not body.get("stream"):
            if kind == "tool":
                content = [{"type": "tool_use", "id": "toolu_1", "name": value[0],
                            "input": json.loads(value[1] or "{}")}]
                stop = "tool_use"
            else:
                content = [{"type": "text", "text": value}]
                stop = "end_turn"
            return JSONResponse({"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
                                 "content": content, "stop_reason": stop, "stop_sequence": None,
                                 "usage": usage})

        def ev(name, data):
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"

        async def gen():
            yield ev("message_start", {"type": "message_start", "message": {
                "id": "msg_fake", "type": "message", "role": "assistant", "model": model, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 1, "cache_read_input_tokens": 10}}})
            yield ev("ping", {"type": "ping"})
            if kind == "tool":
                name, args = value
                yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                                 "content_block": {"type": "tool_use", "id": "toolu_1",
                                                                   "name": name, "input": {}}})
                for piece in _pieces(args):
                    yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                     "delta": {"type": "input_json_delta", "partial_json": piece}})
                yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
                stop = "tool_use"
            else:
                yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                                 "content_block": {"type": "text", "text": ""}})
                for piece in _pieces(value):
                    if STREAM_DELAY:
                        await asyncio.sleep(STREAM_DELAY)
                    yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                     "delta": {"type": "text_delta", "text": piece}})
                yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
                stop = "end_turn"
            yield ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                       "usage": {"output_tokens": 25}})
            yield ev("message_stop", {"type": "message_stop"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def reset() -> None:
    global scripted, RESPONSE_DELAY
    received.clear()
    scripted = None
    RESPONSE_DELAY = 0.0
