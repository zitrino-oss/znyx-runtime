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
BATCH_DONE = False        # whether submitted batches have finished
batches: Dict[str, List[Dict[str, Any]]] = {}


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

    # -- batch APIs ------------------------------------------------------------------

    @app.post("/v1/messages/batches")
    async def anthropic_batch(request: Request):
        body = await request.json()
        batch_id = f"msgbatch_{len(batches) + 1}"
        batches[batch_id] = [{"custom_id": r["custom_id"], "body": r["params"], "api": "anthropic"}
                             for r in body["requests"]]
        received.append({"api": "anthropic-batch", "body": body, "headers": dict(request.headers)})
        return JSONResponse({"id": batch_id, "type": "message_batch", "processing_status": "in_progress"})

    @app.get("/v1/messages/batches/{batch_id}")
    async def anthropic_batch_status(batch_id: str):
        if not BATCH_DONE:
            return JSONResponse({"id": batch_id, "processing_status": "in_progress"})
        return JSONResponse({"id": batch_id, "processing_status": "ended",
                             "results_url": f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"})

    @app.get("/v1/messages/batches/{batch_id}/results")
    async def anthropic_batch_results(batch_id: str):
        lines = [json.dumps({"custom_id": item["custom_id"], "result": {
            "type": "succeeded", "message": _batch_answer("anthropic", item["body"])}}) for item in batches[batch_id]]
        return Response("\n".join(lines), media_type="application/jsonl")

    @app.post("/v1/files")
    async def openai_upload(request: Request):
        form = await request.form()
        content = (await form["file"].read()).decode()
        file_id = f"file-{len(batches) + 1}"
        batches[file_id] = [json.loads(line) for line in content.splitlines() if line.strip()]
        received.append({"api": "openai-file", "body": {"purpose": form.get("purpose"), "lines": batches[file_id]},
                         "headers": dict(request.headers)})
        return JSONResponse({"id": file_id, "object": "file", "purpose": "batch"})

    @app.post("/v1/batches")
    async def openai_batch(request: Request):
        body = await request.json()
        batch_id = f"batch_{len(batches) + 1}"
        batches[batch_id] = [{"custom_id": line["custom_id"], "body": line["body"], "api": "openai"}
                             for line in batches[body["input_file_id"]]]
        received.append({"api": "openai-batch", "body": body, "headers": dict(request.headers)})
        return JSONResponse({"id": batch_id, "object": "batch", "status": "validating"})

    @app.get("/v1/batches/{batch_id}")
    async def openai_batch_status(batch_id: str):
        if not BATCH_DONE:
            return JSONResponse({"id": batch_id, "status": "in_progress"})
        return JSONResponse({"id": batch_id, "status": "completed", "output_file_id": f"out-{batch_id}"})

    @app.get("/v1/files/{file_id}/content")
    async def openai_file_content(file_id: str):
        items = batches[file_id.removeprefix("out-")]
        lines = [json.dumps({"custom_id": item["custom_id"], "response": {
            "status_code": 200, "body": _batch_answer("openai", item["body"])}}) for item in items]
        return Response("\n".join(lines), media_type="application/jsonl")

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

    @app.api_route("/mcp", methods=["POST", "DELETE"])
    async def mcp_json(request: Request):
        return await _mcp(request, sse=False)

    @app.api_route("/mcp-sse", methods=["POST", "DELETE"])
    async def mcp_sse(request: Request):
        return await _mcp(request, sse=True)

    return app


# -- a fake MCP server (Streamable HTTP) -------------------------------------------------------

MCP_TOOLS: List[Dict[str, Any]] = []
mcp_received: List[Dict[str, Any]] = []
mcp_results: Dict[str, Any] = {}          # tool name -> the result to return (else an echo)
mcp_errors: Dict[str, Any] = {}           # tool name -> a JSON-RPC error object to answer with
mcp_extra: List[Dict[str, Any]] = []      # messages streamed (SSE) before the answer


def default_mcp_tools() -> List[Dict[str, Any]]:
    return [
        {"name": "search", "description": "Search the CRM", "inputSchema": {"type": "object"}},
        {"name": "delete_record", "description": "Delete a CRM record", "inputSchema": {"type": "object"},
         "annotations": {"destructiveHint": True}},
        {"name": "send_email", "description": "Send an email", "inputSchema": {"type": "object"}},
    ]


async def _mcp(request: Request, sse: bool):
    if request.method == "DELETE":
        mcp_received.append({"method": "DELETE", "headers": dict(request.headers)})
        return Response(status_code=200)
    body = await request.json()
    mcp_received.append({"body": body, "headers": dict(request.headers)})
    method, msg_id = body.get("method"), body.get("id")
    if msg_id is None:
        return Response(status_code=202)
    headers = {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake-crm", "version": "1"}}
        headers["mcp-session-id"] = "upstream-session-1"
    elif method == "tools/list":
        result = {"tools": [dict(t) for t in MCP_TOOLS]}
    elif method == "tools/call":
        name = body["params"]["name"]
        if name in mcp_errors:
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": mcp_errors[name]})
        result = mcp_results.get(name) or {
            "content": [{"type": "text", "text": f"{name} ran with {json.dumps(body['params'].get('arguments'))}"}]}
    elif method == "resources/read":
        result = {"contents": [{"uri": body["params"]["uri"], "text": mcp_results.get("resource", "a document")}]}
    elif method == "prompts/get":
        content = mcp_results.get("prompt", "hi")
        content = content if isinstance(content, dict) else {"type": "text", "text": content}
        result = {"messages": [{"role": "user", "content": content}]}
    elif method == "tasks/result":
        result = mcp_results.get("task") or {"content": [{"type": "text", "text": "task done"}]}
    elif method in ("ping", "prompts/list", "completion/complete"):
        result = {}
    else:
        answer = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "unknown method"}}
        return JSONResponse(answer)
    answer = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    if not sse:
        return JSONResponse(answer, headers=headers)
    progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1, "total": 2}}
    stream = "".join(f"event: message\ndata: {json.dumps(m)}\n\n" for m in [progress, *mcp_extra, answer])
    return Response(stream, media_type="text/event-stream", headers=headers)


def reset() -> None:
    global scripted, RESPONSE_DELAY, BATCH_DONE
    received.clear()
    batches.clear()
    scripted = None
    RESPONSE_DELAY = 0.0
    BATCH_DONE = False
    mcp_received.clear()
    mcp_results.clear()
    mcp_errors.clear()
    mcp_extra.clear()
    MCP_TOOLS[:] = default_mcp_tools()


def _batch_answer(api: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """What a batch returns for one request: the scripted answer, else the SAY text."""
    if scripted is not None:
        out = scripted(body)
        if isinstance(out, dict):
            return out
    kind, value = _script(body)
    text = value if kind == "text" else "ok"
    if api == "anthropic":
        return {"id": "msg_b", "type": "message", "role": "assistant", "model": body.get("model"),
                "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 25}}
    return {"id": "chatcmpl-b", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}}
