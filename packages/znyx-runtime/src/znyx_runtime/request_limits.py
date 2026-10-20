"""Request body limits for the runtime's own routes.

FastAPI reads and parses a route's JSON body before the route's authentication
dependency runs, so without a limit anyone who can reach the port could make the
runtime hold any amount of memory, or keep connections open by sending a body very
slowly. Bodies are capped (``ZNYX_MAX_REQUEST_BYTES``, default 16 MB) and must
arrive within ``ZNYX_BODY_TIMEOUT_SECONDS`` (default 30). The gateway's routes check
the caller's key before reading anything and apply their own limits, so they are
left to do that.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_BODY_TIMEOUT_SECONDS = 30.0
GATEWAY_PATHS = ("/v1/chat/completions", "/v1/messages", "/openai/deployments/")


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: Optional[int] = None, timeout: Optional[float] = None):
        self.app = app
        self.max_bytes = max_bytes or int(os.getenv("ZNYX_MAX_REQUEST_BYTES") or DEFAULT_MAX_REQUEST_BYTES)
        self.timeout = timeout or float(os.getenv("ZNYX_BODY_TIMEOUT_SECONDS") or DEFAULT_BODY_TIMEOUT_SECONDS)

    async def __call__(self, scope, receive, send) -> None:
        if (scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH")
                or str(scope.get("path", "")).startswith(GATEWAY_PATHS)):
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reply(send, 413, f"Request body larger than {self.max_bytes} bytes")
            return
        chunks = []

        async def read():
            size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return message
                body = message.get("body", b"")
                size += len(body)
                if size > self.max_bytes:
                    return None
                chunks.append(body)
                if not message.get("more_body", False):
                    return message

        try:
            last = await asyncio.wait_for(read(), timeout=self.timeout)
        except asyncio.TimeoutError:
            await self._reply(send, 408, "The request body did not arrive in time")
            return
        if last is None:
            await self._reply(send, 413, f"Request body larger than {self.max_bytes} bytes")
            return
        if last["type"] == "http.disconnect":
            return
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _reply(send, status: int, message: str) -> None:
        body = json.dumps({"detail": message}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
