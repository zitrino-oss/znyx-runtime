"""Batch submission (technique 8): a call the app marks as not urgent
(``X-Znyx-Urgency: low``) goes through the provider's discounted batch API.

The call is identified, permitted and guarded on the way in like any other, then kept in
a local SQLite file (shared by the worker processes) and answered ``202`` with a job id.
The leader worker submits waiting calls (Anthropic Message Batches; OpenAI Files and
Batches), polls the provider, runs the output guard on each result under the policy in
force when it arrives, charges the batch price and records the call. The app fetches the
checked answer with ``GET /v1/znyx/batches/{id}`` using the same gateway key.

Provider credentials are never stored: they are resolved when the batch is submitted.
Requests are deleted once answered; answers are kept for ``ZNYX_BATCH_RETENTION_DAYS``
(default 7) and then deleted.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

BATCH_DISCOUNT = 0.5                 # both providers bill batch calls at half the standard price
DEFAULT_RETENTION_DAYS = 7
POLL_SECONDS = 30
MAX_PER_BATCH = 1000


def default_path() -> Path:
    raw = os.environ.get("ZNYX_BATCH_DB")
    if raw:
        return Path(raw)
    from znyx_runtime.install_state import STATE_DIR
    return Path(STATE_DIR) / "batches.sqlite"


def supports_batch(provider) -> bool:
    """Providers whose batch API the gateway speaks."""
    if provider.uses_client_key:
        return False                 # the caller's own provider key would have to be stored
    host = (urlsplit(provider.base_url).hostname or "").lower()
    return (provider.format == "anthropic" and host == "api.anthropic.com") or \
        (provider.format == "openai" and host == "api.openai.com")


class BatchStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or default_path())
        # The file holds requests and answers: its folder, and the database with its journal
        # files, are readable by this user only.
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self._lock = threading.Lock()
        old = os.umask(0o077)
        try:
            with self._connect() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, created REAL, updated REAL, key_sha256 TEXT, member TEXT, api TEXT,
                    provider TEXT, url TEXT, body TEXT, status TEXT, provider_batch TEXT, result TEXT,
                    error TEXT, record TEXT)""")
                db.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)")
        finally:
            os.umask(old)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = sqlite3.Row
        return db

    def add(self, *, key_sha256: str, member: str, api: str, provider: str, url: str, body: Dict[str, Any],
            record: Dict[str, Any]) -> str:
        job_id = "zbatch_" + uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (job_id, now, now, key_sha256, member, api, provider, url, json.dumps(body), "queued",
                        None, None, None, json.dumps(record, default=str)))
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def by_status(self, status: str, limit: int = MAX_PER_BATCH) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status = ? ORDER BY created LIMIT ?", (status, limit)).fetchall()
        return [dict(r) for r in rows]

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        names = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._connect() as db:
            db.execute(f"UPDATE jobs SET {names}, updated = ? WHERE id = ?", (*fields.values(), time.time(), job_id))

    def purge(self, retention_days: Optional[float] = None) -> int:
        days = retention_days if retention_days is not None else float(
            os.environ.get("ZNYX_BATCH_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
        cutoff = time.time() - days * 86400
        with self._lock, self._connect() as db:
            return db.execute("DELETE FROM jobs WHERE created < ?", (cutoff,)).rowcount


class BatchRunner:
    """The leader's loop: submit waiting calls, poll submitted batches, finish results."""

    def __init__(self, gateway, store: Optional[BatchStore] = None, interval: float = POLL_SECONDS):
        self.gateway = gateway
        self.store = store
        self.interval = interval
        self._task = None

    def start(self) -> None:
        import asyncio
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _loop(self) -> None:
        import asyncio
        while True:
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 - the loop must keep running
                logger.exception("gateway: batch loop failed")
            await asyncio.sleep(self.interval)

    async def tick(self) -> None:
        import asyncio
        if self.store is None:
            if not default_path().exists():
                return                         # nothing was ever queued on this install
            self.store = self.gateway.batch_store()
        await self._recover()
        await self._submit()
        await self._poll()
        await asyncio.to_thread(self.store.purge)

    async def _recover(self) -> None:
        """A submission whose outcome is unknown (the process stopped, or the provider's
        answer never arrived) is not sent again, which could bill it twice: after ten minutes
        it is failed and recorded."""
        import asyncio
        for job in await asyncio.to_thread(self.store.by_status, "submitting"):
            if time.time() - float(job["updated"] or 0) > 600:
                await self.gateway.finish_batch_job(job, None, "the submission's outcome is unknown")

    # -- providers ----------------------------------------------------------------

    def _provider(self, job: Dict[str, Any]):
        return self.gateway.config.providers.get(job["provider"])

    async def _client_headers(self, provider) -> Dict[str, str]:
        credential = provider.credential()
        return self.gateway.upstream.headers(provider, {}, credential)

    async def _submit(self) -> None:
        import asyncio
        import httpx
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for job in await asyncio.to_thread(self.store.by_status, "queued"):
            # The key, its budgets and the egress gate are checked again now, at the crossing.
            refused = await self.gateway.batch_job_allowed(job)
            if refused:
                await self.gateway.finish_batch_job(job, None, refused)
                continue
            groups.setdefault(job["provider"], []).append(job)
        for name, jobs in groups.items():
            provider = self._provider(jobs[0])
            if provider is None or not supports_batch(provider):
                for job in jobs:
                    await self.gateway.finish_batch_job(job, None, "the provider is no longer available")
                continue
            for job in jobs:
                await asyncio.to_thread(self.store.update, job["id"], status="submitting")
            try:
                batch_id = await (self._submit_anthropic(provider, jobs) if provider.format == "anthropic"
                                  else self._submit_openai(provider, jobs))
            except httpx.HTTPStatusError as exc:
                # The provider refused it: nothing was accepted, so it is tried again later.
                logger.warning("gateway: batch submission to %s refused: %s", name, exc)
                for job in jobs:
                    await asyncio.to_thread(self.store.update, job["id"], status="queued")
                continue
            except Exception as exc:  # noqa: BLE001 - outcome unknown: left for _recover, never resent
                logger.warning("gateway: batch submission to %s failed: %s", name, exc)
                continue
            for job in jobs:
                await asyncio.to_thread(self.store.update, job["id"], status="submitted", provider_batch=batch_id)

    async def _submit_anthropic(self, provider, jobs) -> str:
        client = await self.gateway.upstream.client(provider)
        headers = await self._client_headers(provider)
        requests = [{"custom_id": job["id"], "params": json.loads(job["body"])} for job in jobs]
        response = await client.post(f"{provider.base_url}/v1/messages/batches", headers=headers,
                                     json={"requests": requests})
        response.raise_for_status()
        return str(response.json()["id"])

    async def _submit_openai(self, provider, jobs) -> str:
        client = await self.gateway.upstream.client(provider)
        headers = await self._client_headers(provider)
        lines = "\n".join(json.dumps({"custom_id": job["id"], "method": "POST", "url": "/v1/chat/completions",
                                      "body": json.loads(job["body"])}) for job in jobs)
        upload_headers = {k: v for k, v in headers.items() if k != "content-type"}
        uploaded = await client.post(f"{provider.base_url}/files", headers=upload_headers,
                                     data={"purpose": "batch"},
                                     files={"file": ("batch.jsonl", lines.encode(), "application/jsonl")})
        uploaded.raise_for_status()
        created = await client.post(f"{provider.base_url}/batches", headers=headers,
                                    json={"input_file_id": uploaded.json()["id"], "endpoint": "/v1/chat/completions",
                                          "completion_window": "24h"})
        created.raise_for_status()
        return str(created.json()["id"])

    async def _poll(self) -> None:
        import asyncio
        batches: Dict[str, List[Dict[str, Any]]] = {}
        for job in await asyncio.to_thread(self.store.by_status, "submitted"):
            batches.setdefault(job["provider_batch"], []).append(job)
        for batch_id, jobs in batches.items():
            provider = self._provider(jobs[0])
            if provider is None:
                # The provider left the configuration: its answers cannot be fetched.
                for job in jobs:
                    await self.gateway.finish_batch_job(job, None, "the provider is no longer available")
                continue
            try:
                results = await (self._results_anthropic(provider, batch_id) if provider.format == "anthropic"
                                 else self._results_openai(provider, batch_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("gateway: batch %s could not be polled: %s", batch_id, exc)
                continue
            if results is None:
                continue                                   # still running
            for job in jobs:
                payload, error = results.get(job["id"], (None, "no result for this request"))
                await self.gateway.finish_batch_job(job, payload, error)

    async def _results_anthropic(self, provider, batch_id: str):
        client = await self.gateway.upstream.client(provider)
        headers = await self._client_headers(provider)
        status = await client.get(f"{provider.base_url}/v1/messages/batches/{batch_id}", headers=headers)
        status.raise_for_status()
        info = status.json()
        if info.get("processing_status") != "ended":
            return None
        lines = await client.get(info["results_url"], headers=headers)
        lines.raise_for_status()
        out = {}
        for line in lines.text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            result = item.get("result") or {}
            if result.get("type") == "succeeded":
                out[item.get("custom_id")] = (result.get("message"), None)
            else:
                out[item.get("custom_id")] = (None, f"the provider reported {result.get('type') or 'an error'}")
        return out

    async def _results_openai(self, provider, batch_id: str):
        client = await self.gateway.upstream.client(provider)
        headers = await self._client_headers(provider)
        status = await client.get(f"{provider.base_url}/batches/{batch_id}", headers=headers)
        status.raise_for_status()
        info = status.json()
        if info.get("status") in ("failed", "expired", "cancelled"):
            return {}
        if info.get("status") != "completed":
            return None
        out = {}
        for file_field in ("output_file_id", "error_file_id"):
            file_id = info.get(file_field)
            if not file_id:
                continue
            content = await client.get(f"{provider.base_url}/files/{file_id}/content", headers=headers)
            content.raise_for_status()
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                response = item.get("response") or {}
                if response.get("status_code") == 200 and isinstance(response.get("body"), dict):
                    out[item.get("custom_id")] = (response["body"], None)
                else:
                    out[item.get("custom_id")] = (None, "the provider reported an error")
        return out
