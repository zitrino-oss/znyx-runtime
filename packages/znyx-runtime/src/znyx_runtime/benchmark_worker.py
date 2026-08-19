"""Runs control-plane-queued benchmarks against this runtime's own inference sidecar.

WHY THE RUNTIME AND NOT THE CONTROL PLANE
-----------------------------------------
Benchmarking a model-backed detector means putting every sample in front of the model, so
whoever evaluates has to reach an inference sidecar. The control plane cannot:

  * it must never dial a tenant-supplied URL (that would turn an org-admin text field into
    an SSRF primitive against its own network), and
  * customer sidecars sit on private networks with no inbound path.

Evaluating there anyway does not fail loudly — escalation fails open to the deterministic
base, the run reports "completed", and metrics describing the REGEX layer get published
under ``model_id@revision`` as though the model produced them.

This runtime already owns the sidecar relationship, so the work comes here. The control
plane queues a run; this worker claims it, evaluates it locally, and posts results back.
Same direction as every other channel — the runtime pulls, the control plane never dials in.

SHAPE
-----
Claim -> page samples -> evaluate page -> post page -> repeat -> complete. Streaming rather
than buffering: a large suite is several MB of eval text, and a crash mid-run then costs one
page instead of the whole run. Each posted page also renews the lease, so a long run is not
reaped while it is making progress.

Deliberately NOT a bundle-cycle listener. ``bundle_manager._spawn_cycle`` skips a tick while
the previous one is still in flight, so a multi-minute benchmark parked there would starve
model-pin pushes and heartbeats — and the control plane marks a deployment stale after 300s.
It gets its own task.

Best-effort throughout, matching ``runtime_report``: an unreachable or older control plane,
or a malformed response, must never affect policy enforcement. Silent when no control-plane
URL is configured (the ordinary OSS/YAML case).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30.0
# Page size for both fetching samples and posting results. Must stay <= the control plane's
# bound (MAX_PAGE_SIZE / MAX_RESULTS_BATCH = 500) or the runtime would 422 itself into
# silence — the same pairing as _MAX_REPORTED_MODELS <-> max_length on the heartbeat.
_PAGE_SIZE = 200

# Idle backoff. Benchmarks are user-initiated — someone clicked Run and is watching a
# spinner — so the floor is well under the bundle poll's flat 30s. The ceiling keeps an
# idle runtime from generating pointless traffic forever.
_POLL_MIN_SECONDS = 5.0
_POLL_MAX_SECONDS = 60.0


class BenchmarkWorker:
    """Claims queued benchmark runs and evaluates them against the local sidecar."""

    def __init__(self, control_plane_url: str, runtime_token: str, evaluator,
                 poll_min_seconds: float = _POLL_MIN_SECONDS,
                 poll_max_seconds: float = _POLL_MAX_SECONDS,
                 page_size: int = _PAGE_SIZE):
        self.control_plane_url = (control_plane_url or "").rstrip("/")
        self.runtime_token = runtime_token or ""
        self.evaluator = evaluator
        self.poll_min_seconds = poll_min_seconds
        self.poll_max_seconds = poll_max_seconds
        self.page_size = max(1, min(page_size, 500))
        self._unsupported = False   # set once a control plane answers 404
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    @property
    def enabled(self) -> bool:
        # Needs somewhere to claim from and something to authenticate with. Both absent is
        # the ordinary self-hosted case, not a misconfiguration.
        return bool(self.control_plane_url and self.runtime_token)

    # ── lifecycle ────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.enabled:
            logger.debug("benchmark-worker: not configured; idle")
            return
        self._stopping = False
        self._task = asyncio.create_task(self._loop())
        logger.info("benchmark-worker: polling %s for queued runs", self.control_plane_url)

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        delay = self.poll_min_seconds
        while not self._stopping:
            try:
                await asyncio.sleep(delay)
                if self._unsupported:
                    return
                did_work = await self.poll_once()
                # Reset on work, back off geometrically when idle.
                delay = (self.poll_min_seconds if did_work
                         else min(delay * 2, self.poll_max_seconds))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never let the loop die
                logger.debug("benchmark-worker: cycle failed (will retry): %s", exc)
                delay = min(delay * 2, self.poll_max_seconds)

    # ── one cycle ────────────────────────────────────────────────────────────────────

    async def poll_once(self) -> bool:
        """Claim and run at most one benchmark. Returns True when work was done."""
        if not self.enabled or self._unsupported:
            return False

        claimed = await self._claim()
        if not claimed:
            return False

        run_id = claimed.get("run_id")
        policy = claimed.get("policy") or {}
        total = int(claimed.get("total_samples") or 0)
        logger.info("benchmark-worker: claimed run %s (%s samples)", run_id, total)

        try:
            evaluated = await self._run(run_id, policy)
            await self._complete(run_id)
            logger.info("benchmark-worker: run %s finished (%s samples evaluated)",
                        run_id, evaluated)
        except Exception as exc:  # noqa: BLE001
            # Tell the control plane why rather than letting the lease silently lapse —
            # the console can then show a reason instead of a run that just stops.
            logger.warning("benchmark-worker: run %s failed: %s", run_id, exc)
            await self._complete(run_id, error=str(exc)[:2000])
        return True

    async def _run(self, run_id: str, policy: Dict[str, Any]) -> int:
        """Stream through the run's samples: fetch a page, evaluate it, post it, repeat."""
        offset = 0
        evaluated = 0
        while True:
            page = await self._fetch_samples(run_id, offset)
            if page is None:
                raise RuntimeError("could not fetch samples")
            items = page.get("items") or []
            page_total = int(page.get("total") or 0)
            if not items:
                # A page may be empty because every sample on it was a lifecycle-hook
                # stage the control plane filtered out; only stop once the offset is past
                # the end of the dataset.
                if offset >= page_total:
                    break
                offset += self.page_size
                continue

            results = [await self._evaluate(item, policy) for item in items]
            await self._post_results(run_id, results)
            evaluated += len(results)

            offset += self.page_size
            if offset >= page_total:
                break
        return evaluated

    async def _evaluate(self, item: Dict[str, Any],
                        policy: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate one sample and shape it as the control plane's result record.

        This is exactly ``BenchmarkResult`` minus ``is_correct`` — correctness is scored on
        the control-plane side against labels that never leave it.
        """
        from znyx_core.core.models import EvaluationRequest

        stage = item.get("stage") or "input"
        request = EvaluationRequest(
            request_id=f"bench-{item.get('sample_id')}",
            tenant_id="benchmark",
            app_id="benchmark",
            text=item.get("eval_text") or "",
            metadata=item.get("metadata"),
        )
        import time
        t0 = time.perf_counter()
        try:
            resp = await self.evaluator.evaluate(request, context=stage, policy=policy)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "sample_id": item.get("sample_id"),
                "actual_decision": resp.decision.value if resp.decision else None,
                "actual_risk_score": resp.risk_score or 0,
                "actual_rule_hits": [h.rule_id for h in (resp.rule_hits or [])],
                "latency_ms": latency_ms,
                # execution_mode is what makes a silent fallback detectable: if the sidecar
                # was unreachable this says local_deterministic, and the control plane
                # refuses to publish a scorecard from a run where the model never fired.
                "detector_results": [
                    {"detector_name": d.detector_name, "decision": d.decision,
                     "risk_score": d.risk_score, "latency_ms": d.latency_ms,
                     "execution_mode": getattr(d, "execution_mode", None)}
                    for d in (resp.detector_results or [])
                ] or None,
            }
        except Exception as exc:  # noqa: BLE001 - one bad sample is a result, not a crash
            logger.debug("benchmark-worker: sample %s errored: %s",
                         item.get("sample_id"), exc)
            return {
                "sample_id": item.get("sample_id"),
                "actual_decision": "ERROR",
                "actual_risk_score": 0,
                "actual_rule_hits": [],
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "detector_results": None,
            }

    # ── transport ────────────────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.runtime_token}

    async def _claim(self) -> Optional[Dict[str, Any]]:
        resp = await self._request("POST", "/v1/benchmarks/claim", json={})
        if resp is None or resp.status_code == 204:
            return None
        if resp.status_code >= 400:
            logger.debug("benchmark-worker: claim rejected: %s %s",
                         resp.status_code, resp.text[:200])
            return None
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return None

    async def _fetch_samples(self, run_id: str,
                             offset: int) -> Optional[Dict[str, Any]]:
        resp = await self._request(
            "GET", f"/v1/benchmarks/{run_id}/samples",
            params={"offset": offset, "limit": self.page_size})
        if resp is None or resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return None

    async def _post_results(self, run_id: str, results: List[Dict[str, Any]]) -> bool:
        resp = await self._request("POST", f"/v1/benchmarks/{run_id}/results",
                                   json={"results": results})
        if resp is None or resp.status_code >= 400:
            raise RuntimeError(
                f"reporting results failed: {getattr(resp, 'status_code', 'no response')}")
        return True

    async def _complete(self, run_id: str, error: Optional[str] = None) -> bool:
        body: Dict[str, Any] = {"error": error} if error else {}
        resp = await self._request("POST", f"/v1/benchmarks/{run_id}/complete", json=body)
        return resp is not None and resp.status_code < 400

    async def _request(self, method: str, path: str, *, json: Any = None,
                       params: Any = None):
        """One HTTP call. Returns None on transport failure; latches off on 404 so an
        older control plane is not polled forever."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.request(
                    method, f"{self.control_plane_url}{path}",
                    json=json, params=params, headers=self._headers(),
                )
            # A 404 on claim means the control plane predates this feature. A 404 on a
            # run-scoped path just means that run is gone, which is not a reason to stop.
            if resp.status_code == 404 and path.endswith("/claim"):
                self._unsupported = True
                logger.debug("benchmark-worker: control plane has no benchmark dispatch; "
                             "disabling for this process")
            return resp
        except Exception as exc:  # noqa: BLE001 - never affect enforcement
            logger.debug("benchmark-worker: %s %s failed: %s", method, path, exc)
            return None
