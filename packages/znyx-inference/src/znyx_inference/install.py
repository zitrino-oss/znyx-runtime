"""Background model install manager for the inference sidecar.

Handles explicit, operator-triggered model installs via the UI. Downloads a vetted
model from the HuggingFace Hub, exports to ONNX (for CPU-served runners), and writes
the artifact into ZNYX_INFERENCE_ARTIFACTS_DIR. One install at a time (serialized).

This is the HTTP-driven equivalent of ``scripts/fetch_inference_model.py`` — the same
``_fetch.py`` pipeline, triggered from the console instead of the CLI.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InstallJob:
    job_id: str
    task: str
    model_id: str
    revision: str
    runner: str
    status: str = "pending"  # pending → downloading → exporting → complete | failed
    sha256: Optional[str] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InstallManager:
    """In-memory job tracker. Runs installs in a single background thread so the
    fetch/export (CPU-bound, minutes-long) doesn't block the uvicorn event loop."""

    def __init__(self) -> None:
        self._jobs: Dict[str, InstallJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _active_job(self) -> Optional[InstallJob]:
        for job in self._jobs.values():
            if job.status in ("pending", "downloading", "exporting"):
                return job
        return None

    def start_install(self, task: str, model_id: Optional[str] = None,
                      revision: Optional[str] = None) -> InstallJob:
        from znyx_inference.runners._fetch import resolve_fetch_target

        with self._lock:
            active = self._active_job()
            if active is not None:
                raise RuntimeError(
                    f"install already in progress: {active.model_id} (job {active.job_id})")

        target = resolve_fetch_target(task, model_id=model_id, revision=revision)
        job = InstallJob(
            job_id=str(uuid.uuid4()),
            task=target["task"],
            model_id=target["model_id"],
            revision=target["revision"],
            runner=target["runner"],
        )
        with self._lock:
            self._jobs[job.job_id] = job

        self._executor.submit(self._run_install, job, target)
        logger.info("install job %s started: %s for task %s",
                     job.job_id, job.model_id, job.task)
        return job

    def get_job(self, job_id: str) -> Optional[InstallJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[InstallJob]:
        return list(self._jobs.values())

    def _run_install(self, job: InstallJob, target: Dict[str, Any]) -> None:
        from znyx_inference.runners._fetch import fetch_model

        try:
            job.status = "downloading"
            logger.info("job %s: downloading %s@%s → %s",
                         job.job_id, target["model_id"], target["revision"],
                         target["dest_dir"])

            sha = fetch_model(
                target["model_id"], target["revision"], target["dest_dir"],
                runner=target["runner"],
            )

            job.sha256 = sha
            job.status = "complete"
            job.completed_at = time.time()
            logger.info("job %s: complete (sha256=%s)", job.job_id, sha)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = time.time()
            logger.error("job %s: failed: %s", job.job_id, exc, exc_info=True)
