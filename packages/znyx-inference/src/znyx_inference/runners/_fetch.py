"""Explicit, operator-run model fetch (F3 / P5 Step 1).

The heavy runners load weights ONLY from a verified LOCAL artifact dir and NEVER pull from
the network during inference — see ``_artifacts.py`` (the no-implicit-download warnbox). This
module is the one DELIBERATE, operator-invoked exception: it downloads a pinned public model
into that local dir and computes the sha256 the runner will then verify. It is **never**
called on the inference path — only by ``scripts/fetch_inference_model.py``.

Why an explicit fetch (not an automatic runtime pull): for a security product, a model is a
supply-chain artifact. Downloading it is a deliberate, auditable act that yields a sha256 the
operator pins — so the weights serving traffic are exactly the reviewed ones, and a silent
swap can't happen under us. ``resolve_fetch_target`` also enforces the curated shortlist
(``ml_catalog``): you fetch a vetted model by default, and an off-list model only with an
explicit opt-out.

Fetching is light (it needs only ``huggingface_hub`` — already present via ``datasets``);
SERVING the model needs the heavy stack in ``requirements-inference.txt``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from znyx_inference.runners._artifacts import artifact_sha256, resolve_artifact_dir
from znyx_inference.runners._heavy import require

logger = logging.getLogger(__name__)


def resolve_fetch_target(task: str, *, model_id: Optional[str] = None,
                         revision: Optional[str] = None,
                         allow_unvetted: bool = False) -> Dict[str, Any]:
    """Resolve which model to fetch for ``task`` and where it lands.

    Defaults to the catalog's pinned model (``ML_TASKS[task]``). ``model_id`` may override
    it, but ONLY to another model on that task's vetted shortlist (``CANDIDATE_MODELS``) —
    unless ``allow_unvetted`` is set (the deferred bring-your-own-model escape hatch). Raises
    ``ValueError`` on an unknown task or an off-shortlist model without the opt-out.
    Dependency-free (``ml_catalog`` is stdlib-only)."""
    from znyx_core.engine.ml_catalog import CANDIDATE_MODELS, ML_TASKS

    spec = ML_TASKS.get(task)
    if spec is None:
        raise ValueError(f"unknown task '{task}'. Known tasks: {sorted(ML_TASKS)}")

    chosen_id = model_id or spec.model_id
    chosen_rev = revision or spec.revision
    if model_id and not allow_unvetted:
        vetted = {c.model_id for c in CANDIDATE_MODELS.get(task, [])} | {spec.model_id}
        if model_id not in vetted:
            raise ValueError(
                f"'{model_id}' is not on the vetted shortlist for task '{task}'. "
                f"Choose one of {sorted(vetted)}, or pass allow_unvetted=True to override.")
    return {
        "task": task,
        "runner": spec.runner,
        "model_id": chosen_id,
        "revision": chosen_rev,
        "dest_dir": resolve_artifact_dir({"model_id": chosen_id}),
    }


def _default_downloader(model_id: str, revision: str, dest_dir: str) -> None:
    """Download a HuggingFace model snapshot into ``dest_dir``. Lazy-imports
    ``huggingface_hub`` (missing → RunnerUnavailable, via ``require``) so neither the core
    install nor the lean runtime needs it until an operator actually fetches."""
    hf = require(lambda: __import__("huggingface_hub", fromlist=["snapshot_download"]),
                 "huggingface_hub")
    hf.snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=dest_dir,
        # Pull weights + tokenizer + config only; skip framework duplicates / git metadata.
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", ".gitattributes"],
    )


def fetch_model(model_id: str, revision: str, dest_dir: str, *,
                downloader: Optional[Callable[[str, str, str], None]] = None) -> str:
    """Download ``model_id@revision`` into ``dest_dir`` and return the artifact sha256 to PIN.

    Explicit operator action only — never the inference path. ``downloader`` is injectable
    (tests pass a fake; the default uses ``huggingface_hub``). Returns the digest computed
    over the downloaded tree by the same ``artifact_sha256`` the runner verifies with, so the
    operator pins exactly what will be loaded."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (downloader or _default_downloader)(model_id, revision, dest_dir)
    digest = artifact_sha256(dest_dir)
    logger.info("fetched %s@%s into %s (sha256=%s)", model_id, revision, dest_dir, digest)
    return digest


def list_candidates(task: Optional[str] = None, *, open_only: bool = False) -> List[Dict[str, Any]]:
    """The vetted candidate shortlist (catalog appendix-M), optionally filtered to one task
    and/or open-license-only — for the fetch CLI's ``--list``."""
    from znyx_core.engine.ml_catalog import candidate_models
    rows = candidate_models(open_only=open_only)
    return [r for r in rows if task is None or r["task"] == task]
