"""Runner registry: builds one runner + batcher per configured task, isolating
load failures (heavy deps / missing pinned artifacts → RunnerUnavailable → the task is
marked unavailable and returns 503, the service stays up). Powers GET /v1/models.

Multi-model: beyond the per-task ACTIVE runner (the catalog default an operator reloads),
the registry can hold additional loaded VARIANTS keyed by (task, model_id@revision). A
request that pins model_id/revision routes to its variant, so several projects sharing one
sidecar can each be served the model THEY pinned instead of colliding on a single slot.
Variants are loaded by the pin-sync service (bundle-delivered pins) or explicitly.

Heavy runner kinds register their factory here on import; the dependency-free
``stub`` kind is always available.
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable, Dict, List, Optional, Tuple

from znyx_inference.batching import BatchProcessor
from znyx_inference.config import InferenceConfig
from znyx_inference.contract import ModelInfo
from znyx_inference.runners.base import Runner, RunnerUnavailable, StubRunner

logger = logging.getLogger(__name__)

# runner_kind → factory(task: str, spec: dict) -> Runner. The dependency-free stub is
# always present; heavy kinds are lazy-imported on demand (their modules don't import
# torch at module load — only in Runner.load()).
RUNNER_FACTORIES: Dict[str, Callable[[str, dict], Runner]] = {
    "stub": lambda task, spec: StubRunner(task=task, threshold=float(spec.get("threshold", 0.5))),
}

# Heavy runner kind → module exposing make_runner(task, spec). Imported only when a
# policy actually asks for that kind, so the core/control-plane never pull these in.
_HEAVY_KIND_MODULES = {
    "classifier": "znyx_inference.runners.classifier",
    "embedding": "znyx_inference.runners.embedding",
    "nli": "znyx_inference.runners.nli",
    "guard_llm": "znyx_inference.runners.guard_llm",
    "ner": "znyx_inference.runners.ner",                # token-level PII NER (unstructured)
    "language": "znyx_inference.runners.language",       # language-ID + allow/block mapping
}


def _factory_for(kind: str) -> Optional[Callable[[str, dict], Runner]]:
    if kind in RUNNER_FACTORIES:
        return RUNNER_FACTORIES[kind]
    module_path = _HEAVY_KIND_MODULES.get(kind)
    if module_path is None:
        return None
    module = importlib.import_module(module_path)   # light: no torch at import time
    return module.make_runner


def variant_key(model_id: Optional[str], revision: Optional[str]) -> str:
    """Stable identity of one loaded model: ``model_id@revision`` (revision → 'main')."""
    return f"{model_id or 'model'}@{revision or 'main'}"


class RunnerRegistry:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._batchers: Dict[str, BatchProcessor] = {}
        self._models: Dict[str, ModelInfo] = {}
        # (task, model_id@revision) → additionally loaded variant, NOT the active slot.
        self._variant_batchers: Dict[Tuple[str, str], BatchProcessor] = {}
        self._variant_models: Dict[Tuple[str, str], ModelInfo] = {}
        for task, spec in config.task_specs.items():
            self._build(task, spec or {})

    def _make(self, task: str, spec: dict, *, active: bool) -> Tuple[Optional[BatchProcessor], ModelInfo]:
        """Build (batcher, info) for one spec; on failure return (None, unavailable-info).
        Shared by the active slot and variants so both isolate load failures the same way."""
        kind = spec.get("runner", "stub")
        try:
            factory = _factory_for(kind)
            if factory is None:
                raise RunnerUnavailable(f"unknown runner kind '{kind}'")
            runner = factory(task, spec)
            runner.load()  # heavy runners verify sha256 / local_files_only here
            batcher = BatchProcessor(
                runner,
                max_batch_size=self.config.max_batch_size,
                max_wait_ms=self.config.max_wait_ms,
                max_queue=self.config.max_queue,
                budget_ms=self.config.budget_ms,
            )
            # When the spec pins a model, that pin IS the served identity (heavy runners
            # already derive model_version from it; the stub reports its own). Keeping the
            # pin authoritative makes pinned routing, response identity and the per-model
            # cache key line up — mirroring the failure path below, which reports the
            # configured pin for the same reason.
            mid, rev = spec.get("model_id"), spec.get("revision")
            model_version = f"{mid}@{rev or 'main'}" if mid else runner.model_version
            info = ModelInfo(
                task=task, model_version=model_version, runner=kind, available=True,
                model_id=mid, revision=rev,
                sha256=spec.get("sha256"), active=active,
            )
            return batcher, info
        except Exception as exc:  # noqa: BLE001
            # Any load failure (RunnerUnavailable, or an unexpected OSError/ValueError
            # from a corrupt artifact / bad tokenizer config / incompatible weights)
            # must isolate to THIS task/variant — never crash the whole sidecar.
            level = logging.WARNING if isinstance(exc, RunnerUnavailable) else logging.ERROR
            logger.log(level, "inference task '%s' (runner=%s) unavailable: %s",
                       task, kind, exc, exc_info=not isinstance(exc, RunnerUnavailable))
            detail = str(exc) if isinstance(exc, RunnerUnavailable) else f"load failed: {exc}"
            # Even unavailable, report the CONFIGURED pin (we know what it should be — it
            # just didn't load) so a registry sync preserves provenance instead of erasing
            # it to a sentinel. Falls back to "-" only when truly unpinned.
            mid, rev = spec.get("model_id"), spec.get("revision")
            model_version = f"{mid}@{rev}" if mid and rev else (mid or "-")
            info = ModelInfo(
                task=task, model_version=model_version, runner=kind, available=False,
                model_id=mid, revision=rev, sha256=spec.get("sha256"), detail=detail,
                active=active,
            )
            return None, info

    def _build(self, task: str, spec: dict) -> None:
        batcher, info = self._make(task, spec, active=True)
        if batcher is not None:
            self._batchers[task] = batcher
        self._models[task] = info

    # ── Lookup ──────────────────────────────────────────────────────────────────────

    def get(self, task: str) -> Optional[BatchProcessor]:
        return self._batchers.get(task)

    def model_version(self, task: str) -> Optional[str]:
        info = self._models.get(task)
        return info.model_version if info and info.available else None

    @staticmethod
    def _matches(info: ModelInfo, model_id: str, revision: Optional[str]) -> bool:
        """Does a loaded model satisfy a ``model_id[@revision]`` pin? Prefers the
        configured pin fields; falls back to parsing model_version ('id@rev')."""
        if info.model_id:
            loaded_id, loaded_rev = info.model_id, (info.revision or "main")
        else:
            loaded_id, _, loaded_rev = info.model_version.partition("@")
        return model_id == loaded_id and (revision is None or revision == loaded_rev)

    def get_for(self, task: str, model_id: str,
                revision: Optional[str] = None) -> Optional[Tuple[BatchProcessor, str]]:
        """Resolve the batcher serving exactly ``model_id[@revision]`` for ``task`` —
        the active slot when it matches, else a loaded variant, else None. Returns
        (batcher, model_version) so responses/caching carry the served model's identity."""
        active = self._models.get(task)
        if active is not None and active.available and self._matches(active, model_id, revision):
            batcher = self._batchers.get(task)
            if batcher is not None:
                return batcher, active.model_version
        for (v_task, _v_key), batcher in self._variant_batchers.items():
            if v_task != task:
                continue
            info = self._variant_models.get((v_task, _v_key))
            if info is not None and info.available and self._matches(info, model_id, revision):
                return batcher, info.model_version
        return None

    def serves(self, task: str, model_id: str, revision: Optional[str] = None) -> bool:
        return self.get_for(task, model_id, revision) is not None

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values()) + list(self._variant_models.values())

    # ── Mutation ────────────────────────────────────────────────────────────────────

    async def reload_task(self, task: str, spec: dict) -> ModelInfo:
        """Hot-reload a task's ACTIVE runner after a model install. Stops the old batcher,
        builds a new runner + batcher from the updated spec, and starts it."""
        old = self._batchers.pop(task, None)
        if old is not None:
            await old.stop()
        self._build(task, spec)
        new = self._batchers.get(task)
        if new is not None:
            await new.start()
        return self._models[task]

    async def load_variant(self, task: str, spec: dict) -> ModelInfo:
        """Load an ADDITIONAL model for ``task`` without touching the active slot.

        Idempotent per (task, model_id@revision): reloading an available variant is a
        no-op; a previously failed variant is retried. Used by pin-sync so each
        project's pinned model is servable side by side with the catalog default."""
        key = (task, variant_key(spec.get("model_id"), spec.get("revision")))
        existing = self._variant_models.get(key)
        if existing is not None and existing.available:
            return existing

        old = self._variant_batchers.pop(key, None)
        if old is not None:
            await old.stop()

        batcher, info = self._make(task, spec, active=False)
        if batcher is not None:
            self._variant_batchers[key] = batcher
            await batcher.start()
        self._variant_models[key] = info
        return info

    async def start_all(self) -> None:
        for b in self._batchers.values():
            await b.start()
        for b in self._variant_batchers.values():
            await b.start()

    async def stop_all(self) -> None:
        for b in self._batchers.values():
            await b.stop()
        for b in self._variant_batchers.values():
            await b.stop()

    def batcher_stats(self) -> Dict[str, dict]:
        stats = {task: b.stats() for task, b in self._batchers.items()}
        stats.update({f"{task}::{key}": b.stats()
                      for (task, key), b in self._variant_batchers.items()})
        return stats
