"""Runner registry: builds one runner + batcher per configured task, isolating
load failures (heavy deps / missing pinned artifacts → RunnerUnavailable → the task is
marked unavailable and returns 503, the service stays up). Powers GET /v1/models.

Heavy runner kinds register their factory here on import; the dependency-free
``stub`` kind is always available.
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable, Dict, List, Optional

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


class RunnerRegistry:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._batchers: Dict[str, BatchProcessor] = {}
        self._models: Dict[str, ModelInfo] = {}
        for task, spec in config.task_specs.items():
            self._build(task, spec or {})

    def _build(self, task: str, spec: dict) -> None:
        kind = spec.get("runner", "stub")
        try:
            factory = _factory_for(kind)
            if factory is None:
                raise RunnerUnavailable(f"unknown runner kind '{kind}'")
            runner = factory(task, spec)
            runner.load()  # heavy runners verify sha256 / local_files_only here
            self._batchers[task] = BatchProcessor(
                runner,
                max_batch_size=self.config.max_batch_size,
                max_wait_ms=self.config.max_wait_ms,
                max_queue=self.config.max_queue,
                budget_ms=self.config.budget_ms,
            )
            self._models[task] = ModelInfo(
                task=task, model_version=runner.model_version, runner=kind, available=True,
                model_id=spec.get("model_id"), revision=spec.get("revision"),
                sha256=spec.get("sha256"),
            )
        except Exception as exc:  # noqa: BLE001
            # Any load failure (RunnerUnavailable, or an unexpected OSError/ValueError
            # from a corrupt artifact / bad tokenizer config / incompatible weights)
            # must isolate to THIS task — never crash the whole sidecar at startup.
            level = logging.WARNING if isinstance(exc, RunnerUnavailable) else logging.ERROR
            logger.log(level, "inference task '%s' (runner=%s) unavailable: %s",
                       task, kind, exc, exc_info=not isinstance(exc, RunnerUnavailable))
            detail = str(exc) if isinstance(exc, RunnerUnavailable) else f"load failed: {exc}"
            # Even unavailable, report the CONFIGURED pin (we know what it should be — it
            # just didn't load) so a registry sync preserves provenance instead of erasing
            # it to a sentinel. Falls back to "-" only when truly unpinned.
            mid, rev = spec.get("model_id"), spec.get("revision")
            model_version = f"{mid}@{rev}" if mid and rev else (mid or "-")
            self._models[task] = ModelInfo(
                task=task, model_version=model_version, runner=kind, available=False,
                model_id=mid, revision=rev, sha256=spec.get("sha256"), detail=detail,
            )

    def get(self, task: str) -> Optional[BatchProcessor]:
        return self._batchers.get(task)

    def model_version(self, task: str) -> Optional[str]:
        info = self._models.get(task)
        return info.model_version if info and info.available else None

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    async def start_all(self) -> None:
        for b in self._batchers.values():
            await b.start()

    async def stop_all(self) -> None:
        for b in self._batchers.values():
            await b.stop()

    def batcher_stats(self) -> Dict[str, dict]:
        return {task: b.stats() for task, b in self._batchers.items()}
