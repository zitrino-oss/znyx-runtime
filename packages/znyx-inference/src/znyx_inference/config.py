"""Inference service configuration (F3). Env-driven; safe dependency-free defaults so
the service boots on the StubRunner with no ML stack installed.

Per-task spec keys:
  runner: stub | classifier | embedding | nli | guard_llm
  model_id / revision / sha256: pinned artifact identity (heavy runners; verified on load)
  threshold: decision threshold
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

# Default task set runs on the dependency-free StubRunner.
_DEFAULT_TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "prompt_injection": {"runner": "stub", "threshold": 0.5},
    "toxicity": {"runner": "stub", "threshold": 0.5},
    "jailbreak": {"runner": "stub", "threshold": 0.5},
}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class InferenceConfig:
    task_specs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: dict(_DEFAULT_TASK_SPECS))
    max_batch_size: int = 16
    max_wait_ms: int = 10
    max_queue: int = 256
    budget_ms: int = 2000
    cache_maxsize: int = 4096
    # Artifacts dir + no-download posture for heavy runners (F3 warnbox).
    model_artifacts_dir: str = ""
    require_local_files: bool = True     # never pull weights from the network at startup

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        cfg = cls(
            max_batch_size=_int("ZNYX_INFERENCE_MAX_BATCH", 16),
            max_wait_ms=_int("ZNYX_INFERENCE_MAX_WAIT_MS", 10),
            max_queue=_int("ZNYX_INFERENCE_MAX_QUEUE", 256),
            budget_ms=_int("ZNYX_INFERENCE_BUDGET_MS", 2000),
            cache_maxsize=_int("ZNYX_INFERENCE_CACHE_SIZE", 4096),
            model_artifacts_dir=os.getenv("ZNYX_INFERENCE_ARTIFACTS_DIR", ""),
            require_local_files=os.getenv("ZNYX_INFERENCE_REQUIRE_LOCAL_FILES", "true").lower()
            not in ("0", "false", "no"),
        )
        # Optional JSON override of the task→spec map (e.g. to wire heavy runners).
        raw = os.getenv("ZNYX_INFERENCE_TASKS")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed:
                    cfg.task_specs = parsed
                    return cfg
            except (ValueError, TypeError):
                pass
        # Otherwise, an opt-in named profile. ZNYX_INFERENCE_PROFILE=heavy loads the
        # canonical ML task catalog (classifier/nli/embedding/guard_llm pins). The boot
        # default stays "stub" so the service runs with zero ML deps unless asked.
        if os.getenv("ZNYX_INFERENCE_PROFILE", "stub").lower() == "heavy":
            from znyx_core.engine.ml_catalog import inference_task_specs
            specs = inference_task_specs()
            if specs:
                cfg.task_specs = specs
        return cfg

    def artifact_path(self, *parts: str) -> str:
        base = self.model_artifacts_dir or str(Path.home() / ".znyx" / "models")
        return str(Path(base, *parts))
