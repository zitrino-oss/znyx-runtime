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

Fetching/exporting is the ONLY step that needs the heavy stack (torch + optimum, the
``[export]`` extra). SERVING then needs only the lean ``[onnx]`` stack — that asymmetry is
the whole point: the multi-GB torch/CUDA payload lives here, offline, and never ships in the
inference image.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from znyx_inference.runners._artifacts import artifact_sha256, resolve_artifact_dir
from znyx_inference.runners._heavy import require

logger = logging.getLogger(__name__)

# Runner kinds served on CPU via onnxruntime → their pinned artifact must be an ONNX graph,
# so the default fetch EXPORTS + dynamically-quantizes the checkpoint to ONNX. guard_llm
# (torch, [torch] extra) keeps a raw HF snapshot instead.
_ONNX_RUNNERS = {"classifier", "ner", "language", "nli", "embedding"}

# runner kind → the optimum ORT auto-class used to export that model to ONNX.
_ORT_EXPORT_CLASS = {
    "classifier": "ORTModelForSequenceClassification",
    "language": "ORTModelForSequenceClassification",
    "nli": "ORTModelForSequenceClassification",
    "ner": "ORTModelForTokenClassification",
    "embedding": "ORTModelForFeatureExtraction",
}


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


def _snapshot_download(model_id: str, revision: str, dest_dir: str) -> None:
    """Download a raw HuggingFace model snapshot into ``dest_dir`` (used for torch-served
    runners like guard_llm). Lazy-imports ``huggingface_hub`` (missing → RunnerUnavailable)."""
    hf = require(lambda: __import__("huggingface_hub", fromlist=["snapshot_download"]),
                 "huggingface_hub")
    hf.snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=dest_dir,
        # Pull weights + tokenizer + config only; skip framework duplicates / git metadata.
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", ".gitattributes"],
    )


def _export_onnx(model_id: str, revision: str, dest_dir: str, runner: str) -> None:
    """Export ``model_id`` to an ONNX graph + tokenizer.json under ``dest_dir`` and, when
    possible, write a dynamically-quantized (int8) ``model_quantized.onnx`` alongside it —
    the artifact the lean CPU sidecar serves. Needs the ``[export]`` extra (torch + optimum +
    transformers); those are lazy-imported here (missing → RunnerUnavailable) and never
    reach the serving image. This is the deliberate, one-time, offline heavy step."""
    ort_mod = require(
        lambda: __import__("optimum.onnxruntime", fromlist=list(_ORT_EXPORT_CLASS.values())),
        "optimum[exporters,onnxruntime] ([export] extra)")
    tfm = require(lambda: __import__("transformers", fromlist=["AutoTokenizer"]),
                  "transformers ([export] extra)")
    cls_name = _ORT_EXPORT_CLASS[runner]
    ort_cls = getattr(ort_mod, cls_name)
    # export=True converts the checkpoint to ONNX on the fly (no pre-existing .onnx needed).
    model = ort_cls.from_pretrained(model_id, revision=revision, export=True)
    model.save_pretrained(dest_dir)
    # Save the fast tokenizer as tokenizer.json — what the serving runner loads (no torch).
    tfm.AutoTokenizer.from_pretrained(model_id, revision=revision).save_pretrained(dest_dir)
    _quantize_onnx(dest_dir)


def _quantize_onnx(dest_dir: str) -> None:
    """Best-effort dynamic int8 quantization of the exported ONNX graph (CPU). Non-fatal:
    if quantization is unavailable for this op set, the fp32 ``model.onnx`` still serves."""
    try:
        ort_mod = __import__("optimum.onnxruntime",
                             fromlist=["ORTQuantizer", "configuration"])
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        quantizer = ort_mod.ORTQuantizer.from_pretrained(dest_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=dest_dir, quantization_config=qconfig)
        logger.info("wrote quantized ONNX (int8) into %s", dest_dir)
    except Exception as exc:  # noqa: BLE001 — quantization is an optimization, not required
        logger.warning("ONNX quantization skipped (%s); serving fp32 model.onnx", exc)


def _default_downloader(model_id: str, revision: str, dest_dir: str,
                        runner: Optional[str] = None) -> None:
    """Dispatch the offline fetch by runner kind: ONNX-served runners get an ONNX export +
    quantization; everything else (guard_llm / torch) gets a raw HF snapshot."""
    if runner in _ONNX_RUNNERS:
        _export_onnx(model_id, revision, dest_dir, runner)
    else:
        _snapshot_download(model_id, revision, dest_dir)


def fetch_model(model_id: str, revision: str, dest_dir: str, *,
                runner: Optional[str] = None,
                downloader: Optional[Callable[[str, str, str], None]] = None) -> str:
    """Fetch ``model_id@revision`` into ``dest_dir`` and return the artifact sha256 to PIN.

    Explicit operator action only — never the inference path. For an ONNX-served ``runner``
    the default fetch exports + quantizes to ONNX; otherwise it snapshots the raw weights.
    ``downloader`` is injectable (tests pass a fake with the ``(model_id, revision, dest_dir)``
    signature; when omitted, the runner-aware default is used). Returns the digest computed
    over the resulting tree by the same ``artifact_sha256`` the runner verifies with, so the
    operator pins exactly what will be loaded."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    effective = downloader or (lambda m, r, d: _default_downloader(m, r, d, runner))
    effective(model_id, revision, dest_dir)
    digest = artifact_sha256(dest_dir)
    logger.info("fetched %s@%s into %s (sha256=%s)", model_id, revision, dest_dir, digest)
    return digest


def list_candidates(task: Optional[str] = None, *, open_only: bool = False) -> List[Dict[str, Any]]:
    """The vetted candidate shortlist (catalog appendix-M), optionally filtered to one task
    and/or open-license-only — for the fetch CLI's ``--list``."""
    from znyx_core.engine.ml_catalog import candidate_models
    rows = candidate_models(open_only=open_only)
    return [r for r in rows if task is None or r["task"] == task]
