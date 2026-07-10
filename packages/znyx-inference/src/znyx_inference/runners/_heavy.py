"""Shared skeleton for heavy (ML) runners. Keeps the no-egress / fail-closed
posture in one place: lazy-import the stack (missing → RunnerUnavailable), resolve the
LOCAL pinned artifact dir, verify its sha256, then build the model with
``local_files_only=True``. No heavy import at module load — only inside ``load()``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from znyx_inference.runners._artifacts import resolve_artifact_dir, verify_pinned
from znyx_inference.runners.base import InferOutput, Runner, RunnerUnavailable


class HeavyRunner(Runner):
    runner_kind = "heavy"

    def __init__(self, task: str, spec: Dict[str, Any]):
        self.task = task
        self.spec = spec or {}
        self.model_id = self.spec.get("model_id") or task
        self.revision = self.spec.get("revision") or "main"
        self.threshold = float(self.spec.get("threshold", 0.5))
        self.model_version = f"{self.model_id}@{self.revision}"
        self._ready = False
        self._artifact_dir = ""

    # subclasses implement these
    def _import_stack(self) -> None:
        """Lazy-import the ML deps; raise RunnerUnavailable if absent."""
        raise NotImplementedError

    def _build(self, path: str) -> None:
        """Load model objects from the verified local ``path``."""
        raise NotImplementedError

    def load(self) -> None:
        self._import_stack()                                   # deps present?
        self._artifact_dir = resolve_artifact_dir(self.spec)   # local only
        verify_pinned(self._artifact_dir, self.spec.get("sha256"))  # exists + sha256 ok
        self._build(self._artifact_dir)
        self._ready = True

    def _decision(self, unsafe_prob: float) -> str:
        if unsafe_prob >= self.threshold:
            return "BLOCK"
        if unsafe_prob >= self.threshold / 2:
            return "WARN"
        return "ALLOW"

    def _output(self, unsafe_prob: float, label_scores: Dict[str, float] | None = None) -> InferOutput:
        unsafe_prob = max(0.0, min(1.0, float(unsafe_prob)))
        # Defensively clamp every label score into the contract's [0,1] range so a
        # runner's raw value (e.g. a negative cosine) can't turn a successful inference
        # into a response-validation 500. Runners should still emit probabilities here.
        if label_scores:
            label_scores = {k: round(max(0.0, min(1.0, float(v))), 4)
                            for k, v in label_scores.items()}
        return InferOutput(
            decision=self._decision(unsafe_prob),
            risk_score=int(round(unsafe_prob * 100)),
            confidence=round(unsafe_prob, 4),
            calibrated_score=round(unsafe_prob, 4),
            label_scores=label_scores,
            threshold=self.threshold,
        )

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        raise NotImplementedError


def require(import_fn, what: str):
    """Run a lazy import, converting ImportError into RunnerUnavailable so a missing ML
    stack degrades the task to 'unavailable' instead of crashing the service."""
    try:
        return import_fn()
    except ImportError as exc:  # noqa: PERF203
        raise RunnerUnavailable(f"{what} not installed (inference image only): {exc}") from exc


# Preferred ONNX file names, best (quantized) first. The offline export tool
# (scripts/fetch_inference_model.py) writes model_quantized.onnx alongside model.onnx;
# we serve the quantized graph when present so the CPU image is fast and small.
_ONNX_FILE_PREFERENCE = ("model_quantized.onnx", "model.onnx")

# onnxruntime input type string → numpy dtype. Models vary (optimum exports int64 ids, but a
# bring-your-own graph may want int32); feed each input in the dtype the graph declares so a
# mismatch can't surface as a request-time INVALID_ARGUMENT after load reported 'available'.
_ORT_TO_NP = {
    "tensor(int64)": "int64", "tensor(int32)": "int32", "tensor(int8)": "int8",
    "tensor(float)": "float32", "tensor(float16)": "float16", "tensor(double)": "float64",
    "tensor(bool)": "bool",
}


class OnnxTextRunner(HeavyRunner):
    """Shared base for encoder runners served on CPU via ``onnxruntime`` + the standalone
    Rust ``tokenizers`` (NO torch, NO transformers at serve time). Loads an ONNX graph +
    ``tokenizer.json`` + ``config.json`` from the verified local artifact dir, tokenizes a
    batch, runs the session, and hands subclasses back raw numpy logits/hidden-states.

    Everything heavy is imported in ``load()`` (``_import_stack``), so importing the runner
    module stays torch-free and the task degrades to 'unavailable' when ``[onnx]`` is absent.
    """
    runner_kind = "onnx"
    max_length = 512

    def _import_stack(self) -> None:
        self._ort = require(lambda: __import__("onnxruntime"), "onnxruntime")
        require(lambda: __import__("tokenizers"), "tokenizers")
        self._np = require(lambda: __import__("numpy"), "numpy")

    def _resolve_onnx_file(self, path):
        from pathlib import Path
        p = Path(path)
        for name in _ONNX_FILE_PREFERENCE:
            if (p / name).is_file():
                return p / name
        found = sorted(p.glob("*.onnx"))
        if not found:
            raise RunnerUnavailable(f"no .onnx model in artifact dir {path} "
                                    "(export it with scripts/fetch_inference_model.py)")
        return found[0]

    def _build(self, path: str) -> None:
        import json
        from pathlib import Path

        from tokenizers import Tokenizer
        p = Path(path)
        onnx_file = self._resolve_onnx_file(p)
        # CPU-only, deterministic. GPU is a future EP swap (onnxruntime-gpu), not a rebuild.
        self._session = self._ort.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"])
        # name → numpy dtype the graph expects for that input (default int64 for id tensors).
        self._input_dtypes = {i.name: _ORT_TO_NP.get(i.type, "int64")
                              for i in self._session.get_inputs()}
        self._input_names = set(self._input_dtypes)

        tok_file = p / "tokenizer.json"
        if not tok_file.is_file():
            raise RunnerUnavailable(f"tokenizer.json missing in artifact dir {path}")
        self._tok = Tokenizer.from_file(str(tok_file))
        self._tok.enable_truncation(max_length=self.max_length)
        self._tok.enable_padding()   # pad to the longest item in each batch (mask handles it)

        cfg = {}
        cfg_file = p / "config.json"
        if cfg_file.is_file():
            cfg = json.loads(cfg_file.read_text())
        self._id2label = {int(k): v for k, v in (cfg.get("id2label") or {}).items()}
        self._build_extra(cfg)

    def _build_extra(self, cfg: Dict[str, Any]) -> None:
        """Optional subclass hook for extra artifact-derived state (default: nothing)."""

    # --- inference primitives shared by all encoder subclasses --------------------
    def _forward(self, inputs):
        """Tokenize ``inputs`` (each item a str, or a ``(text, pair)`` tuple for cross-encoders),
        run the session, and return ``(logits_or_hidden, encodings)``. ``encodings`` exposes
        per-token ``attention_mask`` so token-level runners (NER) can skip padding."""
        np = self._np
        encs = self._tok.encode_batch(list(inputs))
        raw = {
            "input_ids": [e.ids for e in encs],
            "attention_mask": [e.attention_mask for e in encs],
            "token_type_ids": [e.type_ids for e in encs],
        }
        # Feed only the inputs the graph declares, each in the dtype it expects.
        feeds = {name: np.array(vals, dtype=self._input_dtypes[name])
                 for name, vals in raw.items() if name in self._input_names}
        out = self._session.run(None, feeds)[0]
        return np.asarray(out), encs

    def _softmax(self, x, axis=-1):
        np = self._np
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)
