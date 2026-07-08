"""Embedding-similarity runner (F3) — a centroid of operator-supplied unsafe examples;
risk = max cosine similarity to that centroid. Used for fuzzy/semantic escalation (e.g.
system_prompt_leakage in P2). Served on CPU via onnxruntime + tokenizers (torch-free): the
model is an ONNX feature-extraction graph (``ORTModelForFeatureExtraction`` export) whose
token hidden-states we masked-mean-pool and L2-normalize into a sentence embedding — the
sentence-transformers default pooling, with no torch at serve time. The ONNX graph loads
only in ``load()`` from the verified local artifact dir.
"""
from __future__ import annotations

from typing import Any, Dict, List

from znyx_inference.runners._heavy import OnnxTextRunner
from znyx_inference.runners.base import InferOutput, RunnerUnavailable


class EmbeddingRunner(OnnxTextRunner):
    runner_kind = "embedding"

    def _build_extra(self, cfg: Dict[str, Any]) -> None:
        examples = self.spec.get("unsafe_examples") or []
        if not examples:
            raise RunnerUnavailable("embedding runner requires spec['unsafe_examples']")
        centroid = self._embed(list(examples)).mean(axis=0)
        norm = self._np.linalg.norm(centroid) or 1.0
        self._centroid = centroid / norm                  # renormalize for a clean cosine

    def _embed(self, texts: List[str]):
        """L2-normalized sentence embeddings for ``texts`` [N, H]. Masked-mean-pools token
        hidden-states [N, T, H]; if the graph already emits a pooled [N, H] embedding, uses
        it directly."""
        np = self._np
        out, encs = self._forward(texts)
        if out.ndim == 2:                                 # already-pooled sentence embedding
            pooled = out
        else:                                             # token hidden-states → masked mean
            mask = np.array([e.attention_mask for e in encs], dtype=np.float32)  # [N, T]
            mask = mask[:, :out.shape[1], None]           # align to T, broadcast over H
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = (out * mask).sum(axis=1) / counts    # mean over real tokens
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return pooled / norms

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        embs = self._embed(list(texts))
        sims = embs @ self._centroid                      # cosine (both normalized)
        return [self._output(max(0.0, float(s)), {"unsafe_similarity": round(float(s), 4)})
                for s in sims]


def make_runner(task: str, spec: Dict[str, Any]) -> EmbeddingRunner:
    return EmbeddingRunner(task, spec)
