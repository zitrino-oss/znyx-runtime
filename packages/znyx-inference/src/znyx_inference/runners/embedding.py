"""Embedding-similarity runner (F3) — sentence-transformers + a centroid of operator-
supplied unsafe examples; risk = max cosine similarity to that centroid. Used for
fuzzy/semantic escalation (e.g. system_prompt_leakage in P2). Heavy deps load only in
``load()``; weights from the verified local artifact dir.
"""
from __future__ import annotations

from typing import Any, Dict, List

from znyx_inference.runners._heavy import HeavyRunner, require
from znyx_inference.runners.base import InferOutput, RunnerUnavailable


class EmbeddingRunner(HeavyRunner):
    runner_kind = "embedding"

    def _import_stack(self) -> None:
        require(lambda: __import__("sentence_transformers"), "sentence-transformers")
        self._np = require(lambda: __import__("numpy"), "numpy")

    def _build(self, path: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(path)
        examples = self.spec.get("unsafe_examples") or []
        if not examples:
            raise RunnerUnavailable("embedding runner requires spec['unsafe_examples']")
        embs = self._model.encode(list(examples), normalize_embeddings=True)
        self._centroid = self._np.asarray(embs).mean(axis=0)
        # renormalize centroid for a clean cosine
        norm = self._np.linalg.norm(self._centroid) or 1.0
        self._centroid = self._centroid / norm

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        np = self._np
        embs = np.asarray(self._model.encode(list(texts), normalize_embeddings=True))
        sims = embs @ self._centroid                      # cosine (both normalized)
        return [self._output(max(0.0, float(s)), {"unsafe_similarity": round(float(s), 4)})
                for s in sims]


def make_runner(task: str, spec: Dict[str, Any]) -> EmbeddingRunner:
    return EmbeddingRunner(task, spec)
