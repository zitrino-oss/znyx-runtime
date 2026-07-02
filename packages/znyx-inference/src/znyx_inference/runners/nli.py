"""NLI cross-encoder runner (F3) — entailment between a premise (provided source) and a
hypothesis (a claim/quote), for grounding checks (e.g. citation_integrity NLI-backed in
P2). The input text is a JSON object ``{"premise": ..., "hypothesis": ...}`` (or
``premise [SEP] hypothesis``); risk = contradiction/non-entailment probability. Heavy
deps load only in ``load()``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from znyx_inference.runners._heavy import HeavyRunner, require
from znyx_inference.runners.base import InferOutput


def _split_pair(text: str) -> Tuple[str, str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "premise" in obj and "hypothesis" in obj:
            return str(obj["premise"]), str(obj["hypothesis"])
    except (ValueError, TypeError):
        pass
    if "[SEP]" in text:
        a, _, b = text.partition("[SEP]")
        return a.strip(), b.strip()
    return text, text


class NliRunner(HeavyRunner):
    runner_kind = "nli"

    def _import_stack(self) -> None:
        self._torch = require(lambda: __import__("torch"), "torch")
        require(lambda: __import__("sentence_transformers"), "sentence-transformers")

    def _build(self, path: str) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(path, local_files_only=True)
        # entailment label order varies; default to MNLI [contradiction, neutral, entailment]
        self._labels = self.spec.get("nli_labels") or ["contradiction", "neutral", "entailment"]

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        pairs = [_split_pair(t) for t in texts]
        scores = self._model.predict(pairs, apply_softmax=True)
        outs = []
        for row in scores:
            row = list(row) if hasattr(row, "__iter__") else [float(row)]
            label_scores = {self._labels[i]: round(float(v), 4)
                            for i, v in enumerate(row) if i < len(self._labels)}
            entail = label_scores.get("entailment", 0.0)
            # unsupported claim = NOT entailed → risk is 1 - entailment
            outs.append(self._output(1.0 - entail, label_scores))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> NliRunner:
    return NliRunner(task, spec)
