"""Transformer sequence-classification runner (F3) — e.g. Prompt-Guard / DeBERTa for
prompt-injection / toxicity / jailbreak. Heavy deps (torch + transformers) are imported
only in ``load()``; weights load from the verified local artifact dir, never the network.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from znyx_inference.runners._heavy import HeavyRunner, require
from znyx_inference.runners.base import InferOutput

# A label is "unsafe" if its name signals harm (else we treat index>0 as the positive class).
_UNSAFE_LABEL = re.compile(r"unsafe|inject|jailbreak|toxic|harm|attack|label_?1|positive", re.IGNORECASE)


class ClassifierRunner(HeavyRunner):
    runner_kind = "classifier"

    def _import_stack(self) -> None:
        self._torch = require(lambda: __import__("torch"), "torch")
        require(lambda: __import__("transformers"), "transformers")

    def _build(self, path: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self._model.eval()
        self._id2label = dict(getattr(self._model.config, "id2label", {}) or {})

    def _unsafe_prob(self, probs: List[float]) -> tuple[float, Dict[str, float]]:
        label_scores: Dict[str, float] = {}
        unsafe = 0.0
        for idx, p in enumerate(probs):
            name = str(self._id2label.get(idx, f"LABEL_{idx}"))
            label_scores[name] = round(float(p), 4)
            if _UNSAFE_LABEL.search(name) or (not self._id2label and idx > 0):
                unsafe = max(unsafe, float(p))
        return unsafe, label_scores

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        torch = self._torch
        enc = self._tok(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).tolist()
        outs = []
        for row in probs:
            unsafe, label_scores = self._unsafe_prob(row)
            outs.append(self._output(unsafe, label_scores))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> ClassifierRunner:
    return ClassifierRunner(task, spec)
