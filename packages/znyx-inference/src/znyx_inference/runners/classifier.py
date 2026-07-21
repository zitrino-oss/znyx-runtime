"""Sequence-classification runner — e.g. Prompt-Guard / DeBERTa for prompt-injection
/ toxicity / jailbreak. Served on CPU via onnxruntime + tokenizers (torch-free); the ONNX
graph + tokenizer load only in ``load()`` from the verified local artifact dir, never the
network.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from znyx_inference.runners._heavy import OnnxTextRunner
from znyx_inference.runners.base import InferOutput

# A label is "unsafe" if its name signals harm (else we treat index>0 as the positive class).
_UNSAFE_LABEL = re.compile(r"unsafe|inject|jailbreak|toxic|harm|attack|label_?1|positive", re.IGNORECASE)


class ClassifierRunner(OnnxTextRunner):
    runner_kind = "classifier"

    def _unsafe_prob(self, probs: List[float]) -> Tuple[float, Dict[str, float]]:
        label_scores: Dict[str, float] = {}
        unsafe = 0.0
        for idx, p in enumerate(probs):
            name = str(self._id2label.get(idx, f"LABEL_{idx}"))
            label_scores[name] = round(float(p), 4)
            if _UNSAFE_LABEL.search(name) or (not self._id2label and idx > 0):
                unsafe = max(unsafe, float(p))
        return unsafe, label_scores

    def infer_batch(self, texts: List[str], params: dict | None = None) -> List[InferOutput]:
        logits, _ = self._forward(list(texts))       # [B, L]
        probs = self._softmax(logits, axis=-1)
        outs = []
        for row in probs.tolist():
            unsafe, label_scores = self._unsafe_prob(row)
            outs.append(self._output(unsafe, label_scores))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> ClassifierRunner:
    return ClassifierRunner(task, spec)
