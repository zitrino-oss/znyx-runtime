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

# Labels that NEGATE a harm word — "not-toxic", "non_toxic", "nontoxic", "harmless". Checked
# FIRST, because _UNSAFE_LABEL matches on substrings: "not-toxic" contains "toxic" and
# "harmless" contains "harm", so a model whose benign class is named that way would have its
# benign probability read as the harm score, blocking everything. Deliberately excludes a bare
# "un" prefix so "unsafe" (itself the harm label) keeps matching _UNSAFE_LABEL.
_NEGATED_LABEL = re.compile(r"^(?:not|non|no)[\W_]*(?:toxic|harm|offensive|abusive|hate|inject)"
                            r"|harmless", re.IGNORECASE)


class ClassifierRunner(OnnxTextRunner):
    runner_kind = "classifier"

    # Set from the artifact's config.json in _build_extra. Default False = softmax, matching
    # HuggingFace's own default when problem_type is absent.
    _multi_label = False

    def _build_extra(self, cfg: Dict[str, Any]) -> None:
        """Read the head type so infer_batch can pick the right activation."""
        self._multi_label = str(cfg.get("problem_type") or "").lower() == "multi_label_classification"

    def _unsafe_prob(self, probs: List[float]) -> Tuple[float, Dict[str, float]]:
        label_scores: Dict[str, float] = {}
        unsafe = 0.0
        for idx, p in enumerate(probs):
            name = str(self._id2label.get(idx, f"LABEL_{idx}"))
            label_scores[name] = round(float(p), 4)
            if _NEGATED_LABEL.search(name):
                continue                      # benign class — never the harm score
            if _UNSAFE_LABEL.search(name) or (not self._id2label and idx > 0):
                unsafe = max(unsafe, float(p))
        return unsafe, label_scores

    def infer_batch(self, texts: List[str], params: dict | None = None) -> List[InferOutput]:
        logits, _ = self._forward(list(texts))       # [B, L]
        # Activation follows the head type. Multi-label heads (e.g. unitary/toxic-bert, whose
        # 6 Jigsaw labels are independent) need per-label sigmoid; softmax would normalise
        # them to sum to 1 and report shares instead of probabilities.
        probs = self._sigmoid(logits) if self._multi_label else self._softmax(logits, axis=-1)
        outs = []
        for row in probs.tolist():
            unsafe, label_scores = self._unsafe_prob(row)
            outs.append(self._output(unsafe, label_scores))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> ClassifierRunner:
    return ClassifierRunner(task, spec)
