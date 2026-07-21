"""NLI cross-encoder runner — entailment between a premise (provided source) and a
hypothesis (a claim/quote), for grounding checks (e.g. citation_integrity NLI-backed).
The input text is a JSON object ``{"premise": ..., "hypothesis": ...}`` (or
``premise [SEP] hypothesis``); risk = contradiction/non-entailment probability. Served on
CPU via onnxruntime + tokenizers (torch-free); the ONNX graph loads only in ``load()``. The
premise/hypothesis pair is tokenized as a sentence pair, so the cross-encoder gets its two
segments (and token_type_ids) exactly as at training time.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from znyx_inference.runners._heavy import OnnxTextRunner
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


class NliRunner(OnnxTextRunner):
    runner_kind = "nli"

    def _build_extra(self, cfg: Dict[str, Any]) -> None:
        # entailment label order varies; default to MNLI [contradiction, neutral, entailment].
        # Prefer the spec, then the model's own id2label, then the MNLI default.
        by_config = [self._id2label[i] for i in sorted(self._id2label)] if self._id2label else []
        self._labels = (self.spec.get("nli_labels") or by_config
                        or ["contradiction", "neutral", "entailment"])

    def infer_batch(self, texts: List[str], params: dict | None = None) -> List[InferOutput]:
        pairs = [_split_pair(t) for t in texts]
        logits, _ = self._forward(pairs)                    # [B, L] (pair-encoded)
        # multi-class NLI → softmax; a single-logit regression head → sigmoid.
        if logits.ndim == 2 and logits.shape[1] > 1:
            scores = self._softmax(logits, axis=-1)
        else:
            scores = 1.0 / (1.0 + self._np.exp(-logits.reshape(logits.shape[0], -1)))
        outs = []
        for row in scores.tolist():
            label_scores = {str(self._labels[i]): round(float(v), 4)
                            for i, v in enumerate(row) if i < len(self._labels)}
            # case-insensitive: models label this "entailment" / "ENTAILMENT" / "Entailment".
            entail = next((v for k, v in label_scores.items() if k.lower() == "entailment"), 0.0)
            # unsupported claim = NOT entailed → risk is 1 - entailment
            outs.append(self._output(1.0 - entail, label_scores))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> NliRunner:
    return NliRunner(task, spec)
