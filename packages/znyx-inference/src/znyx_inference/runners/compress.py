"""Prompt compression runner (gateway technique 13): a token-classification model that
scores, per token, whether it is worth keeping (LLMLingua-2 style: label 1 = keep). The
runner keeps the highest-scoring share of tokens (``params.rate``, default 0.5) in their
original order and returns them as character spans of the text sent (``entity_spans``,
label ``KEEP``); the gateway joins the spans. Served on CPU via onnxruntime + tokenizers,
like the NER runner; long texts are read in chunks the model can take.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from znyx_inference.runners._heavy import OnnxTextRunner
from znyx_inference.runners.base import InferOutput

CHUNK_CHARS = 1500
_KEEP_LABELS = ("1", "label_1", "keep", "preserve", "positive")


def keep_spans(tokens: List[Tuple[float, Tuple[int, int]]], rate: float) -> List[Tuple[int, int, str]]:
    """Pure: the character spans of the ``rate`` share of tokens most worth keeping, in
    text order, with neighbours (gap of at most one character) merged."""
    usable = [(p, (s, e)) for p, (s, e) in tokens if e > s]
    if not usable:
        return []
    count = max(1, round(len(usable) * rate))
    chosen = sorted(sorted(usable, key=lambda t: -t[0])[:count], key=lambda t: t[1][0])
    spans: List[Tuple[int, int, str]] = []
    for _p, (s, e) in chosen:
        if spans and s <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e), "KEEP")
        else:
            spans.append((s, e, "KEEP"))
    return spans


def chunks(text: str, size: int = CHUNK_CHARS) -> List[Tuple[int, str]]:
    """Pure: (offset, piece) pieces of at most ``size`` characters, cut at whitespace where
    possible so a word is not split."""
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        out.append((start, text[start:end]))
        start = end
    return out


class CompressRunner(OnnxTextRunner):
    runner_kind = "compress"

    def _keep_index(self) -> int:
        for idx, label in getattr(self, "_id2label", {}).items():
            if str(label).strip().lower() in _KEEP_LABELS:
                return int(idx)
        return 1

    def infer_batch(self, texts: List[str], params: Optional[Dict[str, Any]] = None) -> List[InferOutput]:
        rate = float((params or {}).get("rate", 0.5))
        rate = min(0.9, max(0.1, rate))
        keep = self._keep_index()
        outs: List[InferOutput] = []
        for text in texts:
            spans: List[Tuple[int, int, str]] = []
            for offset, piece in chunks(text):
                logits, encs = self._forward([piece])
                probs = self._softmax(logits, axis=-1)
                enc = encs[0]
                offsets = getattr(enc, "offsets", None) or []
                mask = enc.attention_mask
                tokens = []
                for t in range(min(len(offsets), probs.shape[1])):
                    if t < len(mask) and mask[t] == 0:
                        continue
                    s, e = int(offsets[t][0]), int(offsets[t][1])
                    tokens.append((float(probs[0][t][keep]), (offset + s, offset + e)))
                spans.extend(keep_spans(tokens, rate))
            outs.append(self._output(0.0, None, entity_spans=spans or None))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> CompressRunner:
    return CompressRunner(task, spec)
