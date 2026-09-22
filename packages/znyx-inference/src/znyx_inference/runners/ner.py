"""Token-classification (NER) runner — detects UNSTRUCTURED PII entities (names,
addresses, etc.) that the deterministic regex/checksum PII detector can't catch. Backed
by a token-classification model (the catalog default is Davlan multilingual NER; any
BIO-labelled token-classification head works). Served on CPU via onnxruntime + tokenizers
(torch-free); the ONNX graph loads only in ``load()`` from the verified local artifact dir,
never the network.

risk = the max probability assigned to any PII (non-"outside") entity token; ``label_scores``
carries the per-entity-type max confidence so the caller sees WHICH PII types were found.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from znyx_inference.runners._heavy import OnnxTextRunner
from znyx_inference.runners.base import InferOutput

# Labels meaning "not a PII entity" (outside), compared case-insensitively AFTER stripping
# any BIO/BILOU prefix (B-/I-/L-/U-/E-). piiranha & most NER heads use "O" for outside.
_OUTSIDE_LABELS = {"o", "outside", "none", "label_0", ""}
_BIO_PREFIX = re.compile(r"^[biloue][-_](.+)$", re.IGNORECASE)


class NerRunner(OnnxTextRunner):
    runner_kind = "ner"

    def _entity_type(self, raw_label: str) -> Optional[str]:
        """PII entity type for a token label, or None if it's the outside/non-entity label.
        Strips a BIO/BILOU prefix (e.g. ``I-SURNAME`` → ``SURNAME``). Pure (no model)."""
        name = (raw_label or "").strip()
        if name.lower() in _OUTSIDE_LABELS:
            return None
        m = _BIO_PREFIX.match(name)
        ent = (m.group(1) if m else name).strip()
        return None if ent.lower() in _OUTSIDE_LABELS else ent.upper()

    def _merge_spans(self, tokens: List[Tuple[str, float, Tuple[int, int]]]
                     ) -> List[Tuple[int, int, str]]:
        """Pure: collapse per-token (label, prob, char-offsets) into entity CHARACTER spans.

        A model labels sub-word tokens, so one name spans several of them ("Sarah" ->
        ``B-PER``, "Chen" -> ``I-PER``, and a word may itself split into ``##`` pieces).
        Consecutive tokens of the same entity type are joined into a single span, and a
        ``B-`` prefix starts a new one so two adjacent entities of the same type
        ("Sarah Chen, Alan Turing") don't merge into one.

        Offsets come straight from the tokenizer and are relative to the ORIGINAL text.
        Special tokens ([CLS]/[SEP]) report (0, 0) and are skipped — a zero-width span would
        otherwise redact at position 0.
        """
        spans: List[Tuple[int, int, str]] = []
        prev_ent: Optional[str] = None
        for raw_label, _prob, (start, end) in tokens:
            ent = self._entity_type(raw_label)
            if ent is None or end <= start:
                prev_ent = None                    # outside token breaks the run
                continue
            starts_entity = bool(re.match(r"^[bu][-_]", (raw_label or "").strip(), re.I))
            if spans and ent == prev_ent and not starts_entity and start <= spans[-1][1] + 1:
                # continuation: widen the open span (allow one char of whitespace between
                # sub-tokens, which the tokenizer excludes from either offset)
                s0, _e0, l0 = spans[-1]
                spans[-1] = (s0, end, l0)
            else:
                spans.append((start, end, ent))
            prev_ent = ent
        return spans

    def _aggregate(self, token_top: List[Tuple[str, float]],
                   spans: Optional[List[Tuple[int, int, str]]] = None) -> InferOutput:
        """Pure: collapse per-token (top-label, prob) pairs into a risk + per-type scores.
        risk = max prob over PII tokens; ``label_scores`` = per-entity-type max prob.
        ``spans`` (when the caller has offsets) travels through so a REDACT-action detector
        can replace the entities rather than only learn that some exist."""
        unsafe = 0.0
        types: Dict[str, float] = {}
        for label, prob in token_top:
            ent = self._entity_type(label)
            if ent is None:
                continue
            p = float(prob)
            unsafe = max(unsafe, p)
            types[ent] = max(types.get(ent, 0.0), p)
        return self._output(unsafe, types or None, entity_spans=spans)

    def infer_batch(self, texts: List[str], params: dict | None = None) -> List[InferOutput]:
        np = self._np
        logits, encs = self._forward(list(texts))          # [B, T, L]
        probs = self._softmax(logits, axis=-1)
        outs: List[InferOutput] = []
        for b in range(probs.shape[0]):
            mask = encs[b].attention_mask
            # Character offsets per token. Present on every `tokenizers` Encoding, but
            # guarded so a tokenizer build without them degrades to score-only output
            # instead of failing the whole inference.
            offsets = getattr(encs[b], "offsets", None)
            token_top: List[Tuple[str, float]] = []
            with_offsets: List[Tuple[str, float, Tuple[int, int]]] = []
            for t in range(probs.shape[1]):
                if t < len(mask) and mask[t] == 0:
                    continue                       # skip padding
                row = probs[b][t]
                idx = int(np.argmax(row))
                label, prob = str(self._id2label.get(idx, f"LABEL_{idx}")), float(row[idx])
                token_top.append((label, prob))
                if offsets is not None and t < len(offsets):
                    off = offsets[t]
                    with_offsets.append((label, prob, (int(off[0]), int(off[1]))))
            spans = self._merge_spans(with_offsets) if with_offsets else None
            outs.append(self._aggregate(token_top, spans or None))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> NerRunner:
    return NerRunner(task, spec)
