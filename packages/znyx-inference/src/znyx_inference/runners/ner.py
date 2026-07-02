"""Token-classification (NER) runner (F3) — detects UNSTRUCTURED PII entities (names,
addresses, etc.) that the deterministic regex/checksum PII detector can't catch. Backed
by a token-classification model (e.g. piiranha). Heavy deps (torch + transformers) import
only in ``load()``; weights load from the verified local artifact dir, never the network.

risk = the max probability assigned to any PII (non-"outside") entity token; ``label_scores``
carries the per-entity-type max confidence so the caller sees WHICH PII types were found.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from znyx_inference.runners._heavy import HeavyRunner, require
from znyx_inference.runners.base import InferOutput

# Labels meaning "not a PII entity" (outside), compared case-insensitively AFTER stripping
# any BIO/BILOU prefix (B-/I-/L-/U-/E-). piiranha & most NER heads use "O" for outside.
_OUTSIDE_LABELS = {"o", "outside", "none", "label_0", ""}
_BIO_PREFIX = re.compile(r"^[biloue][-_](.+)$", re.IGNORECASE)


class NerRunner(HeavyRunner):
    runner_kind = "ner"

    def _import_stack(self) -> None:
        self._torch = require(lambda: __import__("torch"), "torch")
        require(lambda: __import__("transformers"), "transformers")

    def _build(self, path: str) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)
        self._model.eval()
        self._id2label = dict(getattr(self._model.config, "id2label", {}) or {})

    def _entity_type(self, raw_label: str) -> Optional[str]:
        """PII entity type for a token label, or None if it's the outside/non-entity label.
        Strips a BIO/BILOU prefix (e.g. ``I-SURNAME`` → ``SURNAME``). Pure (no model)."""
        name = (raw_label or "").strip()
        if name.lower() in _OUTSIDE_LABELS:
            return None
        m = _BIO_PREFIX.match(name)
        ent = (m.group(1) if m else name).strip()
        return None if ent.lower() in _OUTSIDE_LABELS else ent.upper()

    def _aggregate(self, token_top: List[Tuple[str, float]]) -> InferOutput:
        """Pure: collapse per-token (top-label, prob) pairs into a risk + per-type scores.
        risk = max prob over PII tokens; ``label_scores`` = per-entity-type max prob."""
        unsafe = 0.0
        types: Dict[str, float] = {}
        for label, prob in token_top:
            ent = self._entity_type(label)
            if ent is None:
                continue
            p = float(prob)
            unsafe = max(unsafe, p)
            types[ent] = max(types.get(ent, 0.0), p)
        return self._output(unsafe, types or None)

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        torch = self._torch
        enc = self._tok(list(texts), padding=True, truncation=True, max_length=512,
                        return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self._model(**enc).logits, dim=-1)   # [B, T, L]
        mask = enc.get("attention_mask")
        outs: List[InferOutput] = []
        for b in range(probs.shape[0]):
            token_top: List[Tuple[str, float]] = []
            for t in range(probs.shape[1]):
                if mask is not None and int(mask[b][t]) == 0:
                    continue                       # skip padding
                row = probs[b][t]
                idx = int(torch.argmax(row))
                token_top.append((str(self._id2label.get(idx, f"LABEL_{idx}")), float(row[idx])))
            outs.append(self._aggregate(token_top))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> NerRunner:
    return NerRunner(task, spec)
