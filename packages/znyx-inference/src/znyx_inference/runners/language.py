"""Language-aware runner (F3) — closes the gap where the generic ClassifierRunner scored
0 for language-ID labels (language identification is multi-class, not binary harm, so the
"unsafe-prob" heuristic never fired). Loads a language-identification sequence-classification
model (e.g. XLM-R language-detection) and maps the predicted language to an allow/block
decision using ``allowed_languages`` / ``blocked_languages`` from the runner spec. Heavy
deps import only in ``load()``.

risk = the predicted language's probability WHEN that language is blocked (or outside the
allowed set); else 0. ``label_scores`` carries the full language distribution, so the
detected language is always visible regardless of the decision. With no allow/block lists
configured the runner is a pure language identifier (never blocks).
"""
from __future__ import annotations

from typing import Any, Dict, List

from znyx_inference.runners._heavy import HeavyRunner, require
from znyx_inference.runners.base import InferOutput


class LanguageRunner(HeavyRunner):
    runner_kind = "language"

    def __init__(self, task: str, spec: Dict[str, Any]):
        super().__init__(task, spec)
        # Mirror the deterministic LanguageDetector policy (shared/detectors/language.py):
        # blocked wins; an allowed-list (when set) blocks anything outside it; "unknown"
        # is exempt from the allowed-list check.
        self._allowed = {str(s).lower() for s in (spec.get("allowed_languages") or [])}
        self._blocked = {str(s).lower() for s in (spec.get("blocked_languages") or [])}

    def _import_stack(self) -> None:
        self._torch = require(lambda: __import__("torch"), "torch")
        require(lambda: __import__("transformers"), "transformers")

    def _build(self, path: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self._model.eval()
        self._id2label = dict(getattr(self._model.config, "id2label", {}) or {})

    def _decide(self, lang_probs: Dict[str, float]) -> InferOutput:
        """Pure: given a language→probability distribution, apply the allow/block policy.
        risk = the top language's prob when blocked/disallowed, else 0; ``label_scores`` =
        the distribution (so the detected language is always reported)."""
        if not lang_probs:
            return self._output(0.0, None)
        top_lang, top_p = max(lang_probs.items(), key=lambda kv: kv[1])
        tl = top_lang.lower()
        unsafe = 0.0
        if self._blocked and tl in self._blocked:
            unsafe = top_p
        elif self._allowed and tl != "unknown" and tl not in self._allowed:
            unsafe = top_p
        return self._output(unsafe, lang_probs)

    def infer_batch(self, texts: List[str]) -> List[InferOutput]:
        torch = self._torch
        enc = self._tok(list(texts), padding=True, truncation=True, max_length=512,
                        return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self._model(**enc).logits, dim=-1).tolist()
        outs: List[InferOutput] = []
        for row in probs:
            lang_probs = {str(self._id2label.get(i, f"LABEL_{i}")).lower(): float(p)
                          for i, p in enumerate(row)}
            outs.append(self._decide(lang_probs))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> LanguageRunner:
    return LanguageRunner(task, spec)
