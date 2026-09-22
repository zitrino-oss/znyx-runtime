"""Language-aware runner — closes the gap where the generic ClassifierRunner scored
0 for language-ID labels (language identification is multi-class, not binary harm, so the
"unsafe-prob" heuristic never fired). Loads a language-identification sequence-classification
model (e.g. XLM-R language-detection) and maps the predicted language to an allow/block
decision using ``allowed_languages`` / ``blocked_languages`` from the runner spec. Served
on CPU via onnxruntime + tokenizers (torch-free); the ONNX graph loads only in ``load()``.

risk = the predicted language's probability WHEN that language is blocked (or outside the
allowed set); else 0. ``label_scores`` carries the full language distribution, so the
detected language is always visible regardless of the decision. With no allow/block lists
configured the runner is a pure language identifier (never blocks).
"""
from __future__ import annotations

from typing import Any, Dict, List

from znyx_inference.runners._heavy import OnnxTextRunner
from znyx_inference.runners.base import InferOutput


class LanguageRunner(OnnxTextRunner):
    runner_kind = "language"

    def __init__(self, task: str, spec: Dict[str, Any]):
        super().__init__(task, spec)
        # Mirror the deterministic LanguageDetector policy (shared/detectors/language.py):
        # blocked wins; an allowed-list (when set) blocks anything outside it; "unknown"
        # is exempt from the allowed-list check.
        self._allowed = {str(s).lower() for s in (spec.get("allowed_languages") or [])}
        self._blocked = {str(s).lower() for s in (spec.get("blocked_languages") or [])}

    def _decide(self, lang_probs: Dict[str, float],
                 allowed: set | None = None, blocked: set | None = None) -> InferOutput:
        """Pure: given a language→probability distribution, apply the allow/block policy.
        risk = the top language's prob when blocked/disallowed, else 0; ``label_scores`` =
        the distribution (so the detected language is always reported).

        ``allowed``/``blocked`` override the static spec when provided (per-request params
        from the policy), falling back to ``self._allowed``/``self._blocked``."""
        if not lang_probs:
            return self._output(0.0, None)
        eff_allowed = allowed if allowed is not None else self._allowed
        eff_blocked = blocked if blocked is not None else self._blocked
        unsafe = 0.0
        if eff_blocked:
            # Blocklist: the probability mass sitting ON blocked languages. Summed rather
            # than top-only, so two blocked languages splitting the mass still add up.
            unsafe = sum(p for lang, p in lang_probs.items()
                         if lang.lower() in eff_blocked)
        elif eff_allowed:
            # Allowlist: the mass OUTSIDE the allowed set — deliberately NOT the top label's
            # probability. An allowlist asks "is this an allowed language?", and deciding on
            # the top label answers a different question ("which banned language is it?"),
            # which fails OPEN for any language absent from the model's label set: the model
            # must spread its mass over labels it knows, so no single one clears the
            # threshold. Danish scores top `tr`=0.22 — under a 0.5 threshold that ALLOWS
            # Danish through an English-only policy. Mass-based, the same input reads
            # 1 - P(en) = 1 - 0.058 = 0.94 and is correctly blocked. The model never has to
            # name the language; being confident it is not English is enough.
            #
            # NOTE: unlike the previous top-label rule, an "unknown"/"other" label is no
            # longer exempt — its mass is outside the allowed set, so it counts as risk.
            # For an allowlist that is the correct reading: "I cannot tell that this is
            # English" must not mean "allow".
            allowed_mass = sum(p for lang, p in lang_probs.items()
                               if lang.lower() in eff_allowed)
            unsafe = max(0.0, 1.0 - allowed_mass)
        return self._output(unsafe, lang_probs)

    def infer_batch(self, texts: List[str], params: dict | None = None) -> List[InferOutput]:
        logits, _ = self._forward(list(texts))       # [B, L]
        probs = self._softmax(logits, axis=-1)
        # Per-request params from the policy override the static spec.
        allowed = blocked = None
        if params:
            al = params.get("allowed_languages")
            if al is not None:
                allowed = {str(s).lower() for s in al}
            bl = params.get("blocked_languages")
            if bl is not None:
                blocked = {str(s).lower() for s in bl}
        outs: List[InferOutput] = []
        for row in probs.tolist():
            lang_probs = {str(self._id2label.get(i, f"LABEL_{i}")).lower(): float(p)
                          for i, p in enumerate(row)}
            outs.append(self._decide(lang_probs, allowed, blocked))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> LanguageRunner:
    return LanguageRunner(task, spec)
