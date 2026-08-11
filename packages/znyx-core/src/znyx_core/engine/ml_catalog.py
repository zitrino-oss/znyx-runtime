"""Canonical ML model catalog — the single source of truth for the model-
backed inference tasks: which heavy runner serves each, the pinned model_id/revision,
provenance (license / language / hardware), and the recommended deterministic→ML
escalation defaults per detector.

Shared by THREE consumers so they can't drift:
  * the inference service (``packages/znyx-inference/src/znyx_inference/config.py``) — task→runner+pins (heavy profile)
  * the model-registry seed (control plane) — provenance rows
  * the default-strategy builder (console/API) — a gate-shaped strategy block per detector

Dependency-free (stdlib only): the inference service is a separate, dependency-minimal
deployable and imports this module, so it must never pull heavy deps.

IMPORTANT: pins are canonical *public* model ids at a fixed revision; ``sha256`` is left
unset on purpose — an operator pins the digest of THEIR downloaded artifact and the heavy
runner verifies it (no implicit network downloads). A model is ``available=False`` in the
registry until an operator actually loads its weights, and ships with an honest
``unverified`` scorecard — so a policy/pack that adopts a default strategy still can't
publish a BLOCK (or even publish at all) until a real benchmark lands a passing scorecard.
That gate is by design.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Runner kinds the inference service can actually serve today (registry RUNNER_FACTORIES
# + _HEAVY_KIND_MODULES). "ner" serves token-level PII NER (unstructured PII); "language"
# serves language-ID + allow/block mapping.
SERVABLE_RUNNERS = {"stub", "classifier", "embedding", "nli", "guard_llm", "ner", "language"}


@dataclass(frozen=True)
class MLTaskSpec:
    task: str
    runner: str                          # classifier | embedding | nli | guard_llm | ner(planned)
    model_id: str
    revision: str = "main"
    threshold: float = 0.5
    language_coverage: Tuple[str, ...] = ("en",)
    license: Optional[str] = None
    hardware_req: Optional[str] = None
    supported: bool = True               # False → no runner serves it yet
    detail: Optional[str] = None

    @property
    def model_version(self) -> str:
        return f"{self.model_id}@{self.revision}"


# The 7 tasks — ALL now map to a real servable runner (pii_ner→ner and
# language→language landed; the rest use classifier/nli/embedding/guard_llm).
ML_TASKS: Dict[str, MLTaskSpec] = {
    "prompt_injection": MLTaskSpec(
        "prompt_injection", "classifier",
        "protectai/deberta-v3-base-prompt-injection-v2", "main", 0.5,
        ("en",), "apache-2.0", "cpu-ok; ~440MB"),
    "toxicity": MLTaskSpec(
        "toxicity", "classifier", "unitary/toxic-bert", "main", 0.5,
        ("en",), "apache-2.0", "cpu-ok; ~440MB"),
    "language": MLTaskSpec(
        "language", "language", "papluca/xlm-roberta-base-language-detection", "main", 0.5,
        ("en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ar", "hi"),
        "mit", "cpu-ok; ~1.1GB",
        detail="LanguageRunner predicts the language and applies the detector's "
               "allowed_languages / blocked_languages (from the runner spec) to an "
               "allow/block decision — fixes the generic-classifier-scores-0 gap."),
    "nli": MLTaskSpec(
        "nli", "nli", "cross-encoder/nli-deberta-v3-base", "main", 0.5,
        ("en",), "apache-2.0", "cpu-ok; ~440MB"),
    "safety": MLTaskSpec(
        "safety", "guard_llm", "meta-llama/Llama-Guard-3-1B", "main", 0.5,
        ("en",), "llama3.2-community", "gpu-recommended; ~2.5GB"),
    "topic_intent": MLTaskSpec(
        "topic_intent", "embedding", "sentence-transformers/all-MiniLM-L6-v2", "main", 0.5,
        ("en",), "apache-2.0", "cpu-ok; ~90MB"),
    # Default is the model that actually serves on the NER runner today. The previous pin
    # (iiiorg/piiranha-v1-detect-personal-information) is PII-specific but is not verified
    # through the AutoTokenizer NER path, so a pii_ner task pinned to it reports
    # unavailable; the Davlan multilingual NER checkpoint loads and serves. Keep this in
    # sync with the CANDIDATE_MODELS "primary" row for pii_ner — the console's
    # Servable-tasks "Catalog default" column reads this spec.
    "pii_ner": MLTaskSpec(
        "pii_ner", "ner", "Davlan/bert-base-multilingual-cased-ner-hrl", "main", 0.5,
        ("ar", "de", "en", "es", "fr", "it", "lv", "nl", "pt", "zh"),
        "apache-2.0", "cpu-ok; ~700MB",
        detail="NerRunner does token-level classification to catch UNSTRUCTURED PII "
               "(names/orgs/locations) the deterministic regex/checksum PII detector "
               "misses. Structured PII (cards, SSNs, emails) stays with the deterministic "
               "layer — the ML layer here is additive, not a replacement."),
}


# ── Candidate-model shortlist ────────────────────────────
# Open-license checkpoints per task so an operator can CHOOSE which model to pin.
# Accuracy figures are vendor/benchmark-reported (NOT measured on ZNYX data) and
# every row must clear the scorecard_gate on our own suites before enforcement.
# `open_license` = OSI-permissive / OpenRAIL++ / CC-BY-SA (commercial self-host OK);
# Llama-Community rows are listed for choice but flagged (commercial <700M MAU, AUP).

@dataclass(frozen=True)
class CandidateModel:
    task: str
    runner: str
    model_id: str
    role: str                       # "primary" | "alternative"
    license: str
    open_license: bool              # OSI/OpenRAIL++/CC-BY-SA → True; Llama-Community → False
    commercial: str                 # "yes" | "conditional"
    size: str
    accuracy_reported: str
    hardware_req: str
    revision: str = "main"
    note: Optional[str] = None


CANDIDATE_MODELS: Dict[str, List[CandidateModel]] = {
    "prompt_injection": [
        CandidateModel("prompt_injection", "classifier", "protectai/deberta-v3-base-prompt-injection-v2", "primary", "apache-2.0", True, "yes", "184M", "~99.9% on its own eval set", "CPU ~50–100 ms"),
        # CandidateModel("prompt_injection", "classifier", "protectai/deberta-v3-small-prompt-injection-v2", "alternative", "apache-2.0", True, "yes", "96M", "~94% on unseen data", "CPU ~20–30 ms"),  # gated on HuggingFace
        # CandidateModel("prompt_injection", "classifier", "meta-llama/Llama-Prompt-Guard-2-86M", "alternative", "llama-community", False, "conditional", "86M", "AUC ~0.998 (EN)", "CPU-ok", note="Llama Community: commercial <700M MAU, carries an AUP — not OSI-open."),  # gated
    ],
    "toxicity": [
        CandidateModel("toxicity", "classifier", "unitary/toxic-bert", "primary", "apache-2.0", True, "yes", "110M", "Jigsaw Toxic Comment dataset (EN)", "CPU ~440MB"),
        CandidateModel("toxicity", "classifier", "textdetox/xlmr-large-toxicity-classifier-v2", "alternative", "openrail++", True, "yes", "0.6B", "F1 0.56–0.97 across 15 langs", "GPU pref / CPU-ok"),
        # CandidateModel("toxicity", "classifier", "unitary/multilingual-toxic-xlm-roberta", "alternative", "apache-2.0", True, "yes", "XLM-R-base", "Jigsaw-trained, 7 langs", "CPU-ok"),  # no tokenizer.json on the Hub — install fails, see below
        # ^ Ships only sentencepiece.bpe.model, no prebuilt tokenizer.json, so _export_onnx's
        # AutoTokenizer call has to BUILD the fast tokenizer from the SPM model — which needs
        # the sentencepiece + protobuf packages the [export] extra does not carry. The install
        # job therefore fails for this row alone. Every other multilingual toxicity row above
        # ships tokenizer.json (textdetox/xlmr-large-toxicity-classifier-v2 is also XLM-R and
        # installs fine), so nothing here is lost by dropping it. Re-enable only together with
        # sentencepiece + protobuf in the [export] extra of znyx-inference/pyproject.toml.
        CandidateModel("toxicity", "classifier", "textdetox/bert-multilingual-toxicity-classifier", "alternative", "openrail++", True, "yes", "BERT-base", "multilingual", "CPU-ok"),
        CandidateModel("toxicity", "classifier", "gravitee-io/distilbert-multilingual-toxicity-classifier", "alternative", "openrail++", True, "yes", "DistilBERT", "fastest", "CPU fast"),
    ],
    # "safety" — guard_llm runner needs [torch] extra, not available in lean ONNX image
    # "safety": [
    #     CandidateModel("safety", "guard_llm", "allenai/wildguard", "primary", "apache-2.0", True, "yes", "7B", "open safety classifier", "GPU recommended"),
    #     CandidateModel("safety", "guard_llm", "ibm-granite/granite-guardian-3.0-2b", "alternative", "apache-2.0", True, "yes", "2B", "small guard options", "CPU-ok / GPU"),
    #     CandidateModel("safety", "guard_llm", "meta-llama/Llama-Guard-3-8B", "alternative", "llama-community", False, "conditional", "8B", "strong, gated", "GPU", note="Llama Community: commercial <700M MAU, AUP."),
    # ],
    "language": [
        CandidateModel("language", "language", "papluca/xlm-roberta-base-language-detection", "primary", "mit", True, "yes", "XLM-R-base", "20 languages", "CPU-ok"),
        # CandidateModel("language", "language", "facebook/fasttext-language-identification", "alternative", "cc-by-sa-3.0", True, "yes", "fastText", "176 languages (max coverage)", "CPU fast", note="fastText format — needs a fastText loader, not the transformers LanguageRunner."),  # fastText format incompatible
    ],
    "nli": [
        CandidateModel("nli", "nli", "cross-encoder/nli-deberta-v3-large", "primary", "apache-2.0", True, "yes", "DeBERTa-v3-large", "strong NLI", "GPU pref / CPU-ok"),
        CandidateModel("nli", "nli", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "alternative", "mit", True, "yes", "DeBERTa-v3-large", "adversarial-robust (ANLI)", "GPU pref"),
        CandidateModel("nli", "nli", "cross-encoder/nli-deberta-v3-base", "alternative", "apache-2.0", True, "yes", "DeBERTa-v3-base", "lighter", "CPU-ok"),
        CandidateModel("nli", "nli", "vectara/hallucination_evaluation_model", "alternative", "apache-2.0", True, "yes", "purpose-built", "hallucination-specific", "CPU-ok"),
    ],
    "pii_ner": [
        # CandidateModel("pii_ner", "ner", "urchade/gliner_multi_pii-v1", "primary", "apache-2.0", True, "yes", "GLiNER", "multilingual PII NER", "CPU-ok"),  # GLiNER format incompatible with ONNX export
        CandidateModel("pii_ner", "ner", "Davlan/bert-base-multilingual-cased-ner-hrl", "primary", "apache-2.0", True, "yes", "BERT-base", "multilingual basic NER", "CPU-ok"),
        # CandidateModel("pii_ner", "ner", "microsoft/presidio", "alternative", "mit", True, "yes", "framework", "rules+NER framework", "CPU", note="Framework (not a single HF checkpoint) — analyzer + recognizers."),  # framework, not a model
    ],
    # "topic_intent" — embedding runner needs unsafe_examples config, not supported via UI install
    # "topic_intent": [
    #     CandidateModel("topic_intent", "embedding", "BAAI/bge-m3", "primary", "mit", True, "yes", "568M", "strong multilingual retrieval", "CPU-ok / GPU"),
    #     CandidateModel("topic_intent", "embedding", "intfloat/multilingual-e5-large", "alternative", "mit", True, "yes", "560M", "strong multilingual", "GPU pref"),
    #     CandidateModel("topic_intent", "embedding", "Alibaba-NLP/gte-multilingual-base", "alternative", "apache-2.0", True, "yes", "305M", "efficient", "CPU-ok"),
    #     CandidateModel("topic_intent", "embedding", "Qwen/Qwen3-Embedding-0.6B", "alternative", "apache-2.0", True, "yes", "0.6B", "newer, top MTEB", "GPU pref"),
    # ],
    # "guard_llm" — guard_llm runner needs [torch] extra, not available in lean ONNX image
    # "guard_llm": [
    #     CandidateModel("guard_llm", "guard_llm", "allenai/wildguard", "primary", "apache-2.0", True, "yes", "7B", "permissive guard LLM", "GPU recommended"),
    #     CandidateModel("guard_llm", "guard_llm", "ibm-granite/granite-guardian-3.0-2b", "alternative", "apache-2.0", True, "yes", "2B", "small options", "CPU-ok / GPU"),
    #     CandidateModel("guard_llm", "guard_llm", "meta-llama/Llama-Guard-3-8B", "alternative", "llama-community", False, "conditional", "8B", "strong, gated", "GPU", note="Llama Community: commercial <700M MAU, AUP."),
    # ],
}


def candidate_models(open_only: bool = False) -> List[Dict[str, Any]]:
    """Flat list of candidate models, tagging the currently-pinned
    model per task. `open_only=True` drops the Llama-Community (non-OSI) rows."""
    out: List[Dict[str, Any]] = []
    for task, cands in CANDIDATE_MODELS.items():
        pinned = ML_TASKS[task].model_id if task in ML_TASKS else None
        for c in cands:
            if open_only and not c.open_license:
                continue
            out.append({
                "task": c.task, "runner": c.runner, "model_id": c.model_id, "role": c.role,
                "license": c.license, "open_license": c.open_license, "commercial": c.commercial,
                "size": c.size, "accuracy_reported": c.accuracy_reported,
                "hardware_req": c.hardware_req, "revision": c.revision, "note": c.note,
                "is_current_pin": c.model_id == pinned,
            })
    return out


def inference_task_specs() -> Dict[str, Dict[str, Any]]:
    """Heavy-profile task→spec map for ``InferenceConfig.task_specs``. Prefers the catalog
    default; if its artifacts are missing on disk, falls back to the first installed
    candidate from ``CANDIDATE_MODELS``."""
    base = os.getenv("ZNYX_INFERENCE_ARTIFACTS_DIR") or str(
        Path.home() / ".znyx" / "models"
    )
    specs: Dict[str, Dict[str, Any]] = {}
    for t in ML_TASKS.values():
        if not t.supported or t.runner not in SERVABLE_RUNNERS:
            continue
        model_id = t.model_id
        revision = t.revision
        # If the default model's artifacts aren't on disk, check candidates.
        default_dir = Path(base) / t.model_id.replace("/", "__")
        if not default_dir.is_dir():
            for c in CANDIDATE_MODELS.get(t.task, []):
                cand_dir = Path(base) / c.model_id.replace("/", "__")
                if cand_dir.is_dir():
                    model_id = c.model_id
                    revision = c.revision
                    break
        specs[t.task] = {
            "runner": t.runner,
            "model_id": model_id,
            "revision": revision,
            "threshold": t.threshold,
        }
    return specs


def registry_seed_rows() -> List[Dict[str, Any]]:
    """Provenance rows for the model-registry seed (control plane). ALL tasks (incl.
    ``pii_ner``) for documentation; ``available=False`` (not loaded) + ``unverified``."""
    return [
        {"model_id": t.model_id, "revision": t.revision, "model_version": t.model_version,
         "task": t.task, "runner": t.runner, "license": t.license,
         "language_coverage": list(t.language_coverage), "hardware_req": t.hardware_req,
         "available": False, "validation_status": "unverified", "detail": t.detail}
        for t in ML_TASKS.values()
    ]


@dataclass(frozen=True)
class DetectorMLDefault:
    """Recommended ML upgrade for one deterministic detector.

    Two shapes:
      * ESCALATION (``band`` set): clear-cut lows/highs stay deterministic; only the ambiguous
        ``deterministic_score_between`` middle escalates to the ML layer, which then REPLACES
        the deterministic result. Right for classifier-style tasks (toxicity, jailbreak, topic).
      * ADDITIVE (``additive=True``, ``band=None``): the ML layer always runs and AUGMENTS the
        deterministic result (worst-of) rather than replacing it. Right when the ML layer
        catches what the deterministic layer can't while the deterministic decision must
        survive — pii_ner (unstructured PII on top of regex redaction) and language."""
    detector: str
    task: str
    band: Optional[Tuple[int, int]] = None   # deterministic_score_between [low, high]; None → always-run
    timeout_ms: int = 800
    additive: bool = False


# Keys MUST be canonical detector policy keys (the orchestrator reads strategy config by
# the pipeline key — e.g. "topic_restriction", not "topic"); a key that isn't a real
# pipeline key produces dead config the engine never executes. A guard test asserts this.
DETECTOR_ML_DEFAULTS: Dict[str, DetectorMLDefault] = {
    "jailbreak":        DetectorMLDefault("jailbreak",        "prompt_injection", (35, 70), 800),
    "toxicity":         DetectorMLDefault("toxicity",         "toxicity",         (40, 75), 800),
    "topic_restriction": DetectorMLDefault("topic_restriction", "topic_intent",   (35, 70), 600),
    # Additive ML layers (always-run, worst-of): the deterministic decision is preserved and
    # the ML runner adds what it alone catches. pii_ner = unstructured PII (names/addresses)
    # on top of the regex/checksum PII redaction; language = the language-aware runner.
    "pii":              DetectorMLDefault("pii",      "pii_ner",  additive=True, timeout_ms=800),
    "language":         DetectorMLDefault("language", "language", additive=True, timeout_ms=800),
}


def inference_url() -> str:
    return os.getenv("ZNYX_INFERENCE_URL", "http://localhost:9000").rstrip("/")


def _is_loopback(url: str) -> bool:
    """True if the URL host is loopback (a genuinely co-located sidecar)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}


def available_ml_detectors() -> List[str]:
    """Detector keys with a recommended ML default whose task has a servable runner."""
    return [
        d for d, spec in DETECTOR_ML_DEFAULTS.items()
        if (t := ML_TASKS.get(spec.task)) is not None and t.supported
        and t.runner in SERVABLE_RUNNERS
    ]


def default_strategy_for(
    detector_key: str, *, endpoint_url: Optional[str] = None,
    in_boundary: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Build a gate-shaped ``{strategy, backends}`` config block for a detector's
    recommended deterministic→local_ml escalation, or None when the detector has no
    default / no servable runner.

    ``in_boundary`` is SAFE-BY-DEFAULT: when not given it is inferred from the endpoint
    host — True only for a loopback (genuinely co-located) sidecar, else False so an
    off-box endpoint is treated as a boundary crossing and the egress gate (allowlist
    / residency / redaction / fail-closed audit) applies. This means pointing the builder
    off-box can't silently bypass the gate; pass ``in_boundary`` explicitly to override.

    The result validates against the policy ``StrategyConfig`` / ``DetectorBackendsConfig``
    and IS model-backed (``is_model_backed`` True), pinned to the catalog's model version
    so the scorecard gate checks THAT model. Adopting it in a policy/pack therefore
    requires a passing scorecard to publish (advisory) / BLOCK (enforcement); otherwise
    publish is blocked or the action is pinned to WARN at runtime — by design (gate)."""
    d = DETECTOR_ML_DEFAULTS.get(detector_key)
    if d is None:
        return None
    spec = ML_TASKS.get(d.task)
    if spec is None or not spec.supported or spec.runner not in SERVABLE_RUNNERS:
        return None
    url = endpoint_url or f"{inference_url()}/v1/infer/{spec.task}"
    if in_boundary is None:
        in_boundary = _is_loopback(url)
    strategy: Dict[str, Any] = {
        "order": ["local_deterministic", "local_ml"],
        "fallback": "fallback_to_deterministic",
        "timeout_ms": d.timeout_ms,
    }
    if d.band is not None:
        # Escalation: only the ambiguous deterministic-risk middle escalates to ML.
        strategy["escalate_when"] = {"deterministic_score_between": [d.band[0], d.band[1]]}
    if d.additive:
        # Additive: always run the ML layer and worst-of-merge it with the deterministic
        # result (no band gate) so the deterministic decision is never lost.
        strategy["additive"] = True
    return {
        "strategy": strategy,
        "backends": {
            "local_ml": {
                "task": spec.task,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "threshold": spec.threshold,
                "endpoint_url": url,
                "in_boundary": in_boundary,
                "timeout_ms": d.timeout_ms,
            }
        },
    }
