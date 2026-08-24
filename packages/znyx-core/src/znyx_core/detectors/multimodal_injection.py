"""Cross-modal injection (OWASP LLM01 - Prompt Injection).

2026 folded cross-modal attacks into LLM01: instructions hidden in an image, an audio
track, or a video, extracted by the model's encoder and acted on, while every text filter
sees nothing. LLM01 mitigation #3 is the answer — "Run modality-specific classifiers, OCR
over images, and transcription over audio, then apply text filters to the extracted
content".

**What this detector does and does not do.** It does not perform OCR or transcription.
The runtime is deliberately dependency-minimal, and pulling a vision or speech stack into
it would change what ZNYX is; extraction belongs in the inference sidecar or the calling
application, which already has the media. So this follows the same contract as
``unbounded_consumption``'s token accounting: the caller supplies what it extracted, and
the detector governs it.

That makes it two controls in one, and the second matters more than the first:

* **Governing extracted text.** When the caller supplies OCR or transcript text, it is
  scanned with the SAME injection patterns as ordinary input, so a payload in an image
  is caught by the machinery that already catches it in text.
* **Refusing to be silently blind.** When multimodal content arrives with NO extracted
  text, the request is flagged as unscanned. This is the finding that earns the control:
  without it, an image-bearing request sails through the pipeline looking clean, and
  "no detector fired" reads as "nothing was wrong" when it actually means "nothing was
  looked at". Failing loudly on unscanned media is the difference between coverage and
  the appearance of coverage.

    metadata = {
        "attachments": [{"type": "image", "extracted_text": "IGNORE ALL PREVIOUS..."}],
        "modalities": ["text", "image"],
    }
"""
from typing import Any, Dict, List, Optional

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score
from znyx_core.detectors._injection_patterns import scan_injection

_ATTACHMENT_KEYS = ("attachments", "media", "files", "parts", "content_parts")
_EXTRACTED_KEYS = ("extracted_text", "ocr_text", "transcript", "transcription",
                   "alt_text", "text")
_NON_TEXT_TYPES = ("image", "audio", "video", "document", "pdf", "photo",
                   "input_image", "input_audio", "image_url")


def _attachments(metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    for k in _ATTACHMENT_KEYS:
        v = metadata.get(k)
        if isinstance(v, list):
            return [a for a in v[:32] if isinstance(a, dict)]
    return []


def _is_non_text(att: Dict[str, Any]) -> bool:
    kind = str(att.get("type") or att.get("mime_type") or att.get("kind") or "").lower()
    return any(t in kind for t in _NON_TEXT_TYPES)


def _extracted(att: Dict[str, Any]) -> Optional[str]:
    for k in _EXTRACTED_KEYS:
        v = att.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


class MultimodalInjectionDetector:
    """Governs text extracted from non-text modalities, and flags unscanned media (LLM01)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "BLOCK").upper()
        # Fail loudly when media arrives with nothing extracted. On by default: a control
        # that stays quiet about what it could not inspect is worse than no control, because
        # it makes the gap invisible.
        self.flag_unextracted_media = bool(self.config.get("flag_unextracted_media", True))
        # How many injection patterns must hit the extracted text before it is called one.
        self.match_threshold = max(1, int(self.config.get("match_threshold", 1)))

    def detect(self, text: str,
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        attachments = _attachments(metadata)
        if not attachments:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        unextracted = 0

        for att in attachments:
            if not _is_non_text(att):
                continue
            extracted = _extracted(att)
            if extracted is None:
                unextracted += 1
                continue

            # The SAME scanner the text path uses, including its evasion normalisation.
            # A payload hidden in an image is the same payload; re-implementing the
            # patterns here would mean maintaining two that slowly disagree.
            matched = scan_injection(extracted, "multimodal_injection")
            if len(matched) >= self.match_threshold:
                kind = str(att.get("type") or "media")
                rule_hits.append(RuleHit(
                    rule_id="multimodal_injection.injection_in_extracted_text",
                    severity=Severity.HIGH,
                    message=(f"{len(matched)} injection marker(s) in text extracted from "
                             f"{kind} content: {', '.join(h.rule_id.split('.')[-1] for h in matched[:3])}"),
                ))

        if self.flag_unextracted_media and unextracted:
            rule_hits.append(RuleHit(
                rule_id="multimodal_injection.unscanned_media",
                severity=Severity.MEDIUM,
                message=(f"{unextracted} non-text attachment(s) arrived with no extracted "
                         f"text; their contents were not inspected by any detector"),
            ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"multimodal_injection: {', '.join(sorted({h.rule_id for h in rule_hits}))}"
        # An unscanned attachment is a visibility gap, not a detection. Blocking every
        # request that carries an un-OCR'd image would break ordinary traffic, so that
        # finding warns even in BLOCK mode unless it is the only thing keeping it quiet.
        found_injection = any(h.rule_id.endswith("injection_in_extracted_text") for h in rule_hits)
        if self.action == "BLOCK" and found_injection:
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="Attached content contained instructions that cannot be processed.",
            )
        return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                              rule_hits=rule_hits, developer_message=dev)
