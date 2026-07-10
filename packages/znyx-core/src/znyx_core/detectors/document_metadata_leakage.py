"""Document metadata-leakage detector (deferred backlog, deterministic).

Flags document *artifacts* that leak when office/PDF/web content is pasted or echoed:
tracked-changes & comment markup, revision history, internal absolute file paths, and
hidden zero-width / bidi-override text. Pure-rules, no dependencies — a
deterministic detector. (Image-borne / EXIF metadata waits on the multimodal track.)

Scoped to unambiguous markers to keep false positives low; default action is advisory
(WARN) and the detector is opt-in (default-disabled in the pipeline)."""
import re
from typing import Any, Dict, List

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# Tracked-changes / comments / revision markup (OOXML, RTF, HTML comments).
_MARKUP_RE = re.compile(
    r"<w:(?:ins|del|comment(?:RangeStart|RangeEnd|Reference)?)\b"      # Word OOXML revisions/comments
    r"|<!--.*?-->"                                                     # HTML/XML comments
    r"|\\(?:revised|deleted|revauth|annotation)\b"                     # RTF revision marks
    r"|\[(?:comment|tracked changes?)\b[^\]]*\]",                      # inline [Comment: ...] / [Tracked changes]
    re.IGNORECASE | re.DOTALL,
)
# Document-property fields that leak author/origin (require the explicit label form).
_PROPERTY_RE = re.compile(
    r"\b(?:Last\s+Modified\s+By|Author|Creator|Company|Manager|Revision\s+number|Total\s+editing\s+time)\s*[:=]\s*\S+",
    re.IGNORECASE,
)
# Internal absolute file paths (Windows user dir, UNC share, macOS user dir).
_PATH_RE = re.compile(
    r"[A-Za-z]:\\Users\\[^\\\s]+\\"          # C:\Users\name\
    r"|\\\\[A-Za-z0-9._-]+\\[^\\\s]+"         # \\server\share
    r"|/Users/[^/\s]+/",                      # /Users/name/
)
# Hidden text: a run of zero-width chars, or any bidi-override control char.
_HIDDEN_RE = re.compile(
    r"[​‌‍﻿]{2,}"         # ≥2 consecutive zero-width chars
    r"|[‪-‮⁦-⁩]",          # bidi embedding/override controls
)


class DocumentMetadataLeakageDetector:
    """Deterministic document-artifact / hidden-text leakage detector."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.action = (self.config.get("action") or "WARN").upper()
        self.block_threshold = self.config.get("block_threshold", 60)

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        hits: List[RuleHit] = []
        if _MARKUP_RE.search(text):
            hits.append(RuleHit(rule_id="document_metadata_leakage.revision_markup",
                                message="Tracked-changes / comment / revision markup present", severity=Severity.HIGH))
        if _PROPERTY_RE.search(text):
            hits.append(RuleHit(rule_id="document_metadata_leakage.document_properties",
                                message="Document property field (author/creator/company) leaked", severity=Severity.MEDIUM))
        if _PATH_RE.search(text):
            hits.append(RuleHit(rule_id="document_metadata_leakage.internal_path",
                                message="Internal absolute file path leaked", severity=Severity.MEDIUM))
        if _HIDDEN_RE.search(text):
            hits.append(RuleHit(rule_id="document_metadata_leakage.hidden_text",
                                message="Hidden zero-width / bidi-override text present", severity=Severity.HIGH))

        if not hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(hits)
        if self.action == "BLOCK" and risk_score >= self.block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=hits,
                user_message="This content was blocked: it contains document metadata or hidden text.",
                developer_message=f"document_metadata_leakage: {len(hits)} signal(s)",
            )
        return DetectorResult(
            decision=Decision.WARN, risk_score=risk_score, rule_hits=hits,
            developer_message=f"document_metadata_leakage: {len(hits)} signal(s) (advisory)",
        )
