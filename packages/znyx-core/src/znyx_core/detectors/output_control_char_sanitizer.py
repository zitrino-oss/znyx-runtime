"""Output control-character sanitiser (OWASP LLM10 - Improper Output Handling).

2026 added terminal and log sinks to LLM10: "LLM output containing ANSI escape sequences
or other control characters is written to a terminal, log viewer, or IDE pane that
interprets them, enabling visual spoofing, clipboard hijacking (for example, OSC 52), or
exploitation of known terminal emulator vulnerabilities".

This is the OUTPUT-side counterpart to ``text_normalize.strip_zero_width``, which already
runs on INPUT to stop evasion. The two are not interchangeable and both are needed: input
normalisation stops an attacker hiding instructions from the detectors, while this stops
model output rewriting what a human sees in their terminal. A CLI or IDE assistant is the
common case — the model's answer goes straight to a sink that interprets escapes.

What it flags, worst first:

* **Clipboard hijacking (OSC 52)** — writes to the user's clipboard from a text stream.
  The reader sees ordinary output and pastes something else entirely.
* **Screen manipulation** — cursor moves, line erases, and carriage returns that let
  earlier output be overwritten, so what is on screen is not what was sent.
* **Other C0 and C1 control characters** - BEL, backspace, escape, and the
  8-bit \\x80-\\x9f block, which an 8-bit-clean terminal reads as controls too.

``sanitize()`` returns the neutralised text, so a caller can render safely rather than
only being told it was unsafe. Escapes are made VISIBLE (``\\x1b`` becomes the literal
four characters) rather than deleted, because silently dropping bytes changes the meaning
of the output too, just less obviously.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.core.models import Decision, DetectorResult, RuleHit, Severity
from znyx_core.core.risk import calculate_risk_score

# Terminals accept every one of these sequences in TWO forms: the 7-bit form that
# starts ESC [ or ESC ], and the single-byte C1 form (\x9b for CSI, \x9d for OSC) that
# an 8-bit-clean terminal treats identically. Matching only the 7-bit form left
# "\x9b2J" (erase screen) completely unhandled: not detected, not sanitised, ALLOW.
_CSI_INTRO = r"(?:\x1b\[|\x9b)"
_OSC_INTRO = r"(?:\x1b\]|\x9d)"
# String terminator, likewise: BEL, 7-bit ST (ESC \), or 8-bit ST (\x9c).
_ST = r"(?:\x07|\x1b\\|\x9c)"

# OSC 52 is the clipboard sequence: OSC 52 ; ... ST.
_OSC52_RE = re.compile(_OSC_INTRO + r"52;[^\x07\x1b\x9c]*" + _ST + r"?")
# Any OSC — window title, hyperlinks, and the clipboard case above.
_OSC_RE = re.compile(_OSC_INTRO + r"[^\x07\x1b\x9c]*" + _ST + r"?")
# CSI sequences: colours are harmless, but cursor movement and erase are not.
_CSI_RE = re.compile(_CSI_INTRO + r"[0-9;?]*[A-Za-z]")
_CSI_SCREEN_RE = re.compile(_CSI_INTRO + r"[0-9;?]*[ABCDEFGHJKSTfnsu]")
# SGR (colour / bold / reset) is the one CSI family that only changes appearance and
# cannot move the cursor or erase what is already written.
_CSI_SGR_RE = re.compile(_CSI_INTRO + r"[0-9;]*m")
# Bare escape and the other C0 AND C1 control characters, excluding tab / newline /
# carriage return which are handled separately because they are legitimate in ordinary
# text. The C1 block (\x80-\x9f) is included: those bytes are control characters in
# their own right, not printable text, whether or not they introduce a sequence.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# A carriage return NOT part of a CRLF line ending can rewrite the current line.
_LONE_CR_RE = re.compile(r"\r(?!\n)")


def _visible(match: str) -> str:
    """Render control bytes as printable escapes so nothing is silently dropped."""
    return match.encode("unicode_escape").decode("ascii", "replace")


def sanitize(text: str, allow_color_codes: bool = True) -> Tuple[str, int]:
    """Neutralise control sequences. Returns (safe_text, replacements_made).

    With ``allow_color_codes`` the SGR sequences are protected first and restored at the
    end, so colour survives while everything that can move the cursor, erase the screen,
    or touch the clipboard is made visible."""
    if not text:
        return text, 0
    count = 0
    placeholder = "\x00__SGR%d__\x00"
    preserved: List[str] = []

    if allow_color_codes:
        def keep(m: "re.Match[str]") -> str:
            preserved.append(m.group(0))
            return placeholder % (len(preserved) - 1)
        text = _CSI_SGR_RE.sub(keep, text)

    def repl(m: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        return _visible(m.group(0))

    out = _OSC_RE.sub(repl, text)
    out = _CSI_RE.sub(repl, out)
    out = _LONE_CR_RE.sub(repl, out)
    # The placeholders use NUL, which _CTRL_RE would otherwise eat, so restore before it.
    for i, seq in enumerate(preserved):
        out = out.replace(placeholder % i, "\x00KEEP%d\x00" % i)
    out = _CTRL_RE.sub(repl, out)
    for i, seq in enumerate(preserved):
        out = out.replace(_visible("\x00") + ("KEEP%d" % i) + _visible("\x00"), seq)
    return out, count


class OutputControlCharSanitizerDetector:
    """Flags (and can neutralise) terminal control sequences in model output (LLM10)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        # REDACT is the natural default: the output is usable once neutralised, so
        # blocking the whole response would cost more than it saves.
        self.action = (self.config.get("action") or "REDACT").upper()
        # Colour codes are the one common benign use of CSI, so they are allowed by
        # default; screen-manipulating sequences never are.
        self.allow_color_codes = bool(self.config.get("allow_color_codes", True))

    def detect(self, text: str,
               metadata: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled or not text:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        # Colour codes are permitted by default, so they must not be re-flagged by the
        # generic control-character rule via the ESC byte they start with.
        residual = _CSI_SGR_RE.sub("", text) if self.allow_color_codes else text

        if _OSC52_RE.search(text):
            rule_hits.append(RuleHit(
                rule_id="output_control_char.clipboard_hijack",
                severity=Severity.HIGH,
                message="Output contains an OSC 52 clipboard-write sequence",
            ))
        elif _OSC_RE.search(text):
            rule_hits.append(RuleHit(
                rule_id="output_control_char.osc_sequence",
                severity=Severity.HIGH,
                message="Output contains an OSC terminal sequence",
            ))

        if _CSI_SCREEN_RE.search(text) or _LONE_CR_RE.search(text):
            rule_hits.append(RuleHit(
                rule_id="output_control_char.screen_manipulation",
                severity=Severity.HIGH,
                message=("Output contains cursor/erase sequences that can overwrite what "
                         "is already on screen"),
            ))
        elif not self.allow_color_codes and _CSI_RE.search(residual):
            rule_hits.append(RuleHit(
                rule_id="output_control_char.ansi_sequence",
                severity=Severity.MEDIUM,
                message="Output contains ANSI escape sequences",
            ))

        if _CTRL_RE.search(residual):
            rule_hits.append(RuleHit(
                rule_id="output_control_char.control_characters",
                severity=Severity.MEDIUM,
                message="Output contains non-printable control characters",
            ))

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        risk_score = calculate_risk_score(rule_hits)
        dev = f"output_control_char_sanitizer: {', '.join(sorted({h.rule_id for h in rule_hits}))}"

        if self.action == "BLOCK":
            return DetectorResult(
                decision=Decision.BLOCK, risk_score=risk_score, rule_hits=rule_hits,
                developer_message=dev,
                user_message="The response was withheld because it contained unsafe terminal control codes.",
            )
        if self.action == "WARN":
            return DetectorResult(decision=Decision.WARN, risk_score=risk_score,
                                  rule_hits=rule_hits, developer_message=dev)

        safe, replaced = sanitize(text, self.allow_color_codes)
        return DetectorResult(
            decision=Decision.REDACT, risk_score=risk_score, rule_hits=rule_hits,
            sanitized_text=safe,
            developer_message=f"{dev} ({replaced} sequence(s) neutralised)",
        )
