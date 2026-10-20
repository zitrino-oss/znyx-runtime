"""Streaming evaluator - evaluate text chunks with a sliding window.

Buffers incoming chunks and runs detectors when the window fills or on
explicit flush. Designed for Server-Sent Events (SSE) streaming responses
from LLMs where text arrives incrementally.

Text is never forwarded before it has been evaluated: ``push`` releases only
the part of a window that no later window will re-examine, and ``flush``
evaluates whatever is still buffered before releasing it. What is released is
the text *after* the detectors' REDACT/TRANSFORM edits and any ``on_fail``
remediation, never the raw window.

Two modes:

* ``incremental`` (default) releases text as windows clear it. Values a
  detector matches by pattern (secrets, PII) are never released raw. Some
  checks are inherently whole-response (a score that needs two categories
  anywhere in the text, a disclaimer triggered by a topic, an allowlist
  context far away), so after the last window a full-response check runs: it
  can still append a disclaimer, and it reports a BLOCK or edit that only the
  whole text reveals (``done.full_text_differs``). Text already released
  cannot be recalled.
* ``buffered`` evaluates the whole response before releasing any of it:
  identical to the non-streaming path, at the cost of streaming.

Usage:
    evaluator = StreamingEvaluator(policy, window_size=400)
    for chunk in llm_stream:
        for event in evaluator.push(chunk):
            yield event   # SSE event dict
    for event in evaluator.flush():
        yield event       # trailing chunk/guardrail/block events, then `done`
"""
import bisect
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from znyx_core.core.labels import DECISION_PRECEDENCE
from znyx_core.core.models import (
    Decision, EvaluationRequest, EvaluationResponse, RuleHit,
)
from znyx_core.core.text_edits import apply_edits, normalise_edits
from znyx_core.engine.detector_registry import default_registry
from znyx_core.engine.orchestrator import DetectorOrchestrator

logger = logging.getLogger(__name__)

# Minimum text held back after each release point. A sensitive pattern that may
# contain whitespace and starts before the release point must end inside the window
# about to release it, or its start escapes before any window sees it whole. The
# longest such built-in patterns are user-agent strings and street addresses; 160
# covers typical ones. Unbroken tokens of any length (JWTs, API keys, URLs, base64)
# are protected separately: the release point never falls inside one.
MIN_HOLDBACK = 160

# Already-released text shown to the detectors again at the start of each window, so
# keyword-plus-value patterns ("api_key: <token>") still match when the release point
# falls between the keyword and the value. It is the raw text, as the full-text path
# sees it, counted back to this many letters and digits, so a long run of spaces or
# separators (" = = =") between keyword and value does not push the keyword out of
# view, and it starts on a token boundary. Only new text is ever released.
CONTEXT_CHARS = 64
CONTEXT_MAX = 65536

# An unbroken token is never split across windows, however long: splitting one is how a
# JWT escaped. A token, a buffered response, or a window grown around an unfinished edit
# longer than this fails the stream closed rather than buffering without bound. Read at
# call time, so tests can lower it.
MAX_RUN = 1_048_576

# Characters that can form part of a token-like secret or identifier: printable,
# non-space ASCII. Anything else (whitespace, CJK, emoji, accented letters) ends a run,
# so text in scripts without spaces still streams incrementally.
_BOUNDARY = re.compile(r"[^!-~]")

MODES = ("incremental", "buffered")


def _is_token_char(ch: str) -> bool:
    return "!" <= ch <= "~"


def _retreat_to_boundary(text: str, cut: int) -> int:
    """Move ``cut`` back so it does not split a run of token characters.

    Returns 0 when the run straddling ``cut`` starts at the beginning of ``text``.
    """
    if cut <= 0 or cut >= len(text):
        return cut
    while cut > 0 and _is_token_char(text[cut - 1]) and _is_token_char(text[cut]):
        cut -= 1
    return cut


def _advance_to_boundary(text: str, cut: int) -> int:
    """Move ``cut`` forward so it does not split a run of token characters."""
    if cut <= 0 or cut >= len(text):
        return cut
    if _is_token_char(text[cut - 1]) and _is_token_char(text[cut]):
        m = _BOUNDARY.search(text, cut)
        return m.start() if m else len(text)
    return cut


def _common_prefix_len(a: str, b: str) -> int:
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _common_suffix_len(a: str, b: str, limit: int) -> int:
    lo, hi = 0, min(len(a), len(b), limit)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[len(a) - mid:] == b[len(b) - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _without_abuse(policy: Dict[str, Any]) -> Dict[str, Any]:
    """The policy for every window after the first, with per-request accounting off.

    A stream is one request, so only its first window may consume an abuse rate-limit
    token, charge the session budget, or record a jailbreak conversation turn.
    Without this, every window spent a token (a long, harmless answer was blocked as
    rate-limited), charged the session's reported tokens again, and counted as a turn.
    """
    out = dict(policy)
    for key in ("abuse", "unbounded_consumption"):
        section = policy.get(key)
        cfg = dict(section) if isinstance(section, dict) else {}
        cfg["enabled"] = False
        out[key] = cfg
    jailbreak = policy.get("jailbreak")
    if isinstance(jailbreak, dict):
        out["jailbreak"] = {**jailbreak, "track_conversation": False}
    return out


def policy_stream_mode(policy: Dict[str, Any]) -> Optional[str]:
    """The streaming mode a policy sets explicitly (``streaming.mode``), if any."""
    section = policy.get("streaming")
    if isinstance(section, dict):
        mode = str(section.get("mode") or "").strip().lower()
        if mode in MODES:
            return mode
    return None


# Strategy modes that call an LLM judge (engine/escalation.py _LLM_MODES).
_LLM_JUDGE_MODES = frozenset({"local_llm", "remote_llm"})


def whole_response_checks(policy: Dict[str, Any]) -> List[str]:
    """Enabled checks whose result depends on the whole response, so evaluating it
    window by window cannot reproduce the non-streaming result: on_fail filter_field
    (parses the whole output as JSON), sensitive-business-data scoring (a threshold
    over categories found anywhere), competitor rewriting with allowlist contexts,
    malicious-URL redaction (redacts every URL once any is malicious) and schema
    enforcement.

    Also checks that judge the meaning of the answer and can block: groundedness
    (hallucination, citation_integrity) set to BLOCK, and any detector that escalates
    to an LLM judge (the judge's verdict can block whatever the detector's own
    action). Streamed, these see one window at a time and the full answer only at
    the end, after the reader has seen it; buffered, they judge the whole answer once,
    before any of it is released, and a judge is called once rather than per window.
    Vendor moderation APIs (remote_api) classify text locally and stream as before."""
    def section(key):
        value = policy.get(key)
        return value if isinstance(value, dict) and value.get("enabled") else None

    found: List[str] = []
    if str(policy.get("on_fail") or "").lower() == "filter_field" or any(
            isinstance(v, dict) and str(v.get("on_fail") or "").lower() == "filter_field"
            for v in policy.values()):
        found.append("on_fail:filter_field")
    sbd = section("sensitive_business_data")
    if sbd and str(sbd.get("action") or "").upper() in ("REDACT", "BLOCK"):
        found.append("sensitive_business_data")
    comp = section("competitor")
    if comp and str(comp.get("action") or "").upper() == "RESTRICT" and comp.get("allowlist_contexts"):
        found.append("competitor:allowlist_contexts")
    murl = section("malicious_url")
    if murl and str(murl.get("action") or "").upper() == "REDACT":
        found.append("malicious_url:redact")
    if section("structure") or section("schema_enforcement"):
        found.append("structure")
    for key in ("hallucination", "citation_integrity"):
        cfg = section(key)
        if cfg and str(cfg.get("action") or "").upper() == "BLOCK":
            found.append(f"groundedness:{key}")
    for key, cfg in policy.items():
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            continue
        strategy = cfg.get("strategy")
        order = strategy.get("order") if isinstance(strategy, dict) else None
        if isinstance(order, (list, tuple)) and any(
                str(m).lower() in _LLM_JUDGE_MODES for m in order):
            found.append(f"llm_judge:{key}")
    return found


def resolve_stream_mode(policy: Dict[str, Any], requested: str = "incremental") -> Tuple[str, str]:
    """``(mode, reason)``. An explicit policy mode wins; otherwise a policy with
    whole-response checks is buffered automatically; otherwise the caller's choice.
    A caller can always ask for buffered (stricter), never weaken a policy that
    requires it."""
    requested = str(requested or "incremental").strip().lower()
    explicit = policy_stream_mode(policy)
    if explicit == "buffered" or requested == "buffered":
        return "buffered", "policy" if explicit == "buffered" else "request"
    if explicit == "incremental":
        return "incremental", "policy"
    checks = whole_response_checks(policy)
    if checks:
        return "buffered", "whole-response checks: " + ", ".join(checks)
    return "incremental", "request"


# ---------------------------------------------------------------------------
# Edit segments: where, exactly, the detectors changed a window.
#
# A segment (n_start, n_end, o_start, o_end, edited) maps a stretch of the edited
# window back to the original window. Unedited segments are identical in both;
# an edited segment's text in the edited window replaces o_start:o_end of the
# original. Segments tile both strings in order. They are built from the
# orchestrator's per-step edit log, using each step's exact edits
# (DetectorResult.text_edits) or redaction spans when those reproduce the step's
# output, and one coarse region (first to last difference) otherwise. The result
# is verified against both strings; if it does not check out, the whole changed
# area becomes one edited segment. Every released character therefore comes from
# the edited text, and an unedited segment is released only because it is
# byte-identical in both.
# ---------------------------------------------------------------------------

Segment = Tuple[int, int, int, int, bool]


def _coarse_segments(orig: str, new: str) -> List[Segment]:
    if orig == new:
        return [(0, len(orig), 0, len(orig), False)]
    p = _common_prefix_len(orig, new)
    s = _common_suffix_len(orig, new, min(len(orig), len(new)) - p)
    segs: List[Segment] = []
    if p:
        segs.append((0, p, 0, p, False))
    segs.append((p, len(new) - s, p, len(orig) - s, True))
    if s:
        segs.append((len(new) - s, len(new), len(orig) - s, len(orig), False))
    return segs


def _step_edits(before: str, after: str, spans, edits) -> List[Tuple[int, int, int]]:
    """One rewrite step's edits as (start, end, replacement_length) in ``before``."""
    if before == after:
        return []
    if edits:
        ordered = normalise_edits(edits, len(before))
        if ordered is not None and apply_edits(before, ordered) == after:
            return [(a, b, len(r)) for a, b, r in ordered]
    if spans:
        from znyx_core.engine.escalation import apply_redaction_spans
        ordered_spans = sorted(spans, key=lambda sp: (sp[0], -(sp[1] - sp[0])))
        kept, last = [], 0
        for a, b, label in ordered_spans:
            if a >= last and 0 <= a < b <= len(before):
                kept.append((a, b, label))
                last = b
        if kept and apply_redaction_spans(before, kept) == after:
            return [(a, b, len(f"[{str(label).upper()}]")) for a, b, label in kept]
    anchored = _anchored_edits(before, after)
    if anchored is not None:
        return anchored
    p = _common_prefix_len(before, after)
    s = _common_suffix_len(before, after, min(len(before), len(after)) - p)
    return [(p, len(before) - s, len(after) - s - p)]


_ANCHOR = 12          # characters that must match again after an edit to resync
_SCAN = 4096          # how far ahead an edit's end is searched for
_MAX_TRIES = 512      # resync attempts per edit before giving up
_MAX_EDITS = 4096     # edits per step before giving up


def _anchored_edits(before: str, after: str) -> Optional[List[Tuple[int, int, int]]]:
    """Recover a rewrite's separate edits when the detector did not report them.

    Walks both strings; at each difference, finds the nearest point where an
    ``_ANCHOR``-character stretch of ``before`` reappears in ``after`` and treats
    what lies between as one edit. Every stretch it reports as unedited is equal in
    both strings by construction, so the result is always consistent: released text
    still only ever comes from ``after``. It can only be less precise than the
    detector's own edits, never unsafe. Work is capped; past the cap it returns None
    and the caller uses one coarse region.
    """
    edits: List[Tuple[int, int, int]] = []
    i = j = 0
    nb, na = len(before), len(after)
    while True:
        k = _common_prefix_len(before[i:], after[j:]) if i < nb and j < na else 0
        i += k
        j += k
        if i >= nb or j >= na:
            break
        # Choose the resync point with the smallest edit (deleted + inserted
        # characters). Taking the first anchor found instead mistakes a later
        # repeat of ordinary text for the resync point in repetitive text.
        found = None
        best = _SCAN + _MAX_TRIES
        for di in range(0, min(_MAX_TRIES, nb - i)):
            if di >= best:
                break
            block = before[i + di:i + di + _ANCHOR]
            if len(block) < _ANCHOR:
                break
            pos = after.find(block, j, j + min(_SCAN, best - di) + _ANCHOR)
            if pos != -1 and di + (pos - j) < best:
                best = di + (pos - j)
                found = (di, pos - j)
        if found is None:
            break  # no resync: the rest becomes one trailing edit
        di, dj = found
        if di == 0 and dj == 0:
            return None  # cannot happen after a mismatch; be safe
        edits.append((i, i + di, dj))
        i += di
        j += dj
        if len(edits) > _MAX_EDITS:
            return None
    if i < nb or j < na:
        edits.append((i, nb, na - j))
    return edits


def _apply_step(segs: List[Segment], edits: List[Tuple[int, int, int]]) -> List[Segment]:
    """Compose one step's edits into segments over the post-step text.

    ``segs`` tiles the pre-step text (its n-coordinates are pre-step positions);
    ``edits`` are that step's (start, end, replacement_length), sorted and
    disjoint, in pre-step positions. An edit touching an earlier edited segment
    absorbs it whole, so a segment is never split. Raises on an inconsistency;
    the caller then falls back to the coarse form.
    """
    if not edits:
        return segs
    total = segs[-1][1]
    starts = [seg[0] for seg in segs]

    def seg_at(x: int) -> Segment:
        i = bisect.bisect_right(starts, x) - 1
        return segs[max(0, min(i, len(segs) - 1))]

    out: List[Segment] = []
    n_out = 0
    cur = 0               # pre-step position consumed so far
    si = 0                # index of the next pre-step segment to copy
    last_pre_end = None   # pre-step end of the last edited piece this step emitted

    def copy_until(x: int) -> None:
        """Copy pre-step segments covering [cur, x) into ``out``, including any
        zero-length segment (an earlier deletion) positioned at or before ``x``."""
        nonlocal cur, n_out, si
        while si < len(segs):
            ns, ne, os_, oe, ed = segs[si]
            if ns == ne:
                if ns < cur:          # already absorbed into an edit
                    si += 1
                    continue
                if ns > x:
                    break
                if os_ != oe:         # a deletion: keep it, in order
                    out.append((n_out, n_out, os_, oe, ed))
                si += 1
                continue
            if ne <= cur:             # wholly behind
                si += 1
                continue
            if ns >= x and cur >= x:
                break
            stop = min(x, ne)
            if stop <= cur:
                break
            if ed:
                if cur != ns or stop != ne:
                    raise ValueError("edited segment split")
                out.append((n_out, n_out + (ne - ns), os_, oe, True))
            else:
                o0 = os_ + (cur - ns)
                out.append((n_out, n_out + (stop - cur), o0, o0 + (stop - cur), False))
            n_out += stop - cur
            cur = stop
            if cur >= ne:
                si += 1

    def orig_start(x: int) -> int:
        if x >= total:
            return segs[-1][3]
        ns, _ne, os_, _oe, ed = seg_at(x)
        return os_ if ed else os_ + (x - ns)

    def orig_end(x: int) -> int:
        if x <= 0:
            return 0
        ns, _ne, os_, oe, ed = seg_at(x - 1)
        return oe if ed else os_ + (x - ns)

    for a, b, r in edits:
        if a == b and r == 0:
            continue
        if not (0 <= a <= b <= total):
            raise ValueError("edit out of range")
        a2, b2 = a, b
        if a < total:
            ns, _ne, _os, _oe, ed = seg_at(a)
            if ed and ns < a:
                a2 = ns
        if b > 0:
            _ns, ne, _os, _oe, ed = seg_at(b - 1)
            if ed and b < ne:
                b2 = ne
        if a2 < cur:
            # Reaches back into the edited piece this step just emitted: merge.
            if last_pre_end is None or not out or not out[-1][4]:
                raise ValueError("unexpected overlap")
            ns0, ne0, oa0, _ob0, _ = out[-1]
            length = (ne0 - ns0) + (a - last_pre_end) + r + (b2 - b)
            out[-1] = (ns0, ns0 + length, oa0, orig_end(b2), True)
            n_out = ns0 + length
        else:
            copy_until(a2)
            length = (a - a2) + r + (b2 - b)
            out.append((n_out, n_out + length, orig_start(a2), orig_end(b2), True))
            n_out += length
        cur = b2
        last_pre_end = b2
    copy_until(total)
    return out


def _verify_segments(segs: List[Segment], orig: str, new: str) -> bool:
    n_pos = o_pos = 0
    for ns, ne, os_, oe, ed in segs:
        if ns != n_pos or os_ != o_pos or ne < ns or oe < os_:
            return False
        if not ed and new[ns:ne] != orig[os_:oe]:
            return False
        n_pos, o_pos = ne, oe
    return n_pos == len(new) and o_pos == len(orig)


def _edit_segments(orig: str, new: str, steps: List[tuple]) -> List[Segment]:
    """Segments describing how ``new`` was produced from ``orig`` (see above)."""
    if orig == new:
        return [(0, len(orig), 0, len(orig), False)]
    segs: List[Segment] = [(0, len(orig), 0, len(orig), False)]
    current = orig
    try:
        for step in steps:
            before, after = step[0], step[1]
            spans = step[2] if len(step) > 2 else None
            edits = step[3] if len(step) > 3 else None
            if before is not current and before != current:
                return _coarse_segments(orig, new)
            segs = _apply_step(segs, _step_edits(before, after, spans, edits))
            current = after
        if current != new:
            segs = _apply_step(segs, _step_edits(current, new, None, None))
    except Exception:  # defensive: any surprise falls back to the safe form
        logger.debug("edit segment composition failed", exc_info=True)
        return _coarse_segments(orig, new)
    if not _verify_segments(segs, orig, new):
        return _coarse_segments(orig, new)
    return segs


def _plan_release(segs: List[Segment], orig: str, new_len: int, start: int, cut: int,
                  holdback: int, final: bool,
                  tail_insertions: bool = True) -> Optional[Tuple[int, int, int, Optional[int]]]:
    """Choose what to release from one evaluated window.

    ``start`` is where the pending text begins (after the released context) and
    ``cut`` the proposed release point, both in original-window coordinates. The
    cut is moved out of any edit that straddles it and always lands on a token
    boundary: back to the start of the edit's token when that still releases
    something; otherwise forward to the end of the edit's token, but only if the
    full hold-back still follows it inside the window (so the edit, and anything
    after it, is known to be complete). Once it has moved forward it only moves
    forward, so it cannot oscillate between two edits.

    Returns ``(n_from, n_to, cut, edit_end)``: release ``new[n_from:n_to]``, keep
    the original from ``cut`` on; ``edit_end`` is the offset in the released text
    just after its last edit (None when it has none). Returns None when the
    window must grow before a safe cut exists.
    """
    n = len(orig)
    limit = n if final else n - holdback
    forward = False
    for _ in range(len(segs) + 4):
        straddle = None
        for seg in segs:
            if seg[4] and seg[2] < cut < seg[3]:
                straddle = seg
                break
        if straddle is None:
            snapped = _retreat_to_boundary(orig, cut)
            if snapped == cut:
                break
            if not forward and snapped > start:
                cut = snapped
                continue
            nxt = _advance_to_boundary(orig, cut)
            if nxt > limit:
                return None
            cut, forward = nxt, True
            continue
        if not forward:
            back = _retreat_to_boundary(orig, straddle[2])
            if back > start:
                cut = back
                continue
        nxt = _advance_to_boundary(orig, straddle[3])
        if nxt > limit or nxt <= cut:
            return None
        cut, forward = nxt, True
    else:
        return None

    # Where the release begins in the edited text. Edits wholly inside the context
    # were released with it; an edit reaching from the context into new text is
    # released again from its start, so its replacement covers the new part.
    n_from = new_len
    for ns, _ne, os_, oe, ed in segs:
        if ed and os_ < start < oe:
            n_from = ns
            break
        if (ed and os_ >= start) or (not ed and os_ <= start < oe):
            n_from = ns if ed else ns + (start - os_)
            break

    # Where it ends. An insertion exactly at the cut belongs to the text after it,
    # except at the very end of a final window.
    n_to = new_len
    if not (final and cut == n and tail_insertions):
        for ns, _ne, os_, oe, ed in segs:
            if ed and os_ >= cut:
                n_to = ns
                break
            if not ed and os_ <= cut < oe:
                n_to = ns + (cut - os_)
                break

    edit_end = None
    for ns, ne, _os, _oe, ed in segs:
        if ed and n_from < ne <= n_to:
            edit_end = ne - n_from
    return n_from, max(n_from, n_to), cut, edit_end


_default_remediation = None
_default_remediation_lock = threading.Lock()


def _shared_remediation():
    """A process-wide RemediationHandler for callers that do not pass one."""
    global _default_remediation
    with _default_remediation_lock:
        if _default_remediation is None:
            from znyx_core.engine.remediation import RemediationHandler
            _default_remediation = RemediationHandler()
        return _default_remediation


class StreamingEvaluator:
    """Sliding-window streaming evaluator.

    Buffers text chunks and runs the detector pipeline when enough text
    accumulates (``window_size`` characters). Each window holds back at least
    ``overlap`` characters, which the next window re-evaluates.

    Each ``push()`` call returns a list of SSE event dicts:
    - ``{"event": "guardrail", "data": {...}}`` when a window is evaluated
    - ``{"event": "chunk", "data": {...}}`` carrying text that window cleared
    - ``{"event": "block", "data": {...}}`` if a BLOCK decision is reached

    A ``chunk`` event is only ever emitted *after* the window covering its text
    has been evaluated, and always after that window's ``guardrail`` event, so
    a consumer that stops on a non-ALLOW verdict never renders the text the
    verdict rejected. Released text has the window's redactions and ``on_fail``
    remediation applied.

    Rules that stop a sensitive value escaping across a window boundary:
    the release point never falls inside a run of token characters (an
    unbroken token is always evaluated whole) and never inside an edit, which is
    located exactly (see the edit segments above); at least ``MIN_HOLDBACK``
    characters are held back after it; and each window starts with the raw
    released context so patterns that need a preceding keyword match.

    ``flush()`` evaluates any remaining buffered text, releases it if it is
    not blocked, runs the full-response check (incremental mode), and returns
    those trailing events followed by the ``done`` summary.

    ``orchestrator``, ``judge_ctx`` and ``remediation`` let the runtime pass the
    same configured detector pipeline (egress audit sink, custom plugins), judge
    context and remediation handler that non-streaming evaluations use.
    """

    def __init__(
        self,
        policy: Dict[str, Any],
        context: str = "output",
        window_size: int = 400,
        overlap: int = MIN_HOLDBACK,
        request: Optional[EvaluationRequest] = None,
        *,
        orchestrator: Optional[DetectorOrchestrator] = None,
        judge_ctx: Any = None,
        remediation: Any = None,
        mode: str = "incremental",
        final_check: bool = True,
    ):
        self.policy = policy
        self.context = context
        # A policy that requires buffered streaming cannot be weakened by the caller.
        self.mode, self.mode_reason = resolve_stream_mode(policy, mode)
        self.final_check = final_check
        requested_window = max(50, window_size)
        # A smaller hold-back than MIN_HOLDBACK is unsafe, so it is raised rather
        # than honoured, and the window grows to keep overlap <= window_size // 2.
        self.overlap = max(MIN_HOLDBACK, min(overlap, requested_window // 2))
        self.window_size = max(requested_window, 2 * self.overlap)

        self._policy_after_first = _without_abuse(policy)
        self._buffer = ""
        self._full_text = ""
        self._chunk_count = 0
        self._window_count = 0
        self._blocked = False
        self._failed = False
        self._released = ""        # what the caller has received (edited)
        self._released_raw = ""    # the original text it came from (detector context)
        self._release_count = 0
        self._hits: Dict[Tuple[str, str], RuleHit] = {}
        self._max_risk = 0
        self._total_latency_ms = 0
        self._worst = Decision.ALLOW
        self._full_text_differs = False
        self._lock = threading.Lock()

        # Window planning state: whether the buffer is known to open with an
        # unbroken run (and how far it has been scanned for its end), and the
        # minimum end of the next window after a deferral (it doubles each time,
        # so re-evaluation stays linear in the text).
        self._held = False
        self._scan_pos = 0
        self._min_end = 0

        self._request = request or EvaluationRequest(
            request_id="stream-0", tenant_id="stream", app_id="stream", text="",
        )
        self._orchestrator = orchestrator or DetectorOrchestrator(default_registry)
        self._judge_ctx = judge_ctx
        self._remediation = remediation if remediation is not None else _shared_remediation()

    @property
    def is_blocked(self) -> bool:
        return self._blocked

    @property
    def released_text(self) -> str:
        """The text released to the caller so far, i.e. everything a consumer
        following the ``chunk`` events has been cleared to show."""
        return self._released

    @property
    def final_decision(self) -> Decision:
        """The most severe decision reached by any window so far."""
        return self._worst

    @property
    def max_risk(self) -> int:
        return self._max_risk

    @property
    def rule_hits(self) -> List[RuleHit]:
        """Distinct rule hits across the stream. A value inside the hold-back or
        the context is seen by two windows, so per-window hits double-count."""
        return list(self._hits.values())

    @property
    def total_latency_ms(self) -> int:
        return self._total_latency_ms

    def push(self, chunk: str) -> List[Dict[str, Any]]:
        """Push a text chunk. Returns list of SSE events.

        The chunk is buffered, not forwarded. In incremental mode, whenever the
        buffer holds a full window that window is evaluated, and unless it
        BLOCKs, the part of it the next window will not re-examine is released as
        a ``chunk`` event, edited. In buffered mode nothing is released before
        ``flush``. On a BLOCK nothing further is released.
        """
        if self._blocked:
            return [{"event": "block", "data": {"reason": "Stream blocked by previous window evaluation"}}]

        self._buffer += chunk
        self._full_text += chunk
        self._chunk_count += 1

        if self.mode == "buffered":
            if len(self._buffer) > MAX_RUN:
                return [self._block_now(f"Response longer than {MAX_RUN} characters (fail-closed)")]
            return []

        events: List[Dict[str, Any]] = []
        while not self._blocked:
            plan = self._plan()
            if plan is None:
                break
            events.extend(self._step(*plan, final=False))

        if not self._blocked and self._held and len(self._buffer) > MAX_RUN:
            events.append(self._block_now(
                f"Unbroken token longer than {MAX_RUN} characters (fail-closed)"))
        return events

    def flush(self) -> List[Dict[str, Any]]:
        """Evaluate whatever is still buffered, then return the trailing events.

        Returns the events for the final window (``guardrail``/``block`` verdict,
        plus a ``chunk`` carrying the tail if it was not blocked), then any events
        from the full-response check (incremental mode), then the ``done``
        summary.
        """
        events: List[Dict[str, Any]] = []
        # When a full-response check will follow, the final window holds back an
        # insertion at the very end (a disclaimer): the check releases the one the
        # whole text calls for, once, instead of the last window's version too.
        check_follows = self.mode == "incremental" and self.final_check and self._window_count >= 1
        while self._buffer and not self._blocked:
            n = len(self._buffer)
            events.extend(self._step(n, n, final=True, tail_insertions=not check_follows))

        if (self.mode == "incremental" and self.final_check and not self._blocked
                and not self._failed and self._full_text and self._window_count > 1):
            events.extend(self._full_response_check())

        events.append(self.done_event())
        return events

    def done_event(self, reason: Optional[str] = None) -> Dict[str, Any]:
        """The ``done`` summary for the stream so far."""
        data: Dict[str, Any] = {
            "total_chunks": self._chunk_count,
            "windows_evaluated": self._window_count,
            "final_decision": self._worst.value,
            "max_risk_score": self._max_risk,
            "total_rule_hits": len(self._hits),
            "total_latency_ms": self._total_latency_ms,
            "full_text_length": len(self._full_text),
            # What actually reached the caller. On a clean stream this equals
            # full_text_length; with redactions it is the redacted length; on
            # BLOCK it is the evaluated prefix released before the offending window.
            "released_text_length": len(self._released),
            "effective_window_size": self.window_size,
            "effective_overlap": self.overlap,
            "mode": self.mode,
            "mode_reason": self.mode_reason,
            # True when the full-response check would have produced different output
            # from what was streamed (see the module docstring).
            "full_text_differs": self._full_text_differs,
        }
        if reason:
            data["reason"] = reason
        return {"event": "done", "data": data}

    def fail_closed(self, reason: str) -> List[Dict[str, Any]]:
        """Stop the stream without evaluating further, e.g. when the detector
        pool is saturated or a window missed its deadline. Returns a ``block``
        event and the ``done`` summary; nothing more is released, even by a
        window still running in a worker thread."""
        with self._lock:
            self._failed = True
            self._blocked = True
            self._worst = Decision.BLOCK
        return [{"event": "block", "data": {"reason": reason}}, self.done_event(reason=reason)]

    # -- internals ---------------------------------------------------------

    def _block_now(self, reason: str) -> Dict[str, Any]:
        with self._lock:
            self._blocked = True
            self._worst = Decision.BLOCK
        return {"event": "block", "data": {"reason": reason}}

    def _plan(self) -> Optional[Tuple[int, int]]:
        """Choose the next window while streaming: ``(end, cut)`` in buffer
        coordinates, where ``buffer[:end]`` is evaluated and ``buffer[:cut]``
        released. None means wait for more text."""
        buf = self._buffer
        n = len(buf)
        if not self._held:
            end = max(self.window_size, self._min_end)
            if n < end:
                return None
            if end < n and self._min_end == 0:
                # Do not end the window inside a word either: a token cut off at
                # the window's edge can match a pattern it does not match whole.
                # Only for a first attempt: after a deferral the window must reach
                # at least _min_end, or it would shrink back to the same window
                # and be evaluated forever.
                e2 = end
                while e2 > 0 and not buf[e2 - 1].isspace() and not buf[e2].isspace():
                    e2 -= 1
                if e2 - self.overlap > 0:
                    end = e2
            cut = _retreat_to_boundary(buf, end - self.overlap)
            if cut > 0:
                return end, cut
            # The buffer opens with an unbroken run reaching past the release
            # point. Hold it until it ends, then evaluate it whole.
            self._held = True
            self._scan_pos = end - self.overlap

        m = _BOUNDARY.search(buf, self._scan_pos)
        if m is None:
            self._scan_pos = n
            return None
        run_end = m.start()
        self._scan_pos = run_end
        end = max(run_end + self.overlap, self._min_end)
        if n < end:
            return None  # the full hold-back after the run has not arrived yet
        return end, run_end

    def _context(self) -> str:
        """Raw released text to show the detectors again: back far enough to
        include ``CONTEXT_CHARS`` letters and digits, from a token boundary, at most
        CONTEXT_MAX."""
        r = self._released_raw
        if not r:
            return ""
        floor = max(0, len(r) - CONTEXT_MAX)
        i, seen = len(r), 0
        while i > floor and seen < CONTEXT_CHARS:
            i -= 1
            if r[i].isalnum():
                seen += 1
        # Start on a token boundary: a token cut at the context's start can match
        # differently from the whole token (and change which overlapping match wins).
        while i > floor and _is_token_char(r[i - 1]) and _is_token_char(r[i]):
            i -= 1
        return r[i:]

    def _step(self, end: int, cut: int, final: bool, tail_insertions: bool = True) -> List[Dict[str, Any]]:
        """Evaluate ``buffer[:end]`` (after the released context) and release up
        to ``cut``, edited."""
        context = self._context()
        text = context + self._buffer[:end]
        start = len(context)

        events, decision, edited, steps = self._evaluate_window(text)
        if self._failed:
            return []
        if decision == Decision.BLOCK:
            return events

        segs = _edit_segments(text, edited, steps)
        plan = _plan_release(segs, text, len(edited), start, start + cut, self.overlap, final,
                             tail_insertions=tail_insertions)
        if plan is None:
            # An edit sits across the release point and is not known to be
            # complete. Evaluate a window twice as large instead of guessing. It
            # always grows past both the window just tried and the last minimum,
            # so repeated deferrals cannot settle on the same window.
            self._min_end = max(2 * self._min_end, 2 * end, end + self.overlap)
            if self._min_end > MAX_RUN:
                return [self._block_now(
                    f"Edit spans more than {MAX_RUN} characters (fail-closed)")]
            return []
        n_from, n_to, cut_t, _edit_end = plan
        released = edited[n_from:n_to]

        self._released_raw += self._buffer[:cut_t - start]
        self._buffer = self._buffer[cut_t - start:]
        self._held = False
        self._scan_pos = 0
        self._min_end = 0

        guardrail = events[-1]["data"]
        guardrail["text_preview"] = released[:80] + "..." if len(released) > 80 else released
        if released:
            events.append(self._release(released))
        return events

    def _full_response_check(self) -> List[Dict[str, Any]]:
        """Evaluate the whole response once, as the non-streaming path would, and
        apply what streaming cannot know until the end: a BLOCK only the whole
        text reveals, and an insertion at the very end (a disclaimer)."""
        full = self._full_text
        result, decision, edited, steps, user_message, elapsed_ms = self._run_pipeline(
            full, self._policy_after_first)
        events: List[Dict[str, Any]] = []
        with self._lock:
            if self._failed:
                return []
            self._total_latency_ms += elapsed_ms
            self._max_risk = max(self._max_risk, result.risk_score)
            for hit in result.rule_hits:
                self._hits.setdefault((hit.rule_id, hit.message), hit)
            if DECISION_PRECEDENCE.get(decision, 0) > DECISION_PRECEDENCE.get(self._worst, 0):
                self._worst = decision

        if decision == Decision.BLOCK:
            with self._lock:
                self._blocked = True
                self._full_text_differs = True
            data: Dict[str, Any] = {
                "reason": "Full-response check (text already streamed cannot be recalled)",
                "full_response": True,
                "rule_hits": [{"rule_id": h.rule_id, "message": h.message} for h in result.rule_hits],
            }
            if user_message:
                data["user_message"] = user_message
            events.append({"event": "block", "data": data})
            return events

        tail = ""
        if edited != full:
            segs = _edit_segments(full, edited, steps)
            last = segs[-1] if segs else None
            if last and last[4] and last[2] == last[3] == len(full):
                tail = edited[last[0]:last[1]]
        if tail and not self._released.endswith(tail):
            events.append(self._release(tail))
        expected = edited
        if self._released != expected:
            self._full_text_differs = True
        return events

    def _release(self, text: str) -> Dict[str, Any]:
        """Build the ``chunk`` event for evaluated text.

        The only place a ``chunk`` event is created, so every byte the caller
        receives has passed a detector. ``chunk_index`` counts releases (not
        ``push`` calls): windows do not line up with chunk boundaries, so it is
        the release order that tells a consumer how to reassemble the text.
        """
        self._released += text
        self._release_count += 1
        return {
            "event": "chunk",
            "data": {"text": text, "chunk_index": self._release_count},
        }

    def _run_pipeline(self, text: str, policy: Dict[str, Any]):
        """Detectors, then on_fail remediation, on ``text``. Returns the aggregate
        result, the decision, the edited text, the edit log, any user message and
        the elapsed milliseconds."""
        t0 = time.perf_counter()
        if self.context == "input":
            orch = self._orchestrator.run_input_detectors(
                text, policy, self._request, judge_ctx=self._judge_ctx)
        else:
            orch = self._orchestrator.run_output_detectors(
                text, policy, self._request, judge_ctx=self._judge_ctx)

        from znyx_core.core.decision import DecisionAggregator
        result = DecisionAggregator.aggregate(orch.results)
        decision = result.decision or Decision.ALLOW
        edited = orch.current_text
        steps = list(getattr(orch, "edit_steps", None) or [])
        user_message = None

        # on_fail remediation, as the non-streaming path applies it: refrain and
        # ask_human turn the window into a BLOCK; fix and filter_field edit it.
        if decision != Decision.ALLOW and self._remediation is not None:
            response = EvaluationResponse(
                request_id=self._request.request_id,
                decision=decision,
                risk_score=result.risk_score,
                policy_version=str(self.policy.get("policy_version") or "stream"),
                rule_hits=list(result.rule_hits),
                sanitized_text=edited,
            )
            # The stream's request carries no text (it arrives in chunks): hand the
            # handler this window's text, so filter_field filters it and ask_human
            # records what was flagged, instead of both seeing an empty string.
            window_request = self._request.model_copy(update={"text": text})
            response = self._remediation.apply(response, self.policy, orch.results, window_request)
            decision = response.decision or decision
            user_message = response.user_message
            if (decision in (Decision.REDACT, Decision.TRANSFORM)
                    and response.sanitized_text is not None and response.sanitized_text != edited):
                steps.append((edited, response.sanitized_text, None, None))
                edited = response.sanitized_text

        return result, decision, edited, steps, user_message, int((time.perf_counter() - t0) * 1000)

    def _evaluate_window(self, text: str) -> Tuple[List[Dict[str, Any]], Decision, str, List[tuple]]:
        """Run detectors and remediation on one window. Returns the verdict
        events, the decision, the window text after any edits and the edit log."""
        policy = self.policy if self._window_count == 0 else self._policy_after_first
        self._window_count += 1
        result, decision, edited, steps, user_message, elapsed_ms = self._run_pipeline(text, policy)

        with self._lock:
            if self._failed:
                return [], Decision.BLOCK, text, []
            self._total_latency_ms += elapsed_ms
            self._max_risk = max(self._max_risk, result.risk_score)
            for hit in result.rule_hits:
                self._hits.setdefault((hit.rule_id, hit.message), hit)
            if DECISION_PRECEDENCE.get(decision, 0) > DECISION_PRECEDENCE.get(self._worst, 0):
                self._worst = decision
            if decision == Decision.BLOCK:
                self._blocked = True

        event_data: Dict[str, Any] = {
            "window_index": self._window_count,
            "decision": decision.value,
            "risk_score": result.risk_score,
            "rule_hits": [{"rule_id": h.rule_id, "message": h.message} for h in result.rule_hits],
            "latency_ms": elapsed_ms,
        }

        if decision == Decision.BLOCK:
            # No text_preview on a BLOCK. The verdict event must not carry the
            # very text the stream is withholding: a proxy that forwards or logs
            # every SSE event would otherwise re-open the channel `chunk`
            # suppression just closed. Callers correlate on window_index and
            # rule_hits instead. A remediation message (refrain) is safe to show.
            if user_message:
                event_data["user_message"] = user_message
            return [{"event": "block", "data": event_data}], decision, text, []

        # The preview is added by _step from the released (edited) text, so it
        # never shows anything the chunk event does not.
        return [{"event": "guardrail", "data": event_data}], decision, edited, steps
