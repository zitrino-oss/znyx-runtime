"""A sound pre-filter that skips a regex scan when the text cannot match.

The jailbreak and toxicity detectors run hundreds of patterns over every text; most
patterns can only match when a particular word is present ("ignore", "pretend",
...). For each pattern this module derives a literal that every match must contain
(from the parsed pattern: a run of ASCII literals that is not optional, not inside
an alternation and not inside a lookaround). A substring test for that literal
(C-speed) then decides whether the regex needs to run at all.

Soundness, not speed, is the point of the details: a skipped pattern must be one
that could not have matched. Case-insensitive patterns are compared against the text
folded the way ``re`` folds it, including the four non-ASCII letters ``re`` treats
as equal to ASCII ones under IGNORECASE (dotted and dotless i, long s, Kelvin sign);
without them, "previouſ" would match a pattern for "previous" yet fail the
substring test, which is an evasion. That table is computed from the running
Python's own ``re`` at first use, so it cannot drift from the engine it mirrors.
"""
from __future__ import annotations

import re
import sys
import threading
from typing import Dict, List, Optional, Tuple

try:  # Python 3.11+
    from re import _constants as _c, _parser as _p
except ImportError:  # pragma: no cover - older Pythons
    import sre_constants as _c  # type: ignore
    import sre_parse as _p  # type: ignore

MIN_LITERAL = 3

_lock = threading.Lock()
_fold_table: Optional[Dict[int, str]] = None
_analysis: Dict[re.Pattern, Tuple[Optional[Tuple[str, ...]], bool]] = {}


def _table() -> Dict[int, str]:
    """Non-ASCII characters ``re`` matches against an ASCII letter or digit under
    IGNORECASE, mapped to that character."""
    global _fold_table
    if _fold_table is None:
        with _lock:
            if _fold_table is None:
                every = "".join(chr(c) for c in range(sys.maxunicode + 1) if not 0xD800 <= c <= 0xDFFF)
                table: Dict[int, str] = {}
                for m in re.finditer(r"(?i)[a-z0-9]", every):
                    ch = m.group()
                    if ch.isascii():
                        continue
                    for target in "abcdefghijklmnopqrstuvwxyz0123456789":
                        if re.fullmatch("(?i)" + target, ch):
                            table[ord(ch)] = target
                            break
                _fold_table = table
    return _fold_table


def fold(text: str) -> str:
    """The text as a case-insensitive ``re`` sees it, for substring tests against
    lower-case ASCII literals."""
    return text.translate(_table()).lower()


Requirement = Tuple[str, ...]          # any one of these literals must be present


class _MixedCase(Exception):
    """The pattern switches case sensitivity inside itself."""


def _requirements(items, ignorecase: bool, out: List[Requirement]) -> None:
    """Collect what every match of a parsed (sub)pattern must contain into ``out``:
    each entry is a tuple of literals of which at least one is present (a plain
    literal is a one-element tuple; an alternation whose every branch has a literal
    of its own gives the tuple of those literals). All entries hold at once."""
    run: List[str] = []

    def end() -> None:
        if run:
            literal = "".join(run)
            if len(literal) >= MIN_LITERAL:
                out.append((literal,))
            run.clear()

    for op, av in items:
        if op is _c.LITERAL:
            ch = chr(av)
            if ch.isascii():
                run.append(ch.lower() if ignorecase else ch)
                continue
            end()
        elif op is _c.AT:
            continue                               # zero-width (\b, ^, $): adjacency holds
        elif op is _c.SUBPATTERN:
            end()
            sub = av[-1]
            add, remove = (av[1], av[2]) if len(av) == 4 else (0, 0)
            sub_ignore = (ignorecase or bool(add & re.IGNORECASE)) and not (remove & re.IGNORECASE)
            if sub_ignore != ignorecase:
                # A scoped (?i:...) or (?-i:...): one haystack cannot serve both case
                # rules, so nothing is proven and the regex always runs.
                raise _MixedCase()
            _requirements(sub, sub_ignore, out)
        elif op in (_c.MAX_REPEAT, _c.MIN_REPEAT) or op is getattr(_c, "POSSESSIVE_REPEAT", None):
            end()
            low, _high, sub = av
            if low >= 1:
                _requirements(sub, ignorecase, out)
        elif op is getattr(_c, "ATOMIC_GROUP", None):
            end()
            _requirements(av, ignorecase, out)
        elif op is _c.BRANCH:
            end()
            alternatives: List[str] = []
            for branch in av[1]:
                inner: List[Requirement] = []
                _requirements(branch, ignorecase, inner)
                singles = [r[0] for r in inner if len(r) == 1]
                if not singles:
                    alternatives = []
                    break                          # one branch guarantees nothing
                alternatives.append(max(singles, key=len))
            if alternatives:
                out.append(tuple(dict.fromkeys(alternatives)))
        else:
            # IN, ANY, NOT_LITERAL, CATEGORY, GROUPREF, lookarounds, ...: nothing here
            # is guaranteed, so the run stops.
            end()
    end()


MAX_REQUIREMENTS = 4


def _ranked(requirements: List[Requirement]) -> Optional[Tuple[Requirement, ...]]:
    """The requirements worth checking, most selective first (longer literals are
    rarer; a single literal is cheaper than an alternation of the same length)."""
    if not requirements:
        return None
    unique = list(dict.fromkeys(requirements))
    unique.sort(key=lambda r: (min(len(x) for x in r), len(r) == 1, -len(r)), reverse=True)
    return tuple(unique[:MAX_REQUIREMENTS])


def analyse(pattern: re.Pattern) -> Tuple[Optional[Tuple[Requirement, ...]], bool]:
    """``(requirements, case_insensitive)``: every match contains at least one
    literal of EACH requirement; None when nothing can be proven."""
    cached = _analysis.get(pattern)
    if cached is not None:
        return cached
    requirement: Optional[Tuple[Requirement, ...]] = None
    ignorecase = bool(pattern.flags & re.IGNORECASE)
    if isinstance(pattern.pattern, str):
        try:
            parsed = _p.parse(pattern.pattern, pattern.flags)
            ignorecase = bool(parsed.state.flags & re.IGNORECASE)
            found: List[Requirement] = []
            _requirements(parsed, ignorecase, found)
            requirement = _ranked(found)
        except Exception:  # noqa: BLE001 - unparseable here means "always run the regex"
            requirement = None
    result = (requirement, ignorecase)
    _analysis[pattern] = result
    return result


class Text:
    """A text and its folded form, computed once and shared by every pattern."""

    __slots__ = ("raw", "_folded")

    def __init__(self, raw: str):
        self.raw = raw
        self._folded: Optional[str] = None

    @property
    def folded(self) -> str:
        if self._folded is None:
            self._folded = fold(self.raw)
        return self._folded


def may_match(pattern: re.Pattern, text: Text) -> bool:
    """False only when ``pattern`` provably cannot match ``text``."""
    requirements, ignorecase = analyse(pattern)
    if requirements is None:
        return True
    haystack = text.folded if ignorecase else text.raw
    return all(any(literal in haystack for literal in requirement) for requirement in requirements)


def search(pattern: re.Pattern, text: Text):
    """``pattern.search(text.raw)``, skipped when it provably cannot match."""
    if not may_match(pattern, text):
        return None
    return pattern.search(text.raw)


_word_patterns: Dict[str, re.Pattern] = {}


def word_pattern(word: str) -> re.Pattern:
    r"""``word``, case-insensitive, compiled once. Detectors build these for every
    keyword on every call; with thousands of keywords the ``re`` module's own cache
    (512 entries) thrashes and they were recompiled each time."""
    pattern = _word_patterns.get(word)
    if pattern is None:
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        if len(_word_patterns) < 100_000:
            _word_patterns[word] = pattern
    return pattern

