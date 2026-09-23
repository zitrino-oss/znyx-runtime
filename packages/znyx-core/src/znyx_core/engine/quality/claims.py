"""Atomic-claim splitting for entailment-based grounding.

Sentence boundaries are not claim boundaries. NLI entailment is all-or-nothing over the
whole hypothesis, so a sentence that joins two facts drawn from two different parts of
the source is entailed by *neither* premise on its own and comes back ``neutral`` - i.e.
a truthful answer scores as ungrounded. Measured on a two-fact sentence whose halves
score 0.98 and 0.99 against their own source lines, the joined sentence scores 0.01.

Splitting on coordinators before scoring removes that failure. The split is deliberately
conservative - a wrong split invents a claim nobody made, which is worse than leaving a
compound one intact:

* comma/semicolon coordinators are split unconditionally (", and", "; ", ...) - the
  punctuation is already strong evidence of a clause boundary;
* a bare coordinator ("X and Y", no comma) is split only when BOTH sides are at least
  ``_BARE_MIN_WORDS`` words, so noun phrases ("black and white", "terms and conditions")
  stay whole;
* a fragment shorter than ``min_words`` is merged back into its predecessor rather than
  scored on its own;
* anything that does not yield two usable fragments returns the sentence unchanged.

Used by the NLI paths only. The deterministic token-overlap path keeps whole sentences,
because overlap is a ratio over the claim's own tokens and already degrades gracefully
on compound text - re-splitting there would change long-standing scores for no gain.
"""
from __future__ import annotations

import re
from typing import List

__all__ = ["split_atomic_claims"]

# ", and" / "; " style boundaries: punctuation already marks the clause split.
_PUNCT_COORDINATOR = re.compile(
    r",\s+(?:and|but|so|yet|while|whereas|although|though)\s+|;\s+",
    re.IGNORECASE,
)

# Bare "X and Y" - only honoured when both sides are substantial (see module docstring).
#
# "and" ONLY. A contrastive coordinator without a comma usually joins parts of one
# condition rather than two claims, and splitting it inverts the meaning: "reported after
# 2 business days but within 60 days, liability is capped at 500 USD" splits into
# "reported after 2 business days", which the source then contradicts (that window caps
# at 50 USD) - a true sentence scored as a hallucination. Contrastive forms are still
# split when a comma marks the boundary, which is the case they genuinely coordinate.
_BARE_COORDINATOR = re.compile(r"\s+and\s+", re.IGNORECASE)

# Minimum words on EACH side before a bare coordinator counts as a clause boundary.
_BARE_MIN_WORDS = 5

_WORD = re.compile(r"[A-Za-z0-9]+")


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _merge_short(fragments: List[str], min_words: int) -> List[str]:
    """Fold fragments below ``min_words`` back into the preceding one, so a split never
    manufactures a claim too small to judge."""
    merged: List[str] = []
    for frag in fragments:
        frag = frag.strip().strip(",;").strip()
        if not frag:
            continue
        if merged and _word_count(frag) < min_words:
            merged[-1] = f"{merged[-1]}, {frag}"
        else:
            merged.append(frag)
    # A leading short fragment has no predecessor to merge into; fold it forward instead.
    if len(merged) > 1 and _word_count(merged[0]) < min_words:
        merged[1] = f"{merged[0]}, {merged[1]}"
        merged.pop(0)
    return merged


def _split_bare(fragment: str) -> List[str]:
    """Split on a bare coordinator only where both sides carry enough words to stand as
    their own claim."""
    parts = _BARE_COORDINATOR.split(fragment)
    if len(parts) < 2:
        return [fragment]
    if any(_word_count(p) < _BARE_MIN_WORDS for p in parts):
        return [fragment]
    return parts


def split_atomic_claims(sentence: str, min_words: int = 3) -> List[str]:
    """Split one sentence into atomic claims, or return it unchanged.

    Returns a list with the original sentence when no confident boundary is found, so
    callers can use this unconditionally.
    """
    if not sentence or not sentence.strip():
        return []
    sentence = sentence.strip()

    fragments = _merge_short(_PUNCT_COORDINATOR.split(sentence), min_words)

    # Only reach for the riskier bare-coordinator split when punctuation found nothing.
    if len(fragments) < 2:
        fragments = _merge_short(_split_bare(sentence), min_words)

    if len(fragments) < 2:
        return [sentence]
    return fragments


# A cross-encoder scores one premise against one hypothesis, and its judgement blurs as
# the premise grows: the relevant line is diluted by everything around it. Measured on a
# 1160-char policy document, a truthful paraphrase scored 0.01 entailment against the
# whole document but 0.98 against the single passage that actually supported it.
#
# Callers take the best score across sources, so splitting a document into passages turns
# one blurred judgement into several sharp ones and keeps the strongest.
_MAX_PASSAGE_CHARS = 600
_MAX_PASSAGES = 12
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

__all__ = ["split_atomic_claims", "split_source_passages"]


def _pack_sentences(text: str, max_chars: int) -> List[str]:
    """Group sentences into passages just under ``max_chars``, so a long paragraph is
    divided on sentence boundaries rather than mid-claim."""
    out: List[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


def split_source_passages(source: str, max_chars: int = _MAX_PASSAGE_CHARS,
                          max_passages: int = _MAX_PASSAGES) -> List[str]:
    """Split one grounding source into passages a cross-encoder can judge sharply.

    Paragraph breaks first (they usually mark real section boundaries), then sentence
    packing for any paragraph still over ``max_chars``. Returns ``[source]`` unchanged
    when it is already small enough or yields nothing useful, so callers can use this
    unconditionally.

    Capped at ``max_passages``: each passage costs one scoring call, and past a dozen the
    latency outweighs the accuracy a further split buys. On overflow the tail is kept
    whole rather than dropped, so no source text stops being checked.
    """
    if not source or not source.strip():
        return []
    source = source.strip()
    if len(source) <= max_chars:
        return [source]

    passages: List[str] = []
    for para in _PARAGRAPH.split(source):
        para = para.strip()
        if not para:
            continue
        passages.extend([para] if len(para) <= max_chars else _pack_sentences(para, max_chars))

    if not passages:
        return [source]
    if len(passages) > max_passages:
        head = passages[: max_passages - 1]
        head.append("\n\n".join(passages[max_passages - 1:]))
        passages = head
    return passages
