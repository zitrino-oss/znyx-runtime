"""Task adherence scorer - measures compliance with explicit instruction constraints.

Extracts format, length, and content constraints from the input and checks
whether the output satisfies them.
"""
import re
from typing import Dict, Any, List, Optional, Tuple

from app.shared.core.models import QualityScore


def _extract_constraints(text: str) -> List[Tuple[str, str, Any]]:
    """Extract (constraint_type, description, expected_value) from input text."""
    constraints = []
    lower = text.lower()

    # Format constraints
    for fmt, pattern in [
        ("json", r"\b(in json|as json|json format|json output)\b"),
        ("table", r"\b(as a table|in table format|table form)\b"),
        ("bullet_points", r"\b(in bullet points|as bullet points|bulleted list)\b"),
        ("numbered_list", r"\b(numbered list|as a numbered list|in a numbered list)\b"),
        ("markdown", r"\b(in markdown|as markdown|markdown format)\b"),
        ("csv", r"\b(in csv|as csv|csv format)\b"),
        ("code", r"\b(in code|as code|code block|code snippet)\b"),
    ]:
        if re.search(pattern, lower):
            constraints.append(("format", fmt, fmt))

    # Length constraints - word count
    m = re.search(r"(?:in|under|less than|at most|no more than)\s+(\d+)\s+words?\b", lower)
    if m:
        constraints.append(("max_words", f"max {m.group(1)} words", int(m.group(1))))
    m = re.search(r"(?:at least|minimum|no less than|more than)\s+(\d+)\s+words?\b", lower)
    if m:
        constraints.append(("min_words", f"min {m.group(1)} words", int(m.group(1))))

    # Length constraints - sentence count
    m = re.search(r"(?:in|exactly|using)\s+(\d+)\s+sentence", lower)
    if m:
        constraints.append(("sentence_count", f"{m.group(1)} sentences", int(m.group(1))))

    # Length constraints - item count
    m = re.search(r"(?:list|give|provide|name)\s+(\d+)\b", lower)
    if m:
        constraints.append(("min_items", f"at least {m.group(1)} items", int(m.group(1))))

    # Content constraints - include
    for m in re.finditer(r"(?:include|mention|must contain|make sure to mention)\s+[\"']?([^\"'\n,]+)[\"']?", lower):
        constraints.append(("include", f"include '{m.group(1).strip()}'", m.group(1).strip()))

    # Content constraints - exclude
    for m in re.finditer(r"(?:do not mention|don't mention|exclude|avoid mentioning|without mentioning)\s+[\"']?([^\"'\n,]+)[\"']?", lower):
        constraints.append(("exclude", f"exclude '{m.group(1).strip()}'", m.group(1).strip()))

    return constraints


def _check_format(output: str, fmt: str) -> bool:
    lower = output.strip()
    if fmt == "json":
        return lower.startswith("{") or lower.startswith("[") or "```json" in lower
    if fmt == "table":
        return "|" in output and "-" in output
    if fmt == "bullet_points":
        return bool(re.search(r"^\s*[-*]\s", output, re.M))
    if fmt == "numbered_list":
        return bool(re.search(r"^\s*\d+[.)]\s", output, re.M))
    if fmt == "markdown":
        return bool(re.search(r"(^#+\s|^\*\*|^```)", output, re.M))
    if fmt == "csv":
        lines = output.strip().split("\n")
        return len(lines) > 1 and all("," in line for line in lines[:3])
    if fmt == "code":
        return "```" in output or bool(re.search(r"^\s{4,}\S", output, re.M))
    return True


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return len(re.split(r"[.!?]+\s", text.strip()))


def _item_count(text: str) -> int:
    numbered = len(re.findall(r"^\s*\d+[.)]\s", text, re.M))
    bulleted = len(re.findall(r"^\s*[-*]\s", text, re.M))
    return max(numbered, bulleted)


def score_task_adherence(
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityScore:
    """Score task adherence of output against input constraints."""
    constraints = _extract_constraints(input_text)

    # Also check metadata for constraints
    if metadata and "constraints" in metadata:
        for c in metadata["constraints"]:
            if isinstance(c, dict):
                constraints.append((c.get("type", "custom"), c.get("description", ""), c.get("value")))

    if not constraints:
        return QualityScore(
            metric="task_adherence",
            score=1.0,
            details="No explicit constraints detected; score defaults to 1.0.",
        )

    satisfied = 0
    output_lower = output_text.lower()
    wc = _word_count(output_text)
    sc = _sentence_count(output_text)
    ic = _item_count(output_text)

    for ctype, desc, expected in constraints:
        if ctype == "format":
            if _check_format(output_text, expected):
                satisfied += 1
        elif ctype == "max_words":
            if wc <= expected:
                satisfied += 1
        elif ctype == "min_words":
            if wc >= expected:
                satisfied += 1
        elif ctype == "sentence_count":
            # Allow +/- 1 tolerance
            if abs(sc - expected) <= 1:
                satisfied += 1
        elif ctype == "min_items":
            if ic >= expected:
                satisfied += 1
        elif ctype == "include":
            if expected.lower() in output_lower:
                satisfied += 1
        elif ctype == "exclude":
            if expected.lower() not in output_lower:
                satisfied += 1
        else:
            satisfied += 0.5  # unknown constraint type, partial credit

    score = satisfied / len(constraints) if constraints else 1.0
    score = min(max(score, 0.0), 1.0)

    return QualityScore(
        metric="task_adherence",
        score=round(score, 3),
        details=f"{satisfied}/{len(constraints)} constraints satisfied.",
        sub_scores={
            "satisfied": satisfied,
            "total_constraints": len(constraints),
        },
    )
