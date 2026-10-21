"""Computable claims in an answer, recomputed exactly (technique 18).

Beyond the ``a op b = c`` equations of the ``numerical_consistency`` detector: sums and
differences of several terms, percentages ("15% of 80 is 12"), unit conversions
("5 km is 3.1 miles") and date arithmetic ("from 2026-01-01 to 2026-03-01 is 59 days",
"30 days after 2026-01-01 is 2026-01-31"). Only claims stated in one of these literal
forms are read, so prose cannot trigger a finding; a stated value is accepted within the
precision it is written with (or about 2% when it says "about", "roughly",
"approximately").

Pure rules, no dependencies; returns what is wrong and the correct value, so a caller
can ask the model to correct it or flag the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterator, List, Optional, Tuple

# A number, never started inside another one (the lookbehind) and bounded, so a long run of
# digits or terms is read in one pass instead of once from every position.
_NUM = r"(?<![\d.,$])[-+]?\$?\d{1,15}(?:,\d{3}){0,5}(?:\.\d{1,10})?(?![\d])"
_IS = r"(?:is|=|equals|equal to|makes|comes to|gives)"
_APPROX = r"(?:about|roughly|approximately|around|nearly|~|≈)"

_EQUATION = re.compile(rf"(?P<a>{_NUM}%?)\s*(?P<op>[-+x*×/÷])\s*(?P<b>{_NUM}%?)\s*=\s*(?P<c>{_NUM}%?)", re.I)
# A chain starts only where one begins: not right after an operator (else every term of a
# long chain would start a new attempt).
_SUM = re.compile(rf"(?<![-+*/×÷])(?<![-+*/×÷]\s)(?P<terms>{_NUM}%?(?:\s{{0,3}}[-+]\s{{0,3}}{_NUM}%?){{2,30}})\s{{0,3}}=\s{{0,3}}(?P<c>{_NUM}%?)")
_PERCENT_OF = re.compile(
    rf"(?P<p>(?<![\d.])\d{{1,6}}(?:\.\d{{1,6}})?)\s*(?:%|percent)\s+of\s+(?P<base>{_NUM})\s+{_IS}\s+(?P<approx>{_APPROX}\s+)?(?P<c>{_NUM})",
    re.I)
_IS_PERCENT = re.compile(
    rf"(?P<part>{_NUM})\s+is\s+(?P<approx>{_APPROX}\s+)?(?P<p>(?<![\d.])\d{{1,6}}(?:\.\d{{1,6}})?)\s*(?:%|percent)\s+of\s+(?P<base>{_NUM})",
    re.I)

# unit -> (dimension, factor to the base unit); temperatures are handled apart.
_UNITS = {
    "km": ("length", 1000.0), "kilometer": ("length", 1000.0), "kilometre": ("length", 1000.0),
    "m": ("length", 1.0), "meter": ("length", 1.0), "metre": ("length", 1.0),
    "cm": ("length", 0.01), "centimeter": ("length", 0.01), "centimetre": ("length", 0.01),
    "mm": ("length", 0.001), "mi": ("length", 1609.344), "mile": ("length", 1609.344),
    "ft": ("length", 0.3048), "foot": ("length", 0.3048), "feet": ("length", 0.3048),
    "in": ("length", 0.0254), "inch": ("length", 0.0254), "inches": ("length", 0.0254),
    "yd": ("length", 0.9144), "yard": ("length", 0.9144),
    "kg": ("mass", 1.0), "kilogram": ("mass", 1.0), "g": ("mass", 0.001), "gram": ("mass", 0.001),
    "lb": ("mass", 0.45359237), "lbs": ("mass", 0.45359237), "pound": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125), "ounce": ("mass", 0.028349523125),
    "l": ("volume", 1.0), "liter": ("volume", 1.0), "litre": ("volume", 1.0),
    "ml": ("volume", 0.001), "gal": ("volume", 3.785411784), "gallon": ("volume", 3.785411784),
    "c": ("temp", 0.0), "celsius": ("temp", 0.0), "°c": ("temp", 0.0),
    "f": ("temp", 0.0), "fahrenheit": ("temp", 0.0), "°f": ("temp", 0.0),
}
_UNIT = r"(?:°\s?[cf]|[a-zA-Z]{1,12})"
_CONVERSION = re.compile(
    rf"(?P<a>{_NUM})\s*(?P<ua>{_UNIT})\.?\s+{_IS}\s+(?P<approx>{_APPROX}\s+)?(?P<c>{_NUM})\s*(?P<uc>{_UNIT})\b",
    re.I)

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
_DATE = (r"(?:\d{4}-\d{2}-\d{2}|(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]{0,6}"
         r"\.?(?:\s+\d{1,2})?,?\s+\d{4})")
_SPAN = re.compile(rf"from\s+(?P<d1>{_DATE})\s+(?:to|until|through)\s+(?P<d2>{_DATE})\s+{_IS}\s+(?P<n>(?<!\d)\d{{1,6}})\s+days",
                   re.I)
_AFTER = re.compile(rf"(?P<n>(?<!\d)\d{{1,6}})\s+days\s+(?P<dir>after|before)\s+(?P<d1>{_DATE})\s+{_IS}\s+(?P<d2>{_DATE})", re.I)


@dataclass
class WrongClaim:
    kind: str          # equation | sum | percent | unit | date
    text: str          # the claim as written (bounded)
    correct: str       # the correct value

    def describe(self) -> str:
        return f'"{self.text}" is wrong: the correct value is {self.correct}'


def _number(token: str) -> Optional[float]:
    try:
        return float(token.replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


def _decimals(token: str) -> int:
    token = token.replace(",", "").replace("%", "")
    return len(token.split(".", 1)[1]) if "." in token else 0


def _wrong(expected: float, stated_token: str, approximate: bool) -> bool:
    stated = _number(stated_token)
    if stated is None:
        return False
    if approximate:
        return abs(expected - stated) > max(0.02 * abs(expected), 0.5 * 10 ** -_decimals(stated_token))
    # Within the precision the value is written with ("3.1" stands for 3.05 to 3.15).
    return abs(expected - stated) > max(0.5 * 10 ** -_decimals(stated_token), 1e-3 * abs(expected), 1e-9)


def _fmt(value: float, like: str = "") -> str:
    places = max(_decimals(like), 2 if abs(value) < 100 and value != int(value) else 0)
    text = f"{value:,.{places}f}"
    return text


def _preceded_by_operator(text: str, start: int) -> bool:
    """Whether a match starts in the middle of a longer arithmetic expression ("2 * 3 + 4 = 10"
    must not be read as "3 + 4 = 10"). A "=" before it does not count: "x = 2 + 2 = 5" still
    claims 2 + 2 = 5. Reads only a few characters back."""
    before = text[max(0, start - 8):start].rstrip()
    if not before:
        return False
    return before[-1] in "+-*/×÷" or before.lower().endswith(" x")


def _unit(token: str) -> Optional[Tuple[str, float, str]]:
    raw = token.lower().replace(" ", "")
    for name in (raw, raw.rstrip("s") if len(raw) > 3 else raw, raw[:-2] if raw.endswith("es") else raw):
        if name in _UNITS:
            dim, factor = _UNITS[name]
            canonical = "c" if name in ("c", "celsius", "°c") else "f" if name in ("f", "fahrenheit", "°f") else name
            return dim, factor, canonical
    return None


def _convert(value: float, src: Tuple[str, float, str], dst: Tuple[str, float, str]) -> Optional[float]:
    if src[0] != dst[0]:
        return None
    if src[0] == "temp":
        if src[2] == dst[2]:
            return value
        return value * 9 / 5 + 32 if src[2] == "c" else (value - 32) * 5 / 9
    return value * src[1] / dst[1]


def _date(token: str) -> Optional[date]:
    token = token.strip().rstrip(",")
    try:
        return datetime.strptime(token, "%Y-%m-%d").date()
    except ValueError:
        pass
    words = re.sub(r"[,.]", " ", token.lower()).split()
    day = month = year = None
    for word in words:
        if word.isdigit():
            if len(word) == 4:
                year = int(word)
            elif day is None:
                day = int(word)
        else:
            for name, number in _MONTHS.items():
                if name.startswith(word[:3]) and len(word) >= 3:
                    month = number
                    break
    if not (day and month and year):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _claims(text: str) -> Iterator[WrongClaim]:
    taken: List[Tuple[int, int]] = []

    def free(span: Tuple[int, int]) -> bool:
        return not any(span[0] < end and start < span[1] for start, end in taken)

    for m in _SUM.finditer(text):
        terms = re.findall(rf"([-+]?)\s*({_NUM}%?)", m.group("terms"))
        total, first = 0.0, True
        for sign, token in terms:
            value = _number(token)
            if value is None:
                break
            total += -value if sign == "-" and not first else value
            first = False
        else:
            taken.append(m.span())
            if _wrong(total, m.group("c"), False):
                yield WrongClaim("sum", m.group(0)[:120], _fmt(total, m.group("c")))
    for m in _EQUATION.finditer(text):
        if not free(m.span()) or _preceded_by_operator(text, m.start()):
            continue
        a, b = _number(m.group("a")), _number(m.group("b"))
        op = m.group("op").lower()
        if a is None or b is None:
            continue
        value = {"+": a + b, "-": a - b, "x": a * b, "*": a * b, "×": a * b}.get(op)
        if op in ("/", "÷"):
            value = a / b if b else None
        if value is None:
            continue
        taken.append(m.span())
        if _wrong(value, m.group("c"), False):
            yield WrongClaim("equation", m.group(0)[:120], _fmt(value, m.group("c")))
    for m in _PERCENT_OF.finditer(text):
        p, base = _number(m.group("p")), _number(m.group("base"))
        if p is None or base is None or not free(m.span()):
            continue
        value = p / 100 * base
        if _wrong(value, m.group("c"), bool(m.group("approx"))):
            yield WrongClaim("percent", m.group(0)[:120], _fmt(value, m.group("c")))
    for m in _IS_PERCENT.finditer(text):
        part, p, base = _number(m.group("part")), _number(m.group("p")), _number(m.group("base"))
        if part is None or p is None or not base or not free(m.span()):
            continue
        value = part / base * 100
        if _wrong(value, m.group("p"), bool(m.group("approx"))):
            yield WrongClaim("percent", m.group(0)[:120], _fmt(value, m.group("p")) + "%")
    for m in _CONVERSION.finditer(text):
        src, dst = _unit(m.group("ua")), _unit(m.group("uc"))
        value = _number(m.group("a"))
        if src is None or dst is None or value is None:
            continue
        converted = _convert(value, src, dst)
        if converted is not None and _wrong(converted, m.group("c"), bool(m.group("approx"))):
            yield WrongClaim("unit", m.group(0)[:120], f"{_fmt(converted, m.group('c'))} {m.group('uc')}")
    for m in _SPAN.finditer(text):
        d1, d2 = _date(m.group("d1")), _date(m.group("d2"))
        if d1 is None or d2 is None:
            continue
        days = (d2 - d1).days
        # "from the 1st to the 31st" is 30 days, or 31 counting both ends: either is right.
        if int(m.group("n")) not in (days, days + 1):
            yield WrongClaim("date", m.group(0)[:120], f"{days} days")
    for m in _AFTER.finditer(text):
        d1, d2 = _date(m.group("d1")), _date(m.group("d2"))
        if d1 is None or d2 is None:
            continue
        n = int(m.group("n"))
        expected = d1 + timedelta(days=n if m.group("dir").lower() == "after" else -n)
        if expected != d2:
            yield WrongClaim("date", m.group(0)[:120], expected.isoformat())


MAX_SCAN_CHARS = 20_000


def find_wrong_claims(text: str, limit: int = 10, kinds: Optional[Tuple[str, ...]] = None) -> List[WrongClaim]:
    """The computable claims in ``text`` whose stated value is wrong (at most ``limit``;
    only ``kinds`` when given). Reads the first ``MAX_SCAN_CHARS`` characters."""
    if not text:
        return []
    out: List[WrongClaim] = []
    for claim in _claims(text[:MAX_SCAN_CHARS]):
        if kinds is not None and claim.kind not in kinds:
            continue
        out.append(claim)
        if len(out) >= limit:
            break
    return out
