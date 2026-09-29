"""Console output must never die on a non-UTF-8 terminal.

A Windows/CI console is cp1252 (older ones cp437). Printing the welcome
banner's box-drawing characters there raised UnicodeEncodeError inside the
FastAPI lifespan, which aborted startup outright ("Application startup
failed"). The benchmark report carries the same hazard in its "Wrote scorecard
JSON -> out.json" arrow.
"""
import io
import logging

import pytest

from znyx_runtime.console import (
    SafeStreamHandler,
    safe_print,
    stream_supports,
    to_ascii,
)
from znyx_runtime.main import _build_welcome_banner

BOX = "║"      # in neither cp1252 nor cp437
ARROW = "→"    # in neither
EM_DASH = "—"  # in cp1252, NOT in cp437


def console(encoding: str) -> io.TextIOWrapper:
    """A stdout that encodes exactly like a legacy Windows console."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, newline="")


def read_back(stream: io.TextIOWrapper, encoding: str = "cp1252") -> str:
    stream.flush()
    return stream.buffer.getvalue().decode(encoding)


# cp437 (the old DOS codepage) carries box-drawing natively, so only cp1252 and
# ascii actually force the fallback - all three must simply not raise.
@pytest.mark.parametrize("encoding", ["cp1252", "cp437", "ascii"])
def test_banner_survives_a_legacy_console(encoding):
    stream = console(encoding)
    safe_print(_build_welcome_banner("1.2.3", "https://console.example"), file=stream)

    out = read_back(stream, encoding)
    assert "Welcome to ZNYX AI Runtime v1.2.3" in out
    assert "https://console.example" in out


def test_degraded_banner_keeps_its_box_aligned():
    """Box glyphs degrade 1:1, so the fixed-width padding still lines up."""
    stream = console("cp1252")
    safe_print(_build_welcome_banner("1.2.3", ""), file=stream)

    rows = read_back(stream).strip().splitlines()
    assert len(set(len(r) for r in rows)) == 1
    assert rows[0].startswith("+") and rows[0].endswith("+")
    assert all(r.startswith(("+", "|")) for r in rows)


def test_utf8_console_still_gets_the_real_glyphs():
    stream = console("utf-8")
    safe_print(_build_welcome_banner("1.2.3", ""), file=stream)

    out = read_back(stream, "utf-8")
    assert "╔" in out and BOX in out


def test_benchmark_report_arrow_is_printable_on_cp1252():
    stream = console("cp1252")
    safe_print(f"\nWrote scorecard JSON {ARROW} out.json", file=stream)

    assert "Wrote scorecard JSON -> out.json" in read_back(stream)


def test_a_line_the_stream_can_carry_is_left_alone():
    """cp1252 has an em dash, so the report's em dash must NOT be flattened."""
    stream = console("cp1252")
    safe_print(f"Scorecard {EM_DASH} detector 'toxicity'", file=stream)

    assert f"Scorecard {EM_DASH} detector 'toxicity'" in read_back(stream)


@pytest.mark.parametrize("text, expected", [
    (f"Scorecard {EM_DASH} detector 'x'", "Scorecard - detector 'x'"),
    (f"Wrote JSON {ARROW} out.json", "Wrote JSON -> out.json"),
    ("at least ≥ 100 samples", "at least >= 100 samples"),
    ("✓ unmapped glyph", "? unmapped glyph"),
])
def test_to_ascii_substitutions(text, expected):
    assert to_ascii(text) == expected


def test_stream_supports():
    assert stream_supports("plain", console("cp1252"))
    assert not stream_supports(BOX, console("cp1252"))
    assert stream_supports(EM_DASH, console("cp1252"))
    assert not stream_supports(EM_DASH, console("cp437"))
    assert stream_supports(BOX, io.StringIO())  # no declared encoding


def test_log_handler_degrades_instead_of_erroring():
    stream = console("cp437")
    handler = SafeStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("znyx.test.console")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info("escalation: toxicity %s escalating to llm", EM_DASH)

    out = read_back(stream, "cp437")
    assert "escalation: toxicity - escalating to llm" in out
    assert "Logging error" not in out
