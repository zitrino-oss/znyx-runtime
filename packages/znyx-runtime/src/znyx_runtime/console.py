"""Console output that survives a non-UTF-8 terminal.

Windows consoles and CI runners default to cp1252, which cannot encode the
box-drawing characters in the welcome banner or the em dashes and arrows the
benchmark prints. Plain ``print()`` raises ``UnicodeEncodeError`` there, which
killed the runtime during startup:

    File "znyx_runtime/main.py", line 204, in lifespan
        print(_WELCOME_BANNER)
    UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-55

Rather than reconfigure the stream to UTF-8 (which renders as mojibake on a
legacy console), every user-facing line goes through :func:`safe_print`, which
degrades the handful of characters we actually emit to ASCII when - and only
when - the stream cannot carry them. Log records take the same route via
:class:`SafeStreamHandler`.
"""
import logging
import sys

# The non-ASCII characters this project emits, and the ASCII they degrade to.
# Box-drawing glyphs map 1:1 so the banner's fixed-width padding still lines up.
_ASCII_FALLBACKS = {
    "═": "=", "║": "|",                                    # box double
    "╔": "+", "╗": "+", "╚": "+", "╝": "+",      # box corners
    "─": "-", "│": "|",                                    # box single
    "—": "-", "–": "-", "•": "*",                     # dashes, bullet
    "→": "->", "←": "<-", "↔": "<->",                 # arrows
    "≥": ">=", "≤": "<=", "×": "x", "…": "...",
    "‘": "'", "’": "'", "“": '"', "”": '"',
}


def stream_supports(text: str, stream=None) -> bool:
    """True when `stream` can encode `text` as-is."""
    stream = stream if stream is not None else sys.stdout
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True  # no declared encoding (a StringIO, a capture buffer) - nothing to break
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def to_ascii(text: str) -> str:
    """Replace the characters we emit with ASCII stand-ins; '?' for anything else."""
    for char, replacement in _ASCII_FALLBACKS.items():
        text = text.replace(char, replacement)
    return text.encode("ascii", "replace").decode("ascii")


def safe_print(text: str = "", **kwargs) -> None:
    """``print`` that degrades to ASCII instead of raising on a cp1252 console."""
    text = str(text)
    stream = kwargs.get("file") or sys.stdout
    if not stream_supports(text, stream):
        text = to_ascii(text)
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        # The stream lied about what it can encode (wrapped/reconfigured streams do).
        print(to_ascii(text), **kwargs)


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that ASCII-degrades records the stream cannot encode.

    Without it a log line containing an em dash produces a '--- Logging error ---'
    traceback on every Windows console instead of the message.
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text if stream_supports(text, self.stream) else to_ascii(text)
