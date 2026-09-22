"""ASCII banner printed at the start of a run.

The banner is decoration. It goes to the standard error stream, next to the progress
messages, so that redirecting the standard output never captures it. It is skipped when
the terminal is too narrow for the wordmark, and when the environment variable
``TQTOY_NO_BANNER`` is set to a non-empty value.
"""

from __future__ import annotations

import os
import shutil
import sys

# Five-row block font. Every row of a glyph has the same width, so the glyphs of a word
# concatenate column by column. Only the characters of the wordmarks below are defined.
_FONT = {
    "T": (" _____ ",
          "|_   _|",
          "  | |  ",
          "  | |  ",
          "  |_|  "),
    "Q": ("  ___  ",
          " / _ \\ ",
          "| | | |",
          "| |_| |",
          " \\__\\_\\"),
    "O": ("  ___  ",
          " / _ \\ ",
          "| | | |",
          "| |_| |",
          " \\___/ "),
    "Y": ("__   __",
          "\\ \\ / /",
          " \\ V / ",
          "  | |  ",
          "  |_|  "),
    "F": (" _____ ",
          "|  ___|",
          "| |_   ",
          "|  _|  ",
          "|_|    "),
    "H": (" _   _ ",
          "| | | |",
          "| |_| |",
          "|  _  |",
          "|_| |_|"),
    "E": (" _____ ",
          "| ____|",
          "|  _|  ",
          "| |___ ",
          "|_____|"),
    "-": ("     ",
          "     ",
          " ___ ",
          "|___|",
          "     "),
}

HEIGHT = 5


def wordmark(text: str) -> str:
    """The text drawn in the block font, as five lines.

    Raises KeyError on a character the font does not define, which is a programming error
    rather than a user input: the wordmarks are fixed.
    """
    rows = [_FONT[c.upper()] for c in text]
    return "\n".join("".join(glyph[i] for glyph in rows) for i in range(HEIGHT))


def width(text: str) -> int:
    """Number of columns the wordmark occupies."""
    return sum(len(_FONT[c.upper()][0]) for c in text)


def print_banner(text: str, subtitle: str = "", stream=None) -> bool:
    """Print the banner and say whether the drawn wordmark was used.

    Falls back to a single line when the terminal is narrower than the wordmark, and
    prints nothing at all when TQTOY_NO_BANNER is set.
    """
    if os.environ.get("TQTOY_NO_BANNER"):
        return False
    stream = sys.stderr if stream is None else stream
    art_width = width(text)
    columns = shutil.get_terminal_size((80, 24)).columns
    if columns < art_width:
        print(f"{text} {subtitle}".rstrip(), file=stream)
        return False
    print(wordmark(text), file=stream)
    if subtitle:
        print(subtitle.rjust(art_width), file=stream)
    print(file=stream)
    return True
