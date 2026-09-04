"""
Terminal styling for the CLI. One accent colour, used for structure only.

Numbers are never coloured. A reader should be able to pipe the output into a
file, read it with no colour at all, and lose nothing but the scannability: the
accent marks where a block begins and what it is called, and the verdict words
carry weight rather than hue. Everything degrades in this order: 24-bit, then
256, then the eight-colour bright magenta, then nothing.

Colour is off when stdout is not a terminal, when NO_COLOR is set (no-color.org),
when TERM is dumb or unset, or when OWNBASELINE_COLOR=0. It is forced on with
OWNBASELINE_COLOR=1, which is what the tests use.
"""
from __future__ import annotations

import os
import sys

# The accent: a bright violet that stays legible on light and dark terminals.
ACCENT_RGB = (168, 85, 247)      # #A855F7
ACCENT_256 = 141                 # #af87ff, the nearest xterm-256 neighbour

_ESC = "\x1b["


def _depth():
    force = os.environ.get("OWNBASELINE_COLOR")
    if force == "0":
        return 0
    if os.environ.get("NO_COLOR") is not None and force != "1":
        return 0
    term = os.environ.get("TERM", "")
    if force != "1":
        if not sys.stdout.isatty():
            return 0
        if term in ("", "dumb"):
            return 0
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return 24
    if "256" in term or force == "1":
        return 8
    return 4


DEPTH = _depth()


def _accent_seq():
    if DEPTH == 24:
        r, g, b = ACCENT_RGB
        return f"{_ESC}38;2;{r};{g};{b}m"
    if DEPTH == 8:
        return f"{_ESC}38;5;{ACCENT_256}m"
    if DEPTH == 4:
        return f"{_ESC}95m"          # bright magenta
    return ""


RESET = f"{_ESC}0m" if DEPTH else ""
_ACCENT = _accent_seq()
_BOLD = f"{_ESC}1m" if DEPTH else ""
_DIM = f"{_ESC}2m" if DEPTH else ""
_YELLOW = f"{_ESC}33m" if DEPTH else ""
_RED = f"{_ESC}31m" if DEPTH else ""
_GREEN = f"{_ESC}32m" if DEPTH else ""


def accent(s):     return f"{_ACCENT}{s}{RESET}" if DEPTH else s
def bold(s):       return f"{_BOLD}{s}{RESET}" if DEPTH else s
def dim(s):        return f"{_DIM}{s}{RESET}" if DEPTH else s
def warn(s):       return f"{_YELLOW}{s}{RESET}" if DEPTH else s
def bad(s):        return f"{_RED}{s}{RESET}" if DEPTH else s
def good(s):       return f"{_GREEN}{s}{RESET}" if DEPTH else s
def head(s):       return f"{_ACCENT}{_BOLD}{s}{RESET}" if DEPTH else s


BAR = "▌"       # ▌ left half block, the accent mark on a heading


def banner(version, author=None, right=None):
    """
    One line. The bar and the tool name in the accent, everything else quiet.

        ▌ ownbaseline 0.2.0                                  A. N. Author
    """
    left = f"{accent(BAR)} {head('ownbaseline')} {dim(version)}"
    tail = right or author or ""
    if not tail:
        return left
    # pad on the printable width, which is the styled string minus its escapes
    printable = len(f"{BAR} ownbaseline {version}")
    pad = max(1, 72 - printable - len(tail))
    return f"{left}{' ' * pad}{dim(tail)}"


def rule(width=72):
    return dim("─" * width)


def section(title):
    return f"{accent(BAR)} {head(title)}"
