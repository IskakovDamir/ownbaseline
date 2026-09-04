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
import re
import sys

# The accent: a bright violet that stays legible on light and dark terminals.
ACCENT_RGB = (168, 85, 247)      # #A855F7
ACCENT_256 = 141                 # #af87ff, the nearest xterm-256 neighbour

# The second colour, used once, on the author line beside the name. Volt, the
# neon lime that carries the rest of this author's work.
VOLT_RGB = (199, 242, 60)        # #C7F23C
VOLT_256 = 191                   # #d7ff5f, the nearest xterm-256 neighbour

_ESC = "\x1b["


def _depth():
    force = os.environ.get("OWNBASELINE_COLOR")
    if os.environ.get("FORCE_COLOR") and force is None:
        force = "0" if os.environ["FORCE_COLOR"] == "0" else "1"
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


def _volt_seq():
    if DEPTH == 24:
        r, g, b = VOLT_RGB
        return f"{_ESC}38;2;{r};{g};{b}m"
    if DEPTH == 8:
        return f"{_ESC}38;5;{VOLT_256}m"
    if DEPTH == 4:
        return f"{_ESC}92m"          # bright green
    return ""


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
_VOLT = _volt_seq()
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
def volt(s):       return f"{_VOLT}{_BOLD}{s}{RESET}" if DEPTH else s


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


# ------------------------------------------------------------------- splash
#
# A block alphabet, five rows, drawn here rather than pulled from a font
# package so the splash renders identically everywhere and adds no dependency.
# Only the nine letters the name needs exist.

_GLYPHS = {
    "O": ("█████", "█   █", "█   █", "█   █", "█████"),
    "W": ("█   █", "█   █", "█ █ █", "██ ██", "█   █"),
    "N": ("█   █", "██  █", "█ █ █", "█  ██", "█   █"),
    "B": ("████ ", "█   █", "████ ", "█   █", "████ "),
    "A": (" ███ ", "█   █", "█████", "█   █", "█   █"),
    "S": ("█████", "█    ", "█████", "    █", "█████"),
    "E": ("█████", "█    ", "████ ", "█    ", "█████"),
    "L": ("█    ", "█    ", "█    ", "█    ", "█████"),
    "I": ("█████", "  █  ", "  █  ", "  █  ", "█████"),
    "D": ("████ ", "█   █", "█   █", "█   █", "████ "),
    "M": ("█   █", "██ ██", "█ █ █", "█   █", "█   █"),
    "R": ("████ ", "█   █", "████ ", "█  █ ", "█   █"),
    "K": ("█   █", "█  █ ", "███  ", "█  █ ", "█   █"),
    "V": ("█   █", "█   █", "█   █", " █ █ ", "  █  "),
    ".": ("  ", "  ", "  ", "  ", "██"),
}
GLYPH_W = 5


def _w(ch):
    return len(_GLYPHS[ch][0])


def _block(word):
    rows = []
    for r in range(5):
        rows.append(" ".join(_GLYPHS[c][r] for c in word))
    return rows


def block_width(word):
    return sum(_w(c) for c in word) + (len(word) - 1)


def _block_fit(word, width):
    """
    The same block letters, tracked out so the word ends exactly at `width`.

    The slack is spread over the gaps between letters, the leftover columns
    going to the leftmost gaps, so two words of different lengths line up on
    both edges instead of only on the left.
    """
    ink = sum(_w(c) for c in word)
    gaps = len(word) - 1
    if gaps <= 0 or width < ink:
        return _block(word)          # nothing to distribute, or it will not fit
    slack = width - ink
    base, extra = divmod(slack, gaps)
    seps = [" " * (base + (1 if i < extra else 0)) for i in range(gaps)]
    rows = []
    for r in range(5):
        line = _GLYPHS[word[0]][r]
        for i, c in enumerate(word[1:]):
            line += seps[i] + _GLYPHS[c][r]
        rows.append(line)
    return rows


def _box(styled, plain, pad=1):
    """Border width comes from the printable text, never from the styled one."""
    w = len(plain) + 2 * pad
    return (dim("╭" + "─" * w + "╮"),
            dim("│") + " " * pad + styled + " " * pad + dim("│"),
            dim("╰" + "─" * w + "╯"))


def _monogram(author):
    """Initials with stops, as a block word: ADA LOVELACE -> A.L."""
    letters = [w[0].upper() for w in (author or "").split() if w]
    word = "".join(f"{c}." for c in letters)
    return word if word and all(c in _GLYPHS for c in word) else ""


def splash(version, author, tagline="does your score beat its own baseline?"):
    """
    The full opening screen, shown when the tool is run with no arguments.

    Two bands. The first carries OWN on the left and the author's initials on
    the right, both in the same face, the initials in volt; at five letters and
    four they happen to set to the same width, so the band is symmetrical
    without anything being padded to make it so. The second carries BASELINE
    across the full width. The name itself is one quiet line under the mark,
    flush with its right edge, which is the only part of this that survives a
    narrow terminal intact.

    Deliberately not printed by `check`: a banner a user sees on every run of a
    long audit stops introducing anything and becomes noise, so `check` gets the
    single line from banner() instead.
    """
    plain = f"✳ own-baseline  {tagline}"
    styled = f"{accent('✳')} {bold('own-baseline')}  {dim(tagline)}"
    out = list(_box(styled, plain))
    out.append("")

    W = block_width("BASELINE")
    mono = _monogram(author)
    mono_rows = _block(mono) if mono else None
    lead = W - block_width("OWN") - (block_width(mono) if mono else 0)

    for i, row in enumerate(_block("OWN")):
        line = "  " + accent(row)
        if mono_rows and lead > 0:
            line += " " * lead + volt(mono_rows[i])
        out.append(line)
    for row in _block("BASELINE"):
        out.append("  " + accent(row))

    if author:
        out.append("  " + " " * max(0, W - len(author)) + volt(author))
    out.append("")
    out.append("  " + dim(f"v{version}  ·  MIT"))
    return "\n".join(out)


# ------------------------------------------------------------------- frame

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def printable(s):
    """Width on screen: the string with its escape sequences taken out."""
    return len(_ANSI.sub("", s))


def terminal_width(default=100):
    try:
        return os.get_terminal_size().columns
    except OSError:
        return int(os.environ.get("COLUMNS", default))


def frame(lines, title=None):
    """
    Draw a rule around `lines`, in the accent colour.

    The width comes from the printable length of the content, never from the
    styled length, and the frame is dropped entirely when the content is wider
    than the terminal. A box that wraps is worse than no box, and a reader whose
    window is narrow should still get the numbers.
    """
    body = [l.rstrip() for l in lines]
    inner = max((printable(l) for l in body), default=0)
    if inner == 0:
        return body
    if inner + 4 > terminal_width():
        return body                       # no room; give the content plainly
    top = "╭─" + "─" * inner + "─╮"
    if title:
        t = f" {title} "
        if printable(t) + 4 <= inner:
            top = "╭─" + t + "─" * (inner - printable(t)) + "─╮"
    out = [accent(top)]
    for l in body:
        pad = " " * (inner - printable(l))
        out.append(accent("│") + " " + l + pad + " " + accent("│"))
    out.append(accent("╰─" + "─" * inner + "─╯"))
    return out


# ---------------------------------------------------------------- progress

class Progress:
    """
    A single line on stderr, rewritten in place, for work that takes a while.

    The audit's expensive part is the bootstrap: a thousand resamples of a
    rank-residual fit on forty thousand cells is a minute of arithmetic, and
    until now that minute printed nothing at all. This says what is being
    computed and how far along it is, on stderr so it never lands in piped
    output, and switches itself off when stderr is not a terminal, when colour
    is off, or when the caller asks for quiet.
    """

    def __init__(self, enabled=True, stream=None):
        self.stream = stream or sys.stderr
        self.enabled = bool(enabled) and self.stream.isatty()
        self._width = 0

    def update(self, text):
        if not self.enabled:
            return
        line = text[: max(20, terminal_width() - 2)]
        pad = " " * max(0, self._width - printable(line))
        self.stream.write("\r" + line + pad)
        self.stream.flush()
        self._width = printable(line)

    def done(self, text=None):
        if not self.enabled:
            return
        self.stream.write("\r" + " " * self._width + "\r")
        if text:
            self.stream.write(text + "\n")
        self.stream.flush()
        self._width = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.done()
        return False
