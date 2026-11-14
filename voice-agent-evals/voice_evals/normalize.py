"""Text normalization shared by the Stage 1 (ASR) and Stage 3 (TTS round trip) scorers.

The ground truth and the prediction go through exactly the same steps, so a score only moves when
the words differ, not when the formatting does ("T L F twenty-two ten" and "TLF-2210" compare as
equal). Bump NORMALIZER_VERSION whenever a rule changes; the version is tagged on every trace.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

NORMALIZER_VERSION = "v2"  # v2: filler words ("um", "uh") are dropped before scoring

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicon_v1.csv"

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}
_ORDINALS = {
    "first": "one", "second": "two", "third": "three", "fourth": "four", "fifth": "five",
    "sixth": "six", "seventh": "seven", "eighth": "eight", "ninth": "nine", "tenth": "ten",
    "eleventh": "eleven", "twelfth": "twelve", "thirteenth": "thirteen",
    "fourteenth": "fourteen", "fifteenth": "fifteen", "sixteenth": "sixteen",
    "seventeenth": "seventeen", "eighteenth": "eighteen", "nineteenth": "nineteen",
    "twentieth": "twenty", "thirtieth": "thirty",
}
# Applied token by token after number conversion. Keep this list short and explicit.
_ABBREVIATIONS = {
    "mm": "millimeter", "millimeters": "millimeter", "millimetre": "millimeter",
    "millimetres": "millimeter", "g": "gram", "grams": "gram", "dollar": "dollars",
    "usd": "dollars", "&": "and", "okay": "ok",
}

_NUMBER_TOKEN = re.compile(r"^\d+(\.\d+)?$")


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    category: str
    canonical: str
    variants: tuple[tuple[str, ...], ...]


def canonical_term(term: str) -> str:
    """The single token a domain term collapses to, e.g. "TLF-2210" -> "tlf2210"."""
    return re.sub(r"[^a-z0-9]", "", term.lower())


def _clean(text: str) -> list[str]:
    """Steps 1-3: unicode, case, currency, punctuation. Returns raw tokens."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("'", "")
    # "$1,840.00" -> "1840 dollars"; thousands separators go first.
    while re.search(r"(\d),(\d{3})", text):
        text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"\$\s*(\d+(?:\.\d+)?)", r"\1 dollars", text)
    text = re.sub(r"(\d+)\.0+\b", r"\1", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"(\d)(mm|g)\b", r"\1 \2", text)  # "40g" -> "40 g", "4.5mm" -> "4.5 mm"
    # Keep a decimal point only between digits; every other punctuation mark becomes a space.
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^a-z0-9.& ]", " ", text)
    return text.split()


def _words_to_numbers(tokens: list[str]) -> list[str]:
    """Step 4: spoken numbers to digits. "twenty two" -> "22", "four point five" -> "4.5".

    A new number starts whenever the next word can't extend the current one, so
    "fifty two twelve" -> "52 12" and "five eight two" -> "5 8 2" (joined later).
    """
    out: list[str] = []
    total = current = 0
    last = None  # "ones", "tens", "scale" or None when no number is open
    decimals: list[str] | None = None

    def flush() -> None:
        nonlocal total, current, last, decimals
        if last is None:
            return
        value = str(total + current)
        if decimals:
            value += "." + "".join(decimals)
        out.append(value)
        total = current = 0
        last = None
        decimals = None

    i = 0
    while i < len(tokens):
        tok = _ORDINALS.get(tokens[i], tokens[i])
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if decimals is not None:
            if tok in _ONES and _ONES[tok] < 10:
                decimals.append(str(_ONES[tok]))
                i += 1
                continue
            flush()
        if tok in _ONES:
            value = _ONES[tok]
            if last == "tens" and value < 10:
                current += value
                last = "ones"
            elif last == "scale":
                current += value
                last = "ones"
            else:
                flush()
                current = value
                last = "ones"
        elif tok in _TENS:
            if last == "scale":
                current += _TENS[tok]
            else:
                flush()
                current = _TENS[tok]
            last = "tens"
        elif tok in _SCALES and last is not None:
            if tok == "hundred":
                current = (current or 1) * 100
            else:
                total += (current or 1) * 1000
                current = 0
            last = "scale"
        elif tok == "point" and last is not None and nxt in _ONES and _ONES[nxt] < 10:
            decimals = []
        elif tok == "and" and last == "scale" and nxt is not None and (nxt in _ONES or nxt in _TENS):
            pass
        else:
            flush()
            out.append(tok)
        i += 1
    flush()
    return out


def _join_runs(tokens: list[str]) -> list[str]:
    """Steps 5-6: "t l f" -> "tlf" and "6 0 7 4 4" -> "60744"."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                j += 1
            if j - i >= 2:
                out.append("".join(tokens[i:j]))
                i = j
                continue
        elif len(tokens[i]) == 1 and tokens[i].isdigit():
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isdigit():
                j += 1
            if j - i >= 3:
                out.append("".join(tokens[i:j]))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return out


# Closed compounds that are also written open or hyphenated ("back-ordered" -> "back ordered").
_COMPOUNDS = {"backordered": ["back", "ordered"], "backorder": ["back", "order"]}

# Hesitations carry no content, and transcribers differ on whether they write them, so neither side
# is scored on them. Dropped before numbers are joined: "five eight, uh, two" is 582. Not "mm",
# which is millimeters here.
_FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "hmm"}


def _base_tokens(text: str) -> list[str]:
    tokens = _join_runs(_words_to_numbers([t for t in _clean(text) if t not in _FILLERS]))
    out: list[str] = []
    for t in tokens:
        out.extend(_COMPOUNDS.get(t, [_ABBREVIATIONS.get(t, t)]))
    return out


@lru_cache(maxsize=4)
def load_lexicon(path: str | Path = DEFAULT_LEXICON_PATH) -> tuple[LexiconEntry, ...]:
    entries = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = [row["term"]] + [v for v in row["variants"].split("|") if v.strip()]
            variants = {tuple(_base_tokens(v)) for v in raw}
            entries.append(
                LexiconEntry(
                    term=row["term"],
                    category=row["category"],
                    canonical=canonical_term(row["term"]),
                    variants=tuple(sorted(variants, key=len, reverse=True)),
                )
            )
    return tuple(entries)


def _apply_lexicon(tokens: list[str], lexicon: tuple[LexiconEntry, ...]) -> list[str]:
    """Step 7: replace any lexicon variant with its canonical token, longest match first."""
    patterns = sorted(
        ((v, e.canonical) for e in lexicon for v in e.variants if v),
        key=lambda p: len(p[0]),
        reverse=True,
    )
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for variant, canonical in patterns:
            if tuple(tokens[i : i + len(variant)]) == variant:
                out.append(canonical)
                i += len(variant)
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def normalize_tokens(text: str, lexicon: tuple[LexiconEntry, ...] | None = None) -> list[str]:
    lexicon = load_lexicon() if lexicon is None else lexicon
    return _apply_lexicon(_base_tokens(text or ""), lexicon)


def normalize(text: str, lexicon: tuple[LexiconEntry, ...] | None = None) -> str:
    return " ".join(normalize_tokens(text, lexicon))


def _edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after normalization (same definition as jiwer.wer)."""
    ref, hyp = normalize_tokens(reference), normalize_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(normalize(reference)), list(normalize(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def numbers(text: str) -> list[str]:
    """Numeric tokens left after normalization, e.g. "$1,840 for size fourteen" -> ["1840", "14"]."""
    return [t for t in normalize_tokens(text) if _NUMBER_TOKEN.match(t)]


def terms_in(text: str, lexicon: tuple[LexiconEntry, ...] | None = None) -> list[str]:
    """Lexicon terms (by display name) that appear in the text."""
    lexicon = load_lexicon() if lexicon is None else lexicon
    found = set(normalize_tokens(text, lexicon))
    return [e.term for e in lexicon if e.canonical in found]
