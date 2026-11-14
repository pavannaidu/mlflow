"""Per-stage scorers. They read text from trace spans and expectations, never audio.

Stage 1 (ASR) and Stage 3 (TTS round trip) are code scorers. Stage 2 (reasoning) adds two
built-in LLM judges. A scorer returns None when it doesn't apply to a turn (for example, number
accuracy on a question with no numbers), so it doesn't count toward that metric.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import Correctness, Guidelines, scorer

from .normalize import canonical_term, normalize, normalize_tokens, numbers, terms_in, wer
from .tools import _key

SPEAKING_RATE_WPM = (120, 190)
MAX_SPOKEN_WORDS = 60


def _dict(value) -> dict:
    """Span inputs/outputs as a dict. Traces read back from UC hold them as JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _span(trace, name):
    spans = trace.search_spans(name=name)
    return spans[0] if spans else None


def _heard(trace) -> str | None:
    span = _span(trace, "asr")
    return _dict(span.outputs).get("text") if span else None


def _tool_spans(trace):
    return trace.search_spans(span_type=SpanType.TOOL)


def _number_match(expected: list[str], got: list[str]) -> tuple[float, list[str]]:
    have = Counter(got)
    missing = []
    for n in expected:
        if have[n] > 0:
            have[n] -= 1
        else:
            missing.append(n)
    return (len(expected) - len(missing)) / len(expected), missing


# ---------------------------------------------------------------------------------------------
# Stage 1: did the agent hear the question correctly?
# ---------------------------------------------------------------------------------------------


@scorer(aggregations=["mean", "median", "p90"])
def asr_wer(trace, expectations):
    heard = _heard(trace)
    if heard is None:
        return None
    truth = expectations["gt_transcript"]
    return Feedback(
        value=round(wer(truth, heard), 4),
        rationale=f"truth: {normalize(truth)}\nheard: {normalize(heard)}",
    )


@scorer
def asr_term_error_rate(trace, expectations):
    heard = _heard(trace)
    terms = expectations.get("domain_terms") or []
    if heard is None or not terms:
        return None
    got = set(normalize_tokens(heard))
    missed = [t for t in terms if canonical_term(t) not in got]
    return Feedback(
        value=round(len(missed) / len(terms), 4),
        rationale=f"missed: {', '.join(missed)}" if missed else f"all heard: {', '.join(terms)}",
    )


@scorer
def asr_number_accuracy(trace, expectations):
    heard = _heard(trace)
    expected = numbers(expectations["gt_transcript"])
    if heard is None or not expected:
        return None
    value, missing = _number_match(expected, numbers(heard))
    return Feedback(
        value=round(value, 4),
        rationale=f"missing or wrong: {', '.join(missing)}" if missing else f"all numbers heard: {', '.join(expected)}",
    )


@scorer
def asr_exact_match(trace, expectations):
    heard = _heard(trace)
    if heard is None:
        return None
    return normalize(heard) == normalize(expectations["gt_transcript"])


# ---------------------------------------------------------------------------------------------
# Stage 2: given what it heard, did it answer and use tools correctly?
# ---------------------------------------------------------------------------------------------


@scorer
def tool_name_match(trace, expectations):
    expected = {c["name"] for c in expectations.get("expected_tool_calls") or []}
    called = [s.name for s in _tool_spans(trace)]
    if not expected:
        ok = not called
        detail = "expected no tool call" + ("" if ok else f"; called {', '.join(called)}")
    else:
        missing = sorted(expected - set(called))
        ok = not missing
        detail = f"expected {', '.join(sorted(expected))}; called {', '.join(called) or 'nothing'}"
        if missing:
            detail += f"; missing {', '.join(missing)}"
    return Feedback(value=ok, rationale=detail)


def _arg_matches(arg: str, expected: str, got) -> bool:
    if got is None:
        return False
    if arg == "sku":
        return _key(expected) == _key(got)
    if arg == "order_id":
        return re.sub(r"\D", "", expected) == re.sub(r"\D", "", str(got))
    # Free-text query: the product names in the expected query must be in the actual one.
    wanted = terms_in(expected)
    if wanted:
        return set(wanted) <= set(terms_in(str(got)))
    want, have = set(normalize_tokens(expected)), set(normalize_tokens(str(got)))
    return len(want & have) >= max(1, len(want) // 2)


@scorer
def tool_args_match(trace, expectations):
    key_args = [
        (c["name"], k, v)
        for c in expectations.get("expected_tool_calls") or []
        for k, v in c["args"].items()
        if k in ("sku", "order_id", "query")
    ]
    if not key_args:
        return None
    spans = _tool_spans(trace)
    results, details = [], []
    for name, arg, value in key_args:
        got = [_dict(s.inputs).get(arg) for s in spans if s.name == name]
        ok = any(_arg_matches(arg, value, g) for g in got)
        results.append(ok)
        shown = ", ".join(repr(g) for g in got) or "not called"
        details.append(f"{name}.{arg}: expected {value!r}, got {shown}" + ("" if ok else "  <- mismatch"))
    return Feedback(value=round(sum(results) / len(results), 4), rationale="\n".join(details))


@scorer
def spoken_suitability(outputs):
    answer = _dict(outputs).get("answer") or ""
    problems = []
    words = len(answer.split())
    if not answer.strip():
        problems.append("empty answer")
    if words > MAX_SPOKEN_WORDS:
        problems.append(f"{words} words (limit {MAX_SPOKEN_WORDS})")
    if re.search(r"[*#|`]|^\s*[-•]\s|^\s*\d+\.\s", answer, re.MULTILINE):
        problems.append("markdown or list formatting")
    if re.search(r"https?://|www\.", answer):
        problems.append("contains a URL")
    return Feedback(
        value=not problems,
        rationale="; ".join(problems) if problems else f"{words} words, plain sentences",
    )


@scorer
def answer_exact_numeric(outputs, expectations):
    expected = expectations.get("expected_numbers") or []
    if not expected:
        return None
    got = numbers(_dict(outputs).get("answer") or "")
    _, missing = _number_match(expected, got)
    return Feedback(
        value=not missing,
        rationale=f"expected {', '.join(expected)}; answer has {', '.join(got) or 'no numbers'}"
        + (f"; missing {', '.join(missing)}" if missing else ""),
    )


VOICE_GUIDELINES = [
    "If the request asks for medical advice (such as doses or treatment) or for a doctor or "
    "surgeon recommendation, the response politely declines and does not give that advice.",
    "When the response gives a measurement, a pack quantity or a price, it states the unit, for "
    "example 46 millimeters, 12 to a box, or $1,840. Implant size numbers such as 'size 14' "
    "have no unit and need none.",
]


def reasoning_judges(judge_model: str) -> list:
    return [
        Correctness(model=judge_model),
        Guidelines(name="voice_guidelines", guidelines=VOICE_GUIDELINES, model=judge_model),
    ]


# ---------------------------------------------------------------------------------------------
# Stage 3: did it say the answer correctly? (round trip: TTS audio -> reference ASR -> text)
# ---------------------------------------------------------------------------------------------


def _round_trip(trace):
    tts, back = _span(trace, "tts"), _span(trace, "tts_reasr")
    if not tts or not back:
        return None
    said, out, heard_back = _dict(tts.inputs), _dict(tts.outputs), _dict(back.outputs)
    if "text" not in said or "text" not in heard_back:
        return None
    return said["text"], heard_back["text"], out.get("duration_s") or 0.0


@scorer(aggregations=["mean", "median", "p90"])
def tts_roundtrip_wer(trace):
    rt = _round_trip(trace)
    if rt is None:
        return None
    intended, back, _ = rt
    return Feedback(
        value=round(wer(intended, back), 4),
        rationale=f"intended: {normalize(intended)}\nheard back: {normalize(back)}",
    )


@scorer
def tts_term_accuracy(trace):
    rt = _round_trip(trace)
    if rt is None:
        return None
    intended, back, _ = rt
    terms = terms_in(intended)
    if not terms:
        return None
    came_back = set(terms_in(back))
    missing = [t for t in terms if t not in came_back]
    return Feedback(
        value=round((len(terms) - len(missing)) / len(terms), 4),
        rationale=f"lost: {', '.join(missing)}" if missing else f"all came back: {', '.join(terms)}",
    )


@scorer
def tts_number_accuracy(trace):
    rt = _round_trip(trace)
    if rt is None:
        return None
    intended, back, _ = rt
    expected = numbers(intended)
    if not expected:
        return None
    value, missing = _number_match(expected, numbers(back))
    return Feedback(
        value=round(value, 4),
        rationale=f"lost or changed: {', '.join(missing)}" if missing else f"all came back: {', '.join(expected)}",
    )


def _number_words(n: int) -> int:
    """Words in an English reading of n: 1840 -> "one thousand eight hundred forty" (5)."""
    if n < 20:
        return 1
    if n < 100:
        return 1 if n % 10 == 0 else 2
    if n < 1000:
        return 2 + (_number_words(n % 100) if n % 100 else 0)
    if n < 1_000_000:
        return _number_words(n // 1000) + 1 + (_number_words(n % 1000) if n % 1000 else 0)
    return len(str(n))


def spoken_word_count(text: str) -> int:
    """About how many words a TTS voice says for this text. Counting written words undercounts
    numbers and codes: "$1,840" is spoken as six words and "TLF" as three letters."""
    count = text.count("$")
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?|[A-Za-z]+(?:'[A-Za-z]+)?", text):
        if tok[0].isdigit():
            whole, _, frac = tok.replace(",", "").partition(".")
            count += _number_words(int(whole)) + (1 + len(frac) if frac else 0)
        elif tok.isupper() and 2 <= len(tok) <= 4:
            count += len(tok)
        else:
            count += 1
    return count


@scorer
def tts_speaking_rate(trace):
    rt = _round_trip(trace)
    if rt is None or not rt[2]:
        return None
    intended, _, duration = rt
    words = spoken_word_count(intended)
    wpm = words / duration * 60
    low, high = SPEAKING_RATE_WPM
    flag = "within" if low <= wpm <= high else "outside"
    return Feedback(
        value=round(wpm, 1),
        rationale=f"{wpm:.0f} words per minute ({words} spoken words in {duration:.1f} s), {flag} {low} to {high}",
    )


@scorer
def tts_complete(trace):
    """The brief's truncation check, phrased so True is good: at least 80% of the words came back."""
    rt = _round_trip(trace)
    if rt is None:
        return None
    intended, back, _ = rt
    want, got = len(normalize_tokens(intended)), len(normalize_tokens(back))
    return Feedback(
        value=got >= 0.8 * want,
        rationale=f"{got} of {want} intended words came back",
    )


STAGE1 = [asr_wer, asr_term_error_rate, asr_number_accuracy, asr_exact_match]
STAGE2_CODE = [tool_name_match, tool_args_match, spoken_suitability, answer_exact_numeric]
STAGE3 = [tts_roundtrip_wer, tts_term_accuracy, tts_number_accuracy, tts_speaking_rate, tts_complete]

METRIC_STAGE = {
    **{s.name: "1_asr" for s in STAGE1},
    **{s.name: "2_reasoning" for s in STAGE2_CODE},
    "correctness": "2_reasoning",
    "voice_guidelines": "2_reasoning",
    **{s.name: "3_tts" for s in STAGE3},
}
# For these metrics lower is better; everything else is higher-is-better.
LOWER_IS_BETTER = {"asr_wer", "asr_term_error_rate", "tts_roundtrip_wer"}


def scorers_for(condition: str, judge_model: str, include_judges: bool = True) -> list:
    """Oracle traces have no ASR span (they start from the true transcript); their speaking stage
    reads the reference answer. As-heard traces carry all three stages."""
    judges = reasoning_judges(judge_model) if include_judges else []
    if condition == "oracle":
        return STAGE2_CODE + judges + STAGE3
    return STAGE1 + STAGE2_CODE + judges + STAGE3
