"""Candidate voice systems. Every candidate has the same three methods, so swapping one is config.

A cascaded candidate is three parts (ASR, agent LLM, TTS). `mock_noisy` reuses the baseline and
plants known errors, which proves the scorers catch them and makes the demo run without keys.
Adding a candidate means one class here (or a new combination of parts) and one config block.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Protocol

import mlflow
from mlflow.entities import SpanType

from .audio import write_bytes
from .tools import ToolRegistry
from .tracing import set_inputs, set_outputs


@dataclass
class ASRResult:
    text: str
    latency_ms: float
    source: str = "native"  # native | reference | mock


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class TTSResult:
    uri: str
    duration_s: float
    sample_rate_hz: int
    latency_ms: float


class VoiceCandidate(Protocol):
    name: str
    models: dict[str, str]  # {"asr": ..., "llm": ..., "tts": ...}, tagged on every trace

    def begin_turn(self, utterance: dict) -> None: ...
    def transcribe(self, audio_uri: str) -> ASRResult: ...
    def respond(self, transcript: str, tools: ToolRegistry) -> AgentResult: ...
    def synthesize(self, text: str, out_uri: str) -> TTSResult: ...


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _text(content) -> str:
    """Message content as plain text. Gemini endpoints return a list of content parts."""
    if isinstance(content, list):
        parts = []
        for p in content:
            text = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    return (content or "").strip()


# ---------------------------------------------------------------------------------------------
# ASR parts
# ---------------------------------------------------------------------------------------------


class AiTranscribe:
    """`ai_transcribe` SQL AI Function (Beta). It can also be the round-trip reference (`REFERENCE_ASR`)."""

    label = "ai_transcribe"

    def __init__(self, spark):
        self.spark = spark

    def __call__(self, audio_uri: str) -> str:
        if not audio_uri.startswith("/Volumes/") or "'" in audio_uri:
            raise ValueError(f"Expected a Volume path, got {audio_uri!r}")
        row = self.spark.sql(
            "SELECT to_json(ai_transcribe(content)) AS result "
            f"FROM read_files('{audio_uri}', format => 'binaryFile')"
        ).collect()[0]
        result = json.loads(row["result"])
        if result.get("error_message"):
            raise RuntimeError(f"ai_transcribe failed for {audio_uri}: {result['error_message']}")
        segments = (result.get("response") or {}).get("segments") or []
        return " ".join(s["text"].strip() for s in segments if s.get("text")).strip()


class GeminiTranscriber:
    """Gemini audio input through the chat completions API.

    The audio goes to the model as a base64 data URI. That string never reaches a span: this
    call is made inside the `asr` span, whose inputs hold only the Volume path.
    """

    PROMPT = (
        "Transcribe this audio exactly as spoken. Output only the transcript, with no "
        "commentary. Write product codes and numbers the way a customer support agent would."
    )
    # For the TTS round trip, where the transcriber is the judge: it gets only the audio, never the
    # intended text, the question or product names (the example code is deliberately not one of
    # ours). An LLM transcriber tends to reply to speech that sounds like half a support call: given
    # "Can you repeat the product number?" it wrote "It is AX-1020" (3 of 3 tries with a plainer
    # prompt). Saying the audio may be a question or a refusal, and never to reply, fixed 6 of 6.
    VERBATIM_PROMPT = (
        "This recording is one person speaking: an assistant on a support line. It may be a "
        "question, a request or a refusal. Write down exactly the words spoken in it, verbatim, and "
        "nothing else. Do not reply to it, answer it, or write what anyone else might say. Write "
        "numbers as digits and product codes as written, for example AB-1234."
    )

    def __init__(self, client, model: str, prompt: str | None = None):
        self.client, self.model = client, model
        self.prompt = prompt or self.PROMPT
        self.label = model

    def __call__(self, audio_uri: str) -> str:
        with open(audio_uri, "rb") as f:
            data_uri = "data:audio/wav;base64," + base64.b64encode(f.read()).decode()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "audio_url", "audio_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        return _text(resp.choices[0].message.content)


# ---------------------------------------------------------------------------------------------
# Agent part
# ---------------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the voice assistant for Meridian Ortho, an orthopedic implant supplier.
Callers ask about products and orders. Use the tools to look up facts. Never guess prices, stock,
sizes, quantities or dates.

Rules:
- Your reply is spoken aloud. Use one to three short sentences and fewer than 60 words.
  Plain sentences only: no lists, markdown, tables or URLs.
- State units for sizes, quantities and prices.
- If a lookup finds nothing, say you couldn't find it and ask the caller to repeat the product
  or order number.
- Decline medical advice (doses, treatment) and recommendations of doctors or surgeons, and
  suggest the caller ask their care team. Don't call a tool for these.
- If the caller wants to complain or asks for a person, call escalate_to_human, then tell them
  a specialist will call back."""


# Response fields that some endpoints reject when they're sent back in the conversation.
_NOT_RESENT = {"annotations", "refusal", "audio", "function_call"}


def _assistant_turn(msg) -> dict:
    """The assistant message to send back with the tool results.

    Keeps provider extras such as Gemini's thought signatures on tool calls: Gemini rejects a
    follow-up request whose function calls lost them.
    """
    turn = {k: v for k, v in msg.model_dump(exclude_none=True).items() if k not in _NOT_RESENT}
    turn["role"] = "assistant"
    if turn.get("content") in (None, "", []):
        turn.pop("content", None)
    return turn


def _loggable(obj):
    """A copy for the trace with provider signatures masked. They're opaque base64 blobs,
    which the text-only contract keeps out of spans; they're still sent to the model."""
    if isinstance(obj, dict):
        return {k: "[provider signature omitted]" if "signature" in k.lower() or k == "extra_content"
                else _loggable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_loggable(v) for v in obj]
    return obj


class ToolCallingAgent:
    """A chat model with function calling over the mock tools. Writes agent, llm and tool spans."""

    MAX_STEPS = 5
    FALLBACK = "Sorry, I couldn't finish looking that up. Could you ask me again?"

    def __init__(self, client, model: str):
        self.client, self.model = client, model
        self.label = model

    def __call__(self, transcript: str, tools: ToolRegistry) -> AgentResult:
        start = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        calls: list[dict] = []
        answer = ""
        with mlflow.start_span(name="agent", span_type=SpanType.AGENT) as span:
            set_inputs(span, {"transcript": transcript, "model": self.model, "tools": tools.names()})
            for _ in range(self.MAX_STEPS):
                with mlflow.start_span(name="llm", span_type=SpanType.CHAT_MODEL) as llm_span:
                    set_inputs(llm_span, {"model": self.model, "messages": _loggable(messages)})
                    resp = self.client.chat.completions.create(
                        model=self.model, messages=messages, tools=tools.schemas() or None
                    )
                    msg = resp.choices[0].message
                    set_outputs(llm_span, _loggable(msg.model_dump(exclude_none=True)))
                if not msg.tool_calls:
                    answer = _text(msg.content)
                    break
                messages.append(_assistant_turn(msg))
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = tools.call(tc.function.name, args)
                    calls.append({"name": tc.function.name, "args": args})
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                    )
            exhausted = not answer
            answer = answer or self.FALLBACK
            latency = _ms(start)
            set_outputs(span, {"answer": answer, "tool_calls": calls, "latency_ms": latency,
                               "exhausted_steps": exhausted})
        return AgentResult(answer=answer, tool_calls=calls, latency_ms=latency)


# ---------------------------------------------------------------------------------------------
# TTS part
# ---------------------------------------------------------------------------------------------


class PiperTTS:
    """Piper, an open-source neural TTS that runs on CPU. Voice files are staged in the Volume."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.label = "piper:" + re.sub(r"\.onnx$", "", model_path.rsplit("/", 1)[-1])
        self._voice = None

    def _load(self):
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def __call__(self, text: str, out_uri: str) -> tuple[float, int]:
        voice = self._load()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            if hasattr(voice, "synthesize_wav"):  # piper-tts >= 1.3
                voice.synthesize_wav(text, wav)
            else:
                voice.synthesize(text, wav)
        data = buf.getvalue()
        write_bytes(out_uri, data)
        with wave.open(io.BytesIO(data)) as w:
            return round(w.getnframes() / w.getframerate(), 3), w.getframerate()


# ---------------------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------------------


class CascadeCandidate:
    """ASR -> agent LLM -> TTS, each a separate model."""

    def __init__(self, name: str, asr, agent: ToolCallingAgent, tts: PiperTTS):
        self.name, self._asr, self._agent, self._tts = name, asr, agent, tts
        self.models = {"asr": asr.label, "llm": agent.label, "tts": tts.label}

    def begin_turn(self, utterance: dict) -> None:
        pass

    def transcribe(self, audio_uri: str) -> ASRResult:
        start = time.perf_counter()
        text = self._asr(audio_uri)
        return ASRResult(text=text, latency_ms=_ms(start), source="native")

    def respond(self, transcript: str, tools: ToolRegistry) -> AgentResult:
        return self._agent(transcript, tools)

    def synthesize(self, text: str, out_uri: str) -> TTSResult:
        start = time.perf_counter()
        duration, rate = self._tts(text, out_uri)
        return TTSResult(uri=out_uri, duration_s=duration, sample_rate_hz=rate, latency_ms=_ms(start))


# Fixed per utterance (not random) so every run plants the same failures, one per stage.
MOCK_INJECTIONS = {
    "u_004": {"asr_replace": ("Veltra", "Vectra")},     # Stage 1: domain term misheard
    "u_006": {"asr_replace": ("TLF-2210", "TLF-2201")},  # Stage 1, then a failed lookup in 2B
    "u_008": {"asr_replace": ("14", "40")},             # Stage 1 number error, then 2B
    "u_009": {"tools_only": ["get_product"]},           # Stage 2 reasoning: skips the order lookup
    "u_005": {"tts_keep_fraction": 0.6},                # Stage 3: speech cut off
}


class MockNoisyCandidate:
    """The baseline with planted errors. Its "ASR" is the ground truth with one edit."""

    sequential = True  # begin_turn() keeps per-turn state, so turns run one at a time

    def __init__(self, name: str, base: CascadeCandidate, injections: dict | None = None):
        self.name, self._base = name, base
        self._injections = MOCK_INJECTIONS if injections is None else injections
        self.models = {"asr": "mock(ground truth + planted errors)", **{k: v for k, v in base.models.items() if k != "asr"}}
        self._utt: dict | None = None

    def _rule(self) -> dict:
        return self._injections.get((self._utt or {}).get("utterance_id"), {})

    def begin_turn(self, utterance: dict) -> None:
        self._utt = utterance

    def transcribe(self, audio_uri: str) -> ASRResult:
        text = self._utt["gt_transcript"]
        if "asr_replace" in self._rule():
            old, new = self._rule()["asr_replace"]
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
        return ASRResult(text=text, latency_ms=0.0, source="mock")

    def respond(self, transcript: str, tools: ToolRegistry) -> AgentResult:
        only = self._rule().get("tools_only")
        return self._base.respond(transcript, tools.subset(only) if only else tools)

    def synthesize(self, text: str, out_uri: str) -> TTSResult:
        keep = self._rule().get("tts_keep_fraction")
        if keep:
            words = text.split()
            text = " ".join(words[: max(1, int(len(words) * keep))])
        return self._base.synthesize(text, out_uri)
