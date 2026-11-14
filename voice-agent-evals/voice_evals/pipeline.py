"""Run every candidate x utterance x condition and log one text-only MLflow trace per turn.

Conditions (each stage gets perfect input in "oracle", the previous stage's real output in "as_heard"):
  oracle    the agent answers the human-verified transcript (Stage 2A), and the candidate's TTS
            speaks the human-written reference answer (Stage 3A: speech on identical text)
  as_heard  the full pipeline: ASR -> agent -> TTS of its own answer -> reference ASR
            (Stages 1, 2B and 3B)
"""

from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor

import mlflow
from mlflow.entities import SpanType

from .tracing import set_inputs, set_outputs, trace_tags

EXPECTATION_KEYS = [
    "gt_transcript",
    "domain_terms",
    "expected_tool_calls",
    "expected_facts",
    "expected_numbers",
    "reference_answer",  # not "expected_response": Correctness rejects it alongside expected_facts
]
AUDIO_REF_KEYS = ["audio_uri", "audio_sha256", "duration_s", "sample_rate_hz"]


def new_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("r%Y%m%d_%H%M%S")


def tts_path(volume_root: str, run_id: str, candidate: str, utterance_id: str, suffix: str = "") -> str:
    return f"{volume_root}/tts/{run_id}/{candidate}/{utterance_id}{suffix}.wav"


def run_turn(candidate, utt: dict, condition: str, *, run_id: str, tools, reference_asr,
             volume_root: str, dataset_version: str) -> dict:
    row = {
        "run_id": run_id, "candidate": candidate.name, "condition": condition,
        "utterance_id": utt["utterance_id"], "category": utt["category"], "trace_id": None,
        "asr_text": None, "agent_answer": None, "tool_calls": None, "tts_uri": None,
        "tts_duration_s": None, "reasr_text": None, "latency_ms_asr": None,
        "latency_ms_agent": None, "latency_ms_tts": None, "error": None,
        "dataset_version": dataset_version, "models": json.dumps(candidate.models),
    }
    audio = {k: utt[k] for k in AUDIO_REF_KEYS}
    with mlflow.start_span(name="voice_turn", span_type=SpanType.CHAIN) as root:
        row["trace_id"] = root.trace_id
        tags = trace_tags(
            run_id=run_id, candidate=candidate.name, utterance=utt, condition=condition,
            asr_source="none" if condition == "oracle" else "native",
            dataset_version=dataset_version,
        )
        tags.update({f"{k}_model": v for k, v in candidate.models.items()})
        tags["reference_asr"] = reference_asr.label
        mlflow.update_current_trace(tags=tags)
        # The request is what the caller asked, identical in both conditions: MLflow's run comparison
        # pairs a question's two traces by it, and the judges grade against the real question. What
        # the agent heard is the asr span's output and the agent span's input.
        set_inputs(root, {"question": utt["gt_transcript"], "audio_uri": audio["audio_uri"]})
        candidate.begin_turn(utt)

        if condition == "oracle":
            transcript = utt["gt_transcript"]
        else:
            with mlflow.start_span(name="asr", span_type="ASR") as span:
                set_inputs(span, audio)
                asr = candidate.transcribe(audio["audio_uri"])
                set_outputs(span, {"text": asr.text, "latency_ms": asr.latency_ms, "model": candidate.models["asr"]})
            mlflow.update_current_trace(tags={"asr_source": asr.source})
            transcript = asr.text
            row.update(asr_text=asr.text, latency_ms_asr=asr.latency_ms)

        agent = candidate.respond(transcript, tools)
        row.update(agent_answer=agent.answer, tool_calls=json.dumps(agent.tool_calls),
                   latency_ms_agent=agent.latency_ms)
        outputs = {"answer": agent.answer}

        # As heard, the TTS speaks the agent's own answer; in the oracle run it speaks the
        # reference answer, so every candidate's voice is tested on the same text.
        spoken = agent.answer if condition == "as_heard" else utt.get("reference_answer")
        if spoken:
            suffix = "" if condition == "as_heard" else "_reference"
            out_uri = tts_path(volume_root, run_id, candidate.name, utt["utterance_id"], suffix)
            with mlflow.start_span(name="tts", span_type="TTS") as span:
                set_inputs(span, {"text": spoken, "model": candidate.models["tts"],
                                  "source": "own answer" if condition == "as_heard" else "reference answer"})
                tts = candidate.synthesize(spoken, out_uri)
                set_outputs(span, {"tts_uri": tts.uri, "duration_s": tts.duration_s,
                                   "sample_rate_hz": tts.sample_rate_hz, "latency_ms": tts.latency_ms})
            with mlflow.start_span(name="tts_reasr", span_type="ASR") as span:
                set_inputs(span, {"tts_uri": tts.uri, "model": reference_asr.label})
                start = time.perf_counter()
                back = reference_asr(tts.uri)
                set_outputs(span, {"text": back, "latency_ms": round((time.perf_counter() - start) * 1000, 1)})
            outputs["tts_uri"] = tts.uri
            row.update(tts_uri=tts.uri, tts_duration_s=tts.duration_s, reasr_text=back,
                       latency_ms_tts=tts.latency_ms)

        set_outputs(root, outputs)

    # Ground truth rides on the trace, so the trace view shows heard vs. true text side by side.
    for key in EXPECTATION_KEYS:
        if utt.get(key) is not None:
            mlflow.log_expectation(trace_id=row["trace_id"], name=key, value=utt[key])
    return row


def run_pipeline(candidates: list, golden: list[dict], conditions: list[str], *, run_id: str,
                 tools, reference_asr, volume_root: str, dataset_version: str,
                 max_workers: int = 4, retries: int = 1, log=print) -> list[dict]:
    """Run all turns. A failed turn is retried, then recorded with its error; the run continues."""

    def one(args):
        candidate, utt, condition = args
        for attempt in range(retries + 1):
            try:
                return run_turn(candidate, utt, condition, run_id=run_id, tools=tools,
                                reference_asr=reference_asr, volume_root=volume_root,
                                dataset_version=dataset_version)
            except Exception as e:  # noqa: BLE001 - record and keep going
                error = f"{type(e).__name__}: {e}"[:1000]
                log(f"  {candidate.name} {utt['utterance_id']} {condition} attempt {attempt + 1}: {error}")
        return {"run_id": run_id, "candidate": candidate.name, "condition": condition,
                "utterance_id": utt["utterance_id"], "category": utt["category"], "error": error,
                "dataset_version": dataset_version, "models": json.dumps(candidate.models)}

    rows: list[dict] = []
    for candidate in candidates:
        jobs = [(candidate, utt, cond) for utt in golden for cond in conditions]
        start = time.perf_counter()
        # The mock keeps per-turn state, so it runs one turn at a time.
        workers = 1 if getattr(candidate, "sequential", False) else max_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(one, jobs))
        failed = sum(1 for r in done if r.get("error"))
        log(f"{candidate.name}: {len(done)} turns in {time.perf_counter() - start:.0f}s, {failed} failed")
        rows.extend(done)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    for r in rows:
        r.setdefault("created_at", created)
    return rows
