"""Demo configuration: every name in one place. Adding a candidate means one entry in CANDIDATES
(plus an adapter class if it isn't a new combination of existing parts)."""

from __future__ import annotations

CATALOG = "pavan_naidu_catalog"
SCHEMA = "mlflow"
VOLUME = "voice_audio"
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# The schema is shared with other demos, so every table here starts with "voice_".
TABLES = {
    name: f"{CATALOG}.{SCHEMA}.voice_{name}"
    for name in ["golden_utterances", "domain_lexicon", "pipeline_outputs", "eval_results", "eval_scorecard", "attribution"]
}

WAREHOUSE_ID = "fb3b5c1dee0b9d55"
EXPERIMENT_NAME = "/Users/pavan.naidu@databricks.com/Voice Agent Evals"
CONDITIONS = ["oracle", "as_heard"]

# Same questions and expectations in every version; the audio differs, and v4 rewords some questions.
# Clips are never overwritten, because traces reference them by path and sha256.
#   v1: macOS `say` voices (scripts/make_input_audio.py)
#   v2: Kokoro-82M (notebooks/00_input_audio), a different TTS model from every candidate's
#   v3: the v2 clips with background noise at 5 dB SNR (notebooks/00_noisy_audio). On clean audio
#       every ASR heard nearly every product name; at 5 dB they miss 3 to 5 of 10.
#   v4: Kokoro at 20 dB, with 5 of the 12 questions spoken the way callers talk: fillers, a restart,
#       a corrected order number (data/wording_v4.jsonl; notebooks/00_input_audio)
DATASET_VERSION = "v4"
INPUT_DIRS = {
    "v1": f"{VOLUME_ROOT}/input",
    "v2": f"{VOLUME_ROOT}/input/v2",
    "v3": f"{VOLUME_ROOT}/input/v3",
    "v4": f"{VOLUME_ROOT}/input/v4",
}
INPUT_DIR = INPUT_DIRS[DATASET_VERSION]
WORDING = {"v4": "wording_v4.jsonl"}  # utterance_id -> spoken_text and gt_transcript for that version

# Kokoro-82M (Apache-2.0), ONNX build. Each golden-set speaker maps to a distinct voice.
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin",
}
KOKORO_VOICES = {
    "Samantha": ("af_heart", "en-us"),
    "Daniel": ("bm_george", "en-gb"),
    "Karen": ("bf_emma", "en-gb"),
    "Moira": ("bf_isabella", "en-gb"),
    "Aman": ("am_michael", "en-us"),
}
INPUT_SNR_DB = 20.0  # light background noise, as in v1
NOISY_SNR_DB = 5.0  # v3

# The judge must be at least as strong as the strongest candidate LLM.
JUDGE_MODEL = "databricks:/databricks-claude-opus-5-5"
# Fixed for every candidate's TTS round trip within a run, and tagged on every trace as
# reference_asr. Gemini 4 Carbon (a Unity Gateway model service) with a verbatim prompt: on the same
# 48 clips it got product names and codes right where ai_transcribe wrote "trilog" or "arcas nitray",
# so the round trip measures the voice rather than the transcriber. "ai_transcribe" and
# "gemini:<serving endpoint>" also work.
REFERENCE_ASR = "gateway:system.ai.gemini-4-carbon"

PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_VOICES = {
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_GB-alan-medium": "en/en_GB/alan/medium",
}

CANDIDATES = [
    {
        "name": "baseline_cascade",
        "type": "cascade",
        "asr": "ai_transcribe",
        # A small open model on purpose, so this candidate's weak stage is reasoning: on the v4 true
        # transcripts it picked the right tool and key argument 15 times in 24 (gpt-oss-20b: 23,
        # Qwen3-Next 80B: 21, GLM-5.3 Flash and Gemini 3.1 Flash Lite: 24).
        "llm": "databricks-gemma-3-12b",
        "tts_voice": "en_US-lessac-medium",
        "enabled": True,
    },
    {
        "name": "mock_noisy",
        "type": "mock",
        "base": "baseline_cascade",  # same models, plus the planted errors in MOCK_INJECTIONS
        "enabled": False,  # a check on the scorers, not a candidate: run it by name to validate them
    },
    {
        "name": "gemini_lite_cascade",
        "type": "cascade",
        "asr": "gemini:databricks-gemini-3-1-flash-lite",
        "llm": "databricks-gemini-3-1-flash-lite",
        "tts_voice": "en_GB-alan-medium",
        "enabled": True,
    },
    {
        "name": "gemini_cascade",
        "type": "cascade",
        "asr": "gemini:databricks-gemini-3-8-flash",
        "llm": "databricks-gemini-3-8-flash",
        "tts_voice": "en_GB-alan-medium",
        "enabled": False,  # the second demo run: pass "baseline_cascade,gemini_cascade" to 02
    },
]


def load_golden(version: str = DATASET_VERSION) -> list[dict]:
    """The golden set for a dataset version: golden_v1.jsonl, with that version's rewording applied."""
    import json
    from pathlib import Path

    data = Path(__file__).resolve().parent.parent / "data"
    read = lambda name: [json.loads(line) for line in (data / name).read_text().splitlines() if line.strip()]  # noqa: E731
    rows = read("golden_v1.jsonl")
    if version in WORDING:
        changes = {r["utterance_id"]: r for r in read(WORDING[version])}
        rows = [{**r, **changes.get(r["utterance_id"], {})} for r in rows]
    return rows


def piper_model_path(voice: str) -> str:
    return f"{VOLUME_ROOT}/models/piper/{voice}.onnx"


def piper_model_url(voice: str) -> str:
    return f"{PIPER_BASE_URL}/{PIPER_VOICES[voice]}/{voice}.onnx"


def kokoro_path(name: str) -> str:
    return f"{VOLUME_ROOT}/models/kokoro/{name}"


def experiment_kwargs() -> dict:
    """Arguments for mlflow.set_experiment that bind traces to UC, as in ../call-transcript-evals."""
    from mlflow.entities.trace_location import UnityCatalog

    return {"experiment_name": EXPERIMENT_NAME,
            "trace_location": UnityCatalog(catalog_name=CATALOG, schema_name=SCHEMA)}


def set_experiment():
    """Create or reuse the GenAI experiment. The UC trace location is immutable once set."""
    import os

    import mlflow

    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WAREHOUSE_ID
    try:
        return mlflow.set_experiment(**experiment_kwargs())
    except mlflow.MlflowException:
        return mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)


def gateway_client():
    """OpenAI-compatible client for Unity Gateway model services such as system.ai.gemini-4-carbon."""
    from databricks.sdk import WorkspaceClient
    from openai import OpenAI

    w = WorkspaceClient()
    token = w.config.authenticate()["Authorization"].split(" ", 1)[1]
    return OpenAI(base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1", api_key=token)


def build_asr(spec: str, spark, client, verbatim: bool = False):
    """An ASR part from its config string: "ai_transcribe", "gemini:<serving endpoint>" or
    "gateway:<model service>". `verbatim` is for the round-trip reference."""
    from .candidates import AiTranscribe, GeminiTranscriber

    prompt = GeminiTranscriber.VERBATIM_PROMPT if verbatim else None
    if spec == "ai_transcribe":
        return AiTranscribe(spark)
    if spec.startswith("gemini:"):
        return GeminiTranscriber(client, spec.split(":", 1)[1], prompt)
    if spec.startswith("gateway:"):
        return GeminiTranscriber(gateway_client(), spec.split(":", 1)[1], prompt)
    raise ValueError(f"Unknown ASR {spec!r}")


def build_reference_asr(spec: str, spark, client):
    """The fixed transcriber for every candidate's TTS round trip."""
    return build_asr(spec, spark, client, verbatim=True)


def build_candidates(spark, client, names: list[str] | None = None) -> list:
    """Build the enabled candidates (or the named ones). `client` is an OpenAI-compatible client."""
    from .candidates import CascadeCandidate, MockNoisyCandidate, PiperTTS, ToolCallingAgent

    by_name = {c["name"]: c for c in CANDIDATES}

    def cascade(cfg: dict):
        return CascadeCandidate(
            cfg["name"], build_asr(cfg["asr"], spark, client), ToolCallingAgent(client, cfg["llm"]),
            PiperTTS(piper_model_path(cfg["tts_voice"])),
        )

    wanted = names or [c["name"] for c in CANDIDATES if c.get("enabled")]
    built = []
    for name in wanted:
        cfg = by_name[name]
        if cfg["type"] == "cascade":
            built.append(cascade(cfg))
        elif cfg["type"] == "mock":
            built.append(MockNoisyCandidate(name, cascade(by_name[cfg["base"]])))
        else:
            raise ValueError(f"Unknown candidate type {cfg['type']!r}")
    return built
