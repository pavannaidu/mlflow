# Voice Agent Evals (MVP)

A text-first, three-stage evaluation of a voice agent with MLflow on Databricks:

1. **ASR**: did the agent hear the question correctly?
2. **Reasoning**: given what it heard, did it answer and use tools correctly?
3. **TTS**: did it say the answer correctly?

MLflow stores the text of every turn. The audio stays in a UC Volume, and each trace carries only
the Volume path of each clip:

```
voice_turn   tags: voice_run_id, candidate, condition (oracle | as_heard), utterance_id, category, ...
  in : {"question": "<what the caller asked>", "audio_uri": "/Volumes/.../input/v4/u_006.wav"}
  out: {"answer": "<what it meant to say>", "tts_uri": "/Volumes/.../tts/<run>/<candidate>/u_006.wav"}
  ├─ asr          audio_uri, sha256, duration     → text     (as_heard only)
  ├─ agent        transcript                      → answer   (llm and tool spans inside)
  ├─ tts          text to speak                   → tts_uri, duration
  └─ tts_reasr    tts_uri                         → text (reference ASR, for the round trip)
expectations on the trace: gt_transcript, domain_terms, expected_tool_calls, expected_facts,
                           expected_numbers, reference_answer
feedback after evaluation: one assessment per scorer, with its rationale
```

Every question runs twice per candidate:

| Condition | Agent input | TTS speaks | Measures |
| --- | --- | --- | --- |
| `oracle` | the human-checked transcript | the human-written reference answer | 2A reasoning on its own; 3A the voice on the same text for every candidate |
| `as_heard` | its own ASR transcript | its own answer | 1 hearing; 2B reasoning as deployed; 3B what a caller actually hears |

Tool calls and correctness are scored in both conditions against the golden expectations, so
2A minus 2B is what mishearing cost (the ASR impact). The trace's request is the caller's question
in both conditions; what the agent heard is the `asr` span's output and the `agent` span's input.
So the judges grade against the real question, and MLflow's run comparison pairs each question's
two traces.

A payload guard (`voice_evals/tracing.py`) fails any turn whose span payload is over 64 KB or
contains bytes, an audio data URI or base64. To hear a clip, use the inline players in
`03_evaluate` or the **Voice Eval Inspector** app.

**Illustrative data.** Meridian Ortho and every product, SKU, price and order are made up.
See [PLAN.md](PLAN.md) for the design and how it maps to the build brief.

## What it uses in MLflow

| MLflow feature | How the demo uses it |
| --- | --- |
| Experiment with traces in Unity Catalog | `mlflow.set_experiment(..., trace_location=UnityCatalog(...))` in `voice_evals/config.py`. Traces land in the `<experiment_id>_otel_*` tables in `pavan_naidu_catalog.mlflow`, where SQL can read them. |
| Manual tracing | One trace per turn, written with `mlflow.start_span`: `voice_turn`, `asr`, `agent`, `llm`, one `TOOL` span per tool call, `tts` and `tts_reasr`. Tags record the run, candidate, condition, question, models, and dataset and normalizer versions. No autologging, so audio never enters a span; a payload guard enforces that. |
| Expectations on traces | `mlflow.log_expectation` for the true transcript, domain terms, expected tool calls, facts and numbers, and the reference answer. |
| Offline evaluation over traces | `mlflow.genai.evaluate(data=<list of traces>, scorers=...)` with no `predict_fn`: the logged traces are scored as they are, and the feedback lands on the same trace IDs. One evaluation run per candidate and condition (`eval/<candidate>/<condition>`). |
| Code scorers | `@scorer` functions for all three stages: WER, term and number accuracy, tool name and argument match, numbers in the answer, speakability, round-trip WER, speaking rate and completeness. Each returns `Feedback(value, rationale)`, or `None` when it doesn't apply to a turn. |
| Built-in judges | `Correctness` (against the expected facts) and `Guidelines` (`voice_guidelines`: declines medical advice and surgeon recommendations, states units), with `databricks:/databricks-claude-opus-5-5` as the judge model. |
| Human feedback | The app saves listening ratings with `mlflow.log_feedback` (source `HUMAN`, the reviewer's email) on the turn's trace; a changed rating updates the reviewer's label with `mlflow.update_assessment`. |
| Run comparison | Each candidate's true-transcript and as-heard evaluation runs open side by side in MLflow's compare view. The trace request is the same in both conditions, so the rows pair up by question. |
| Reading results back | `mlflow.search_traces` by tag, and the assessments on each trace, fill the scorecard tables (`voice_eval_results`, `voice_eval_scorecard`, `voice_attribution`). The app links each turn to its trace inside its evaluation run. |
| Models | Agent models through Foundation Model APIs; the round-trip transcriber (Gemini 4 Carbon) through a Unity Gateway model service. |

## Layout

| Path | What it is |
| --- | --- |
| `data/` | 12-question golden set, domain lexicon, mock tool fixtures |
| `voice_evals/normalize.py` | Versioned text normalizer shared by the ASR and TTS scorers |
| `voice_evals/candidates.py` | Candidate interface; `ai_transcribe`, Gemini ASR, tool-calling agent, Piper TTS; `mock_noisy` with planted errors |
| `voice_evals/tools.py` | Four deterministic mock tools, recorded as TOOL spans |
| `voice_evals/pipeline.py` | Runs candidates x utterances x conditions, one trace per turn |
| `voice_evals/scorers.py` | Stage 1, 2 and 3 scorers (code scorers plus two LLM judges) |
| `voice_evals/evaluate.py` | Offline `mlflow.genai.evaluate` over traces, scorecard, ASR impact, attribution |
| `voice_evals/config.py` | All names, models and candidates in one place |
| `notebooks/` | `00_smoke_test`, `01_setup`, `02_run_pipeline`, `03_evaluate` |
| `notebooks/00_input_audio` | Generates the input clips with Kokoro-82M: dataset v4 (the default; 5 of the 12 questions reworded with fillers and a correction, `data/wording_v4.jsonl`) or v2 |
| `notebooks/00_noisy_audio` | Derives dataset v3 from v2: the same clips with background noise at 5 dB SNR |
| `scripts/make_input_audio.py` | Generates the v1 input clips with macOS voices and uploads them |
| `app/` | Voice Eval Inspector: FastAPI backend plus React frontend, a Databricks App |
| `tests/` | Normalizer, payload guard, and an end-to-end pipeline and scorer test on a local MLflow store |

## Run it

Workspace: FEVM (`fevm-pavan-naidu`). Everything lives in `pavan_naidu_catalog.mlflow`, prefixed
`voice_`.

```bash
# 1. Copy this folder to the workspace
databricks sync . /Workspace/Users/pavan.naidu@databricks.com/voice-agent-evals --profile FEVM

# 2. Run the notebooks in order on serverless (environment 5):
#    00_input_audio (makes the v4 question clips with Kokoro), 01_setup, 02_run_pipeline, 03_evaluate
#    00_smoke_test is an optional check of the model endpoints
```

`02_run_pipeline` takes three widgets: `run_id` (blank for a new run), `candidates` (blank for
the ones enabled in `config.py`) and `reference_asr`, the transcriber for every TTS round trip
(blank for `gateway:system.ai.gemini-4-carbon`, a Unity Gateway model service prompted to transcribe
verbatim; `ai_transcribe` and `gemini:<serving endpoint>` also work). The reference is tagged on
every trace. `03_evaluate` scores the latest run unless you pass `run_id`.

### The app

`voice-eval-inspector` on FEVM
(https://voice-eval-inspector-7474645380671326.aws.databricksapps.com) has three views, each with
its own link (`#/scorecard`, `#/turns/u_006`, `#/listening`, plus `?run=<run_id>`):

- **Scorecard**: per-stage metrics for every candidate (best value in bold): tools and
  correctness with the true transcript and as heard, and round-trip WER on the reference answer
  and on its own answer. A bar per candidate shows which stage each failed turn came from.
- **Turns**: one question at a time, candidates side by side: the word diff of what it heard,
  its tool calls (and whether the true transcript led to the same calls), its answer, both
  spoken clips (its own answer and the reference answer) with their round-trip diffs and
  listening ratings, and the failing checks with their rationales.
- **Listening**: every answer a caller would hear (24: two candidates' own answers to 12
  questions), grouped by question. Each rating saves to that turn's trace as MLflow human feedback,
  next to the scorer results; changing a rating updates your earlier one. Reference-answer clips
  are scored automatically and play in Turns, but aren't rated.

Trace links open the trace inside its MLflow evaluation run
(`/ml/experiments/<id>/evaluation-runs/<eval run>/details?selectedEvaluationId=<trace>`); the
experiment's trace list doesn't open a selected trace. "Compare in MLflow" on the Scorecard opens a
candidate's as-heard and true-transcript evaluation runs side by side.

It follows the Databricks brand (`.databricks-impeccable.md`): DM Sans and DM Mono self-hosted
under the SIL Open Font License (`app/frontend/public/fonts`), and the official Databricks
symbol from the brand assets.

### Deploy the app

Databricks Apps runs `npm install` and `npm run build` on deploy, then the uvicorn command in
`app/app.yaml`.

```bash
# once: create it with its resources (SQL warehouse CAN_USE, experiment CAN_EDIT, Volume READ_VOLUME)
databricks apps create --json @scripts/app-create.json --profile FEVM

# once: grant its service principal read on the voice_ tables and read/write on the trace tables
#   GRANT USE CATALOG / USE SCHEMA, SELECT ON voice_*, SELECT, MODIFY ON `<experiment_id>_otel_*`,
#   SELECT ON `<experiment_id>_trace_metadata` and `<experiment_id>_trace_unified`

# every change: sync, then deploy
databricks sync . /Workspace/Users/pavan.naidu@databricks.com/voice-agent-evals --profile FEVM
databricks apps deploy voice-eval-inspector \
  --source-code-path /Workspace/Users/pavan.naidu@databricks.com/voice-agent-evals/app --profile FEVM
```

### Tests

```bash
pip install "mlflow>=3.10" numpy pytest
pytest
```

The end-to-end test runs the real pipeline, tracing and `mlflow.genai.evaluate` against a local
SQLite MLflow store, with fake ASR, LLM and TTS. It checks that every planted `mock_noisy` failure
is caught by its own stage, in both conditions.

## Add a candidate

Add one block to `CANDIDATES` in `voice_evals/config.py`. If it combines existing parts (for
example a different chat model or Piper voice), that's all. For a new kind of system, such as a
realtime speech-to-speech API, write one class with `begin_turn`, `transcribe`, `respond` and
`synthesize`. It must still return the intermediate text: its own transcript, or a
re-transcription by the reference ASR with `source="reference"`. Scorers don't change.

Two candidates run by default: `baseline_cascade` (`ai_transcribe`, Gemma 3 12B, Piper lessac),
whose small open agent model is its weak stage, and `gemini_lite_cascade` (Gemini 3.1 Flash Lite
for both hearing and reasoning, Piper alan). `gemini_cascade` is the same with Gemini 3.8 Flash; for
a second demo run, pass `baseline_cascade,gemini_cascade` in the `candidates` widget. On the v4
true transcripts the open models picked the right tool and key argument 15 (Gemma 3 12B), 21
(Qwen3-Next 80B), 23 (gpt-oss-20b) and 24 (GLM-5.3 Flash) times in 24; Gemini 3.1 Flash Lite 24. `mock_noisy` is a check on the scorers rather than a candidate: it plants one failure per
stage. Pass it in the `candidates` widget to confirm each stage still catches its own failure.

## Known limitations

- **No audio judging in MLflow.** Audio stays in UC Volumes, referenced by path; traces don't play audio.
- **Naturalness and prosody** come only from the human listening ratings. The round trip measures intelligibility, not quality.
- **Round-trip reference.** One fixed transcriber re-transcribes every candidate's speech: Gemini 4 Carbon with a verbatim prompt. It is the judge of the speaking stage, so it gets only the audio: never the intended text, the question or the product names (its formatting example is a made-up code). On the same 48 clips it brought round-trip WER from 4 to 7% (with `ai_transcribe`, which wrote "trilog" for TriLock and "arcas nitray" for Arcus knee tray) down to 0 to 1%, and gave identical transcripts on 45 of 48 when run twice. Two cautions: an LLM transcriber tends to reply to speech that sounds like half a support call (asked "Can you repeat the product number?", a plainer prompt made it write "It is AX-1020"; the prompt now says the audio may be a question or a refusal and must not be answered), and a strong model can repair a slightly mispronounced word from context, so a 0% round trip doesn't prove perfect pronunciation. The human listening check covers that.
- **Realtime voice models** would be called externally through an adapter. Databricks doesn't serve realtime speech-to-speech APIs.
- **No TTS model is served on Databricks.** Piper runs on CPU inside the notebook; a hosted TTS would plug in as another adapter.
- **Single runs are noisy.** With `ai_transcribe` as the reference, two runs flipped which voice did better on the reference answers: one misheard product name moves a 12-sentence average by several points. Most of that noise was the transcriber, but Piper's audio also varies from run to run. Compare voices over more questions or repeated runs.
- **Reference answers** (`reference_answer` in the golden set) are hand-written from the fixtures. Stage 3A compares voices on this text, so check it like the transcripts. It's stored as `reference_answer` because MLflow's Correctness judge raises when a trace has both `expected_facts` and `expected_response`; correctness is graded against the facts.
- **LLM judges** are only as trustworthy as the judge model, which must be at least as capable as the candidates (`databricks-claude-opus-5-5` here). Code scorers against ground truth are the main signal.
- **Synthetic input audio** overstates ASR quality: on clean audio every ASR heard nearly every product name. The default dataset (v4) makes 5 of the 12 questions sound like real callers (fillers, a restart, a corrected order number) at light noise; v3 instead adds heavy background noise (5 dB SNR, steady hiss rather than a real room). The normalizer drops "um" and "uh" from both sides, so an ear isn't scored on whether it writes them. Every table, chart and the app are flagged. The
  questions are read by Kokoro-82M, a different TTS model from the candidates' Piper, so the ASR stage
  isn't tested on its own family of voices. Kokoro has US and UK English voices only, so v2 has less
  accent variety than v1 (macOS voices, which included Australian, Irish and Indian English).
- **`ai_transcribe` is Beta**: region-limited, English and Spanish only, and a workspace admin has to turn on the preview.
- **Trace tag `voice_run_id`, not `run_id`**: MLflow's trace filter treats `tags.run_id` specially, so a plain tag of that name never matches.
- **Deleting traces stored in UC**: `MlflowClient().delete_traces` rejects UC trace IDs. Find them in the
  `<experiment_id>_trace_metadata` view (for example `tags['voice_run_id'] = '...'`), then delete their
  rows from `<experiment_id>_otel_spans`, `_otel_logs` (by `trace_id`) and `_otel_annotations` (by
  `target_id`). Delta time travel can restore them.
