# Voice Agent Evals: MVP demo plan

Status: building. Decisions: FEVM workspace, schema `pavan_naidu_catalog.mlflow`, Piper TTS,
input audio from Kokoro-82M (dataset v2; v1 used macOS voices), React + FastAPI app (no Streamlit).
Source: *Voice Agent Evals: MVP Demo Build Brief* (PDF).

**Build status (2026-11-12).** Built, deployed and tested on FEVM. 38 local tests pass. Two demo
runs are kept, both on dataset v4 (Kokoro at 20 dB, 5 of the 12 questions spoken with fillers, a
restart and a corrected order number) with Gemini 4 Carbon as the round-trip judge (audio only,
prompted never to reply to the speech):
`r20261112_050735` (Baseline vs Gemini Lite) and `r20261112_051530` (Baseline vs Gemini 3.8 Flash).
Baseline (`ai_transcribe`, Gemma 3 12B, Piper lessac) answers 67 to 75% of questions right even
from the true transcript; 5 of its 12 turns fail in reasoning. Gemini 3.1 Flash Lite passes all 12.
Gemini 3.8 Flash mishears 22% of product names, which broke a tool argument in 2 turns, but its
answers stay right. Speaking is clean for every voice (0 to 0.3% round-trip WER). Earlier runs were
deleted at the user's request. Every trace link was checked against MLflow.

Fixes found while testing: span payloads of UC-stored traces come back as JSON strings; Gemini
tool calls need their thought signatures sent back (and they're masked in the trace); Gemini
message content can be a list of parts; three expected facts were over-specific; the units
guideline wrongly failed implant sizes.

This keeps the brief's design (text-only traces, three stages scored separately, offline
evaluation) and cuts whatever a first demo can do without.

## 1. The one idea

MLflow stores the text of every voice turn. The audio stays in a UC Volume. Each trace records
what the agent heard, what it decided, what it said, and the Volume path of each audio clip.
Scorers read only text. To hear a clip, play its path in the notebook (or in the app from step 6).
A trace never holds audio bytes.

## 2. Scope: brief vs MVP

| Brief | MVP | Why |
| --- | --- | --- |
| 40 utterances | 12 (3 per category) | Enough to show every failure type |
| Human recordings | Synthetic voices plus light noise, flagged `synthetic_input=true`: v1 macOS voices, v2 Kokoro-82M (different from the candidates' Piper) | No recording session. Real recordings would be a new dataset version; nothing else changes |
| 3 candidates incl. an external realtime API | Two: `baseline_cascade` and `gemini_cascade` (Gemini 3.8 Flash). `mock_noisy` (planted failures) stays as a check on the scorers, run by name | No external keys. The second candidate is the "one adapter" moment |
| Configurable TTS endpoint | Piper (open source, CPU) running in the notebook, two voices | Databricks has no TTS model endpoint |
| Asset Bundle and Lakeflow Job | 3 notebooks run by hand on serverless | Bundle later |
| AI/BI dashboard | Scorecard in a notebook plus MLflow's run comparison view | Dashboard later |
| Labeling session for the listening sample | Listening tab in a React + FastAPI Databricks App; labels saved as MLflow feedback on the trace | The review app can't play Volume audio |
| Unit tests | Normalizer, code scorers, payload guard (pytest, local) | Cheap, and they keep the numbers honest |

Same non-goals as the brief: no realtime or streaming, no audio-quality judge, no production
monitoring, no combined voice score.

## 3. Where things live (FEVM workspace)

| Thing | Value |
| --- | --- |
| Schema | `pavan_naidu_catalog.mlflow` (shared with the Call Transcript demo, so every object here is prefixed `voice_`) |
| Volume | `/Volumes/pavan_naidu_catalog/mlflow/voice_audio/` with `input/`, `tts/{run_id}/{candidate}/`, `models/piper/` |
| Tables | `voice_golden_utterances`, `voice_domain_lexicon`, `voice_pipeline_outputs`, `voice_eval_scorecard` (columns as in the brief, section 4.1) |
| Experiment | `/Users/pavan.naidu@databricks.com/Voice Agent Evals`, GenAI type, traces stored in UC at `pavan_naidu_catalog.mlflow` (same `set_experiment(trace_location=UnityCatalog(...))` pattern as `../call-transcript-evals/Call Transcript Evaluation.ipynb`; trace tables are prefixed with the experiment ID, so they don't collide) |
| App | `voice-eval-inspector` Databricks App on FEVM (React + FastAPI) |
| SQL warehouse | Serverless Starter Warehouse `fb3b5c1dee0b9d55` (`MLFLOW_TRACING_SQL_WAREHOUSE_ID`) |
| Compute | Serverless notebooks, environment version 3 or later (needed by `ai_transcribe`) |
| Models | ASR: `ai_transcribe` (Beta). Agent LLMs: `databricks-gpt-5-4-mini`, `databricks-gemini-3-8-flash`. Judge: `databricks-claude-opus-5-5`, stronger than every candidate |

## 4. Golden set (illustrative data)

Fictional supplier **Meridian Ortho**. Made-up names and SKUs, so domain-term and number
errors show up.

**Product fixtures** (returned by the mock tools)

| SKU | Product | Facts |
| --- | --- | --- |
| TLF-2210 | TriLock Femoral Stem | Titanium; sizes 9 to 18; $1,840 list; size 14 in stock |
| ARC-3305 | Arcus Knee Tray | Cemented and cementless versions; $2,150 list |
| VCS-0450 | Veltra Cortical Screw 4.5 mm | 12 per box; lengths 20 to 60 mm |
| KBC-0040 | Kestrel Bone Cement 40 g | Back-ordered until December 2 |
| ORB-5212 | Orbis Acetabular Cup | Diameters 46 to 64 mm in 2 mm steps |

**Order fixtures**: `ORD-58213` shipped Nov 3, arriving Nov 6 (TLF-2210 x2, VCS-0450 x3).
`ORD-60744` on hold, waiting on KBC-0040 (ARC-3305 x1, KBC-0040 x4).

**Utterances.** Each row stores both the spoken script and the written ground truth, because
the voice reads "T L F twenty-two ten" while the truth is `TLF-2210`.

| ID | Category | Voice | Spoken question | Expected tool calls | Expected facts |
| --- | --- | --- | --- | --- | --- |
| u_001 | lookup | Samantha (US) | Does the Arcus knee tray come in a cementless version? | `search_products(query="Arcus knee tray")` | Cementless version exists |
| u_002 | lookup | Daniel (UK) | What is the TriLock femoral stem made of? | `search_products(query="TriLock femoral stem")` | Titanium |
| u_003 | lookup | Karen (AU) | Tell me about product O R B fifty-two twelve. | `get_product(sku="ORB-5212")` | Orbis acetabular cup |
| u_004 | numeric | Moira (IE) | How many Veltra four point five millimeter screws come in a box? | `search_products(query="Veltra 4.5 mm screw")` | 12 per box |
| u_005 | numeric | Aman (IN) | What diameters does the Orbis cup come in? | `search_products(query="Orbis cup")` | 46 to 64 mm; 2 mm steps |
| u_006 | numeric | Samantha (US) | What's the list price of T L F twenty-two ten? | `get_product(sku="TLF-2210")` | $1,840 |
| u_007 | multi_step | Daniel (UK) | Order six zero seven four four is late. What's holding it up, and when is that item back in stock? | `get_order_status(order_id="ORD-60744")`, then `get_product(sku="KBC-0040")` | Waiting on Kestrel bone cement; back December 2 |
| u_008 | multi_step | Karen (AU) | Is the TriLock stem in size fourteen in stock, and what does it cost? | `search_products(query="TriLock stem")`, then `get_product(sku="TLF-2210")` | Size 14 in stock; $1,840 |
| u_009 | multi_step | Moira (IE) | Has order five eight two one three shipped, and how many screws come in each box of the Veltra screws in it? | `get_order_status(order_id="ORD-58213")`, then `get_product(sku="VCS-0450")` | Shipped Nov 3; 12 per box |
| u_010 | out_of_scope | Aman (IN) | Who's the best hip surgeon in Denver? | none | Declines; names no surgeon |
| u_011 | out_of_scope | Samantha (US) | How much ibuprofen should I take after knee surgery? | none | Declines medical advice; points to the care team |
| u_012 | out_of_scope | Daniel (UK) | A box of cement arrived damaged and I want to file a complaint. | `escalate_to_human(reason="damaged shipment")` | Hands off to a person |

**Lexicon** (`domain_lexicon`): one row per term with its accepted spellings, e.g.
`TriLock: [tri lock, tri-lock, trilock]`, `TLF-2210: [tlf 2210, t l f 2210, tlf 22 10]`.
The normalizer maps every variant to the canonical term before computing WER, so
"T L F twenty-two ten" and `TLF-2210` compare as equal. Mishearings such as "try lock" are
deliberately not variants, so they still count as errors.

**Audio**: `scripts/make_input_audio.sh` runs macOS `say` once per row
(`--file-format=WAVE --data-format=LEI16@16000`), adds light noise (about 20 dB SNR), and copies
the clips to `input/`. Listen to all 12 once and fix any line the voice garbled; that listen is the
human check on the ground truth.

## 5. Candidates

Every candidate implements the brief's interface: `transcribe(audio_uri)`, `respond(transcript, tools)`, `synthesize(text, out_uri)`.

| Candidate | ASR | Agent LLM | TTS |
| --- | --- | --- | --- |
| `baseline_cascade` | `ai_transcribe` | `databricks-gpt-5-4-mini` with the 4 mock tools | Piper `en_US-lessac-medium` |
| `mock_noisy` (scorer check, off by default) | Ground truth plus planted errors (below) | Same as baseline, plus one planted wrong tool | Baseline TTS, plus one planted truncation |
| `gemini_cascade` | Gemini 3.8 Flash audio input via chat completions | `databricks-gemini-3-8-flash` | Piper `en_GB-alan-medium` |

**Planted failures in `mock_noisy`.** These are fixed per utterance rather than random, so the demo
is scripted and each stage gets a failure it should catch:

| Utterance | Injection | Stage that should catch it |
| --- | --- | --- |
| u_004 | "Veltra" heard as "Vectra" | Stage 1 term error |
| u_006 | `TLF-2210` heard as `TLF-2201`; the lookup fails | Stage 1, plus an ASR-caused Stage 2B failure |
| u_008 | "fourteen" heard as "forty" | Stage 1 number error, plus Stage 2B |
| u_009 | Calls `get_product` only, skipping the order lookup, in both conditions | Stage 2 reasoning error (fails in 2A and 2B) |
| u_005 | TTS audio cut off after 60% of the answer | Stage 3 truncation |

**Reference ASR** for the TTS round trip is Gemini 4 Carbon (`system.ai.gemini-4-carbon`, a Unity
Gateway model service), prompted to transcribe verbatim, for every candidate. It replaced
`ai_transcribe`, which misheard the made-up product names often enough to swamp differences between
voices (4 to 7% round-trip WER against 0 to 1% on the same 48 clips). It is not any candidate's own
ASR, as the brief asks. The verbatim prompt matters: with the looser candidate prompt it once wrote
a customer question that was never spoken. The demo names the reference on every trace.

## 6. Trace contract

One trace per (run_id, candidate, utterance_id, condition). Example: the as-heard trace for u_006:

The run is tagged `voice_run_id` rather than the brief's `run_id`: MLflow's trace filter treats
`tags.run_id` specially and never matches a plain tag of that name (found in the local tests).

```
voice_turn                      tags: voice_run_id, candidate=baseline_cascade, condition=as_heard,
                                      utterance_id=u_006, category=numeric, asr_source=native,
                                      normalizer_version=v1, dataset_version=v1, synthetic_input=true
  inputs : {"question": "What's the list price of TLF-2210?",
            "audio_uri": "/Volumes/pavan_naidu_catalog/mlflow/voice_audio/input/u_006.wav"}
  outputs: {"answer": "The TriLock femoral stem, TLF-2210, lists at $1,840.",
            "tts_uri": "/Volumes/.../voice_audio/tts/<run_id>/baseline_cascade/u_006.wav"}
  ├─ asr          in: {audio_uri, audio_sha256, duration_s, sample_rate_hz}   out: {text, latency_ms}
  ├─ agent        in: {transcript}                                             out: {answer, latency_ms}
  │   ├─ llm      in: chat messages (text only)                                out: tool call
  │   └─ get_product (TOOL)  in: {"sku": "TLF-2210"}                           out: fixture JSON
  ├─ tts          in: {text}                                                   out: {tts_uri, duration_s, latency_ms}
  └─ tts_reasr    in: {tts_uri, model: "ai_transcribe"}                        out: {text}
expectations (logged on the trace): gt_transcript, domain_terms, expected_tool_calls, expected_facts,
                                    expected_numbers, reference_answer
feedback (added by evaluate): asr_wer, tool_name_match, correctness (+ rationale), tts_roundtrip_wer, ...
```

- The root span's request is `{"question": gt_transcript, "audio_uri": ...}` in both conditions.
  MLflow's run comparison pairs traces by request, so each question's two traces line up, and the
  judges grade the answer against what the caller actually asked. What the agent heard is in the
  `asr` span's output and the `agent` span's input.
- Oracle traces (condition `oracle`) have no `asr` span: the agent gets `gt_transcript`. Their
  `tts` span speaks the golden `reference_answer` instead of the agent's answer, so every
  candidate's voice reads the same text (Stage 3A).
- The reference answer is logged as `reference_answer`, not `expected_response`: MLflow's
  Correctness judge raises when a trace has both `expected_facts` and `expected_response`.
- Ground truth sits on the trace as expectations, so the trace view shows heard vs. true text
  side by side.
- **Payload guard**: every `set_inputs` and `set_outputs` goes through `guard()`, which raises if
  the JSON is over 64 KB, contains `bytes`, a `data:audio` URI, or a base64-looking run over 1,000
  characters.
- **No autologging.** Gemini ASR sends the audio as a base64 data URI. `mlflow.openai.autolog()`
  would copy that into a span, so every span is written by hand.

Sketch of the pipeline loop:

```python
with mlflow.start_span(name="voice_turn") as root:
    mlflow.update_current_trace(tags=trace_tags(run_id, cand, utt, condition="as_heard"))
    with mlflow.start_span(name="asr") as s:
        s.set_inputs(guard(utt.audio_ref()))            # path + sha256 + duration, never bytes
        asr = cand.transcribe(utt.audio_uri)
        s.set_outputs(guard({"text": asr.text, "latency_ms": asr.latency_ms}))
    agent = cand.respond(asr.text, TOOLS)               # writes agent, llm and tool spans
    with mlflow.start_span(name="tts") as s:
        s.set_inputs(guard({"text": agent.answer}))
        tts = cand.synthesize(agent.answer, tts_path(run_id, cand, utt))
        s.set_outputs(guard({"tts_uri": tts.uri, "duration_s": tts.duration_s, "latency_ms": tts.latency_ms}))
    with mlflow.start_span(name="tts_reasr") as s:
        s.set_inputs(guard({"tts_uri": tts.uri, "model": "ai_transcribe"}))
        s.set_outputs(guard({"text": reference_asr(tts.uri)}))
    root.set_inputs(guard({"question": utt.gt_transcript, "audio_uri": utt.audio_uri}))
    root.set_outputs(guard({"answer": agent.answer, "tts_uri": tts.uri}))

for name, value in utt.expectations().items():
    mlflow.log_expectation(trace_id=root.trace_id, name=name, value=value)
```

Run size: 2 candidates x 12 utterances x 2 conditions = 48 traces.

## 7. Stages and scorers

All Stage 1 and Stage 3 scorers are code (no LLM). LLM judges are used only in Stage 2, on text.

| Stage | Scorer | Type | Definition |
| --- | --- | --- | --- |
| 1 ASR | `asr_wer` | code | jiwer WER after normalization; rationale shows truth vs heard |
| 1 | `asr_term_error_rate` | code | Share of `domain_terms` missing in the heard text (lexicon variants count) |
| 1 | `asr_number_accuracy` | code | Share of ground-truth numbers present exactly |
| 1 | `asr_exact_match` | code | Normalized heard == normalized truth |
| 2 Reasoning | `tool_name_match` | code | Called tool set == expected set (`none` for declines) |
| 2 | `tool_args_match` | code | Matched key args / expected key args, after normalization (`60744` == `ORD-60744`) |
| 2 | `correctness` | built-in `Correctness`, judge `databricks:/databricks-claude-opus-5-5` | Answer contains `expected_facts` |
| 2 | `voice_guidelines` | built-in `Guidelines`, same judge | Declines out-of-scope requests; states units for numbers |
| 2 | `spoken_suitability` | code | 60 words or fewer; no markdown, lists, tables or URLs |
| 3 TTS | `tts_roundtrip_wer` | code | WER between the spoken text and its re-transcription |
| 3 | `tts_term_accuracy` | code | Domain terms in the answer that come back intact |
| 3 | `tts_number_accuracy` | code | Numbers, sizes and SKUs that come back exactly |
| 3 | `tts_speaking_rate` | code | Words per minute; flag outside 120 to 190 |
| 3 | `tts_truncation` | code | Re-transcribed word count under 80% of intended |
| 3 human | `human_intelligible`, `human_naturalness`, `human_pronunciation_issue` | human feedback from the app | 4 clips per candidate (one per category), saved on the trace |

Stage 2 and Stage 3 each run in both conditions. Stage 2A (true transcript) against 2B (as heard)
gives the ASR impact. Stage 3A speaks the reference answer, the same text for every candidate, so
it compares the voices alone; Stage 3B speaks the candidate's own answer, which is what a caller
hears and also reflects how the agent phrases things (codes, numbers, length).

Example code scorer:

```python
@scorer
def asr_wer(trace, expectations):
    heard = trace.search_spans(name="asr")[0].outputs["text"]
    truth = expectations["gt_transcript"]
    value = jiwer.wer(normalize(truth), normalize(heard))
    return Feedback(value=value, rationale=f"truth: {truth!r} | heard: {heard!r}")
```

The normalizer (`normalize.py`, `NORMALIZER_VERSION = "v1"`): lowercase; strip punctuation; join
spelled letters ("t l f" to "tlf"); number words to digits ("twenty two" to "22"); map lexicon
variants to canonical terms. Its version is a tag on every trace.

## 8. Evaluation flow (notebook 03)

1. `mlflow.search_traces(locations=[EXPERIMENT_ID], filter_string="tags.voice_run_id = '<id>'")`, then split by candidate and condition.
2. One `mlflow.genai.evaluate(data=<traces>, scorers=...)` call per candidate and condition, each
   its own MLflow run named `eval/<candidate>/<condition>`. No `predict_fn`: the traces are scored
   as they are, and the feedback lands on the same trace IDs. As-heard runs get the Stage 1, 2
   and 3 scorers; oracle runs get Stage 2 and Stage 3 (on the reference answer).
3. Write the long-format `eval_scorecard` (run_id, candidate, stage, metric, value, n, dataset_version, created_at).
4. Show, all marked **Illustrative data**:
   - an overview with one row per candidate: Stage 1 WER and term error rate, Stage 2A and 2B tool match and correctness, **ASR impact** (2A minus 2B), Stage 3 round-trip WER and number accuracy, p50 latency per stage;
   - an attribution table for each Stage 2B failure: *ASR-caused* (2A passed, 2B failed, a key term was misheard) or *reasoning* (both failed);
   - the 5 worst utterances per stage, with trace ID, a link to the trace, and inline players for the input and TTS audio (`IPython.display.Audio` reading the Volume path).

## 9. Build steps

| # | Step | Done when |
| --- | --- | --- |
| 0 | **Smoke test the risky pieces** on FEVM: (a) `ai_transcribe` on one WAV (an admin must turn on the AI Transcribe preview); (b) Gemini audio input via chat completions; (c) Piper installs and writes a WAV to the Volume, with voice files staged in `models/piper/`; (d) one UC-stored trace takes `log_expectation`, then `evaluate(data=traces)` adds feedback to that same trace | All four work, or a fallback is chosen |
| 1 | Data: `golden_v1.jsonl`, `lexicon_v1.csv`, tool fixtures, input audio script | 12 WAVs heard and checked |
| 2 | `01_setup`: schema, Volume, tables, audio upload with sha256 and duration, experiment with UC trace location | Idempotent re-run |
| 3 | `02_run_pipeline`: the `baseline_cascade` and `mock_noisy` adapters, mock tools, tracing helpers and payload guard; write `pipeline_outputs` | 48 traces, guard passes, every audio path exists |
| 4 | `03_evaluate`: scorers, evaluate runs, scorecard, ASR impact, attribution, worst-N with players | `mock_noisy` worse on term error rate, tool match and number accuracy; each planted failure attributed to its stage |
| 5 | Tests: `test_normalize.py`, `test_scorers.py`, `test_payload_guard.py` | `pytest` passes locally |
| 6 | App (see section 10), deployed to FEVM | Plays both clips for any trace and saves a human rating as feedback |
| 7 | `gemini_cascade` adapter (one file plus one config block), README with limitations, demo script, row in the repo README | Re-running 02 and 03 adds a third scorecard row with no scorer changes |

## 10. App: Voice Eval Inspector

A Databricks App with a React (Vite) frontend and a FastAPI backend that fills the one gap MLflow
has here, which is playing audio. FastAPI serves the API under `/api` and the built React files
at `/`. Databricks Apps runs `npm install` and `npm run build` on deploy, so the build happens on
the platform.

- **Compare**: pick an utterance and see each candidate side by side: play the input audio, truth
  vs. heard with differences highlighted, tool calls vs. expected, the answer, play the TTS audio,
  the round-trip text, key scores, and an "Open trace in MLflow" link.
- **Listen and rate**: 4 TTS clips per candidate (one per category). The reviewer answers
  intelligible (yes/no), naturalness (1 to 5) and pronunciation issues (free text). Each answer is
  written with `mlflow.log_feedback(trace_id, ..., source=HUMAN <reviewer email>)`, so it shows up
  on the trace next to the automatic Stage 3 scores.
- Audio comes from the Volume through the Files API. Scorecard data comes from the Delta tables
  through the SQL warehouse.
- The app's service principal needs: READ VOLUME, SELECT on the tables, CAN_EDIT on the
  experiment, and MODIFY plus SELECT on the UC trace tables.

## 11. Known limitations (state them in the demo)

- MLflow doesn't play audio. Traces hold Volume paths; playback happens in the notebook or app.
- Round-trip ASR measures intelligibility, not naturalness. Naturalness comes only from the human sample.
- Reference ASR errors add noise to Stage 3, and a strong LLM transcriber can repair a mispronounced word from context. A single fixed reference (Gemini 4 Carbon, verbatim prompt) is used and named; people check pronunciation in the listening sample.
- Input audio is synthetic, which flatters ASR. Every chart is flagged.
- `ai_transcribe` is Beta, region-limited, and English and Spanish only.
- No TTS model is served on Databricks. Piper runs on CPU inside the notebook; a hosted TTS would
  plug in as another adapter.
- Judges are secondary to the code scorers that compare against ground truth. The judge model is
  recorded on every assessment.

## 12. Demo script (about 10 minutes)

1. **The problem** (1 min): a voice agent can fail in three places, and one blended score hides which.
2. **The data** (1 min): the golden table, the Volume of clips, and the text-only trace contract.
3. **A run** (1 min): show the experiment's traces; open one and point at the Volume paths.
4. **Stage 1** (2 min): WER and term errors; open u_004 ("Vectra"), play the clip, read the rationale.
5. **Stage 2** (2 min): oracle vs. as-heard and the ASR impact number; open u_006, where a misheard
   SKU led the agent to the wrong product. Then u_009, which fails in both conditions, so it's a
   reasoning error.
6. **Stage 3** (2 min): round-trip WER; u_005's truncation; rate a clip in the app and watch the
   feedback appear on the trace.
7. **Wrap** (1 min): add `gemini_cascade` (one adapter, one config block), re-run, and the
   scorecard becomes the model-selection decision.

## 13. Repo layout

```
voice-agent-evals/
  PLAN.md
  README.md
  config.yaml                 # UC names, warehouse, models, candidates, thresholds
  voice_evals/
    normalize.py              # versioned normalizer
    tools.py                  # 4 mock tools + fixtures
    candidates.py             # interface + baseline_cascade, mock_noisy, gemini_cascade
    tracing.py                # span helpers + payload guard
    scorers.py                # stage 1/2/3 scorers
  data/
    golden_v1.jsonl
    lexicon_v1.csv
  scripts/make_input_audio.sh
  notebooks/
    01_setup.ipynb
    02_run_pipeline.ipynb
    03_evaluate.ipynb
  app/                        # Databricks App: FastAPI backend + React frontend
    app.py  app.yaml  requirements.txt  package.json
    frontend/                 # Vite + React source
  tests/
```
