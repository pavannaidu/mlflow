# MLflow Examples

Examples of evaluating GenAI applications with MLflow on Databricks. Each folder is self-contained,
and its README lists the MLflow features it uses.

| Folder | What it shows |
| --- | --- |
| [`call-transcript-evals/`](call-transcript-evals/) | Structured extraction from call transcripts with `ai_query()`, a UC-backed evaluation dataset, eight scorers of four kinds (built-in, Guidelines, custom LLM, code), and a three-model comparison with `mlflow.genai.evaluate()` |
| [`voice-agent-evals/`](voice-agent-evals/) | A three-stage voice agent evaluation (hearing, reasoning, speaking): text-only traces that reference audio in a UC Volume, offline scoring of logged traces, built-in judges, human feedback on traces, and a React + FastAPI Databricks App to review turns and play the audio |

## Layout

```
call-transcript-evals/
  Call Transcript Evaluation.ipynb   one notebook, end to end
  README.md
voice-agent-evals/
  voice_evals/   Python package: pipeline, tracing, scorers, evaluation
  notebooks/     00 to 03, run as serverless jobs
  app/           Voice Eval Inspector (FastAPI + React)
  data/  scripts/  tests/
  README.md      what it uses in MLflow, how to run and deploy it
  PLAN.md        design, and how it maps to the build brief
```

## Setup

* A Databricks workspace with serverless compute (environment 5) and a SQL warehouse
* `mlflow[databricks]` 3.15 or later
* Access to Foundation Model APIs; the voice demo also uses the `ai_transcribe` AI Function and a
  Unity Gateway model service
* Both examples write to `pavan_naidu_catalog.mlflow` (voice objects are prefixed `voice_`). Change
  the catalog and schema in the notebook's first cell, or in `voice-agent-evals/voice_evals/config.py`.
