# Call Transcript Evaluation

Extract structured insights from customer support call transcripts with `ai_query()`, then score
the extractions offline with MLflow GenAI evaluation and compare three models on the same dataset.

Everything is in one notebook: [`Call Transcript Evaluation.ipynb`](Call%20Transcript%20Evaluation.ipynb).
Run it on serverless (CPU, environment 5) with `mlflow[databricks]==3.15`.

**Illustrative data.** Acme, the agents, customers, products and amounts are all made up.

## What it does

1. **Source data.** Creates `call_transcripts` (8 synthetic calls: a router fix, repeated double
   billing, a plan upgrade, a firmware issue, a cancellation threat, a two-issue call, a security and
   compliance inquiry, and an outage escalation).
2. **Batch extraction.** One `ai_query()` over the table, with a strict JSON schema
   (`responseFormat`), writes `call_insights`: summary, issue, resolution, sentiment
   (Positive, Negative, Neutral) and category (6 fixed values).
3. **Evaluation dataset.** Registers each transcript with human ground-truth labels for sentiment and
   category.
4. **Scorers.** Defines eight scorers of four kinds (below) and registers them to the experiment.
5. **Model comparison.** Runs `mlflow.genai.evaluate()` once per model, each in its own MLflow run,
   then puts the metrics side by side.

## What it uses in MLflow

| MLflow feature | How the notebook uses it |
| --- | --- |
| Experiment with traces in Unity Catalog | `mlflow.set_experiment(experiment_name=..., trace_location=UnityCatalog(catalog_name, schema_name))`. The trace location is set once and can't change; the notebook falls back to the plain experiment if it already exists. `MLFLOW_TRACING_SQL_WAREHOUSE_ID` points trace reads at a SQL warehouse. |
| Evaluation dataset (UC-backed) | `mlflow.genai.datasets.create_dataset(name="<catalog>.<schema>.call_insights_eval_dataset")`, or `get_dataset` if it exists, then `dataset.merge_records(...)` with `inputs` (the transcript) and `expectations` (`expected_sentiment`, `expected_category`). |
| Built-in judge | `Safety()`, zero configuration. |
| Guidelines judges | `Guidelines(name=..., guidelines=[...])` for `faithfulness` (nothing in the extraction that the transcript doesn't support), `completeness` (reason, action and outcome all present) and `classification_accuracy` (sentiment and category fit the call). Pass or fail, with a rationale. |
| Custom LLM scorer | `@scorer professionalism`: a 1 to 5 score from its own prompt, called through `ai_query()`, returned as `Feedback(value, rationale, source=AssessmentSource(LLM_JUDGE, ...))`. |
| Code scorers | `@scorer schema_validity` (required fields and enum values), `ground_truth_match` (sentiment and category against the labels in `expectations`) and `summary_length` (50 to 500 characters). Fast and reproducible. |
| Scorer registration | `scorer.register(experiment_id=...)` and `mlflow.genai.list_scorers(...)`, so the same scorers can be fetched later with `get_scorer()` or reused for monitoring. |
| Evaluation with a predict function | `mlflow.genai.evaluate(data=dataset, predict_fn=make_predict_fn(model), scorers=ALL_SCORERS)`. The predict function calls `ai_query()` with the model and the same JSON schema; evaluation traces each call automatically. |
| Runs for comparison | One `mlflow.start_run(run_name="call-insights-eval-<model>")` per model, with `mlflow.log_params` (model, task, response format, record count). The notebook collects `result.metrics` into one table, and the MLflow UI can compare the runs. |

Models compared (Unity Gateway names): `system.ai.meta-llama-3-3-70b-instruct`,
`system.ai.gpt-5-4-mini` and `system.ai.gemini-3-5-flash`. The batch extraction in step 2 and the
`professionalism` scorer use `databricks-meta-llama-3-3-70b-instruct`.

## What it creates

In `pavan_naidu_catalog.mlflow` (change `CATALOG`, `SCHEMA`, `WAREHOUSE_ID` and `EXPERIMENT_NAME` in
the first code cell for your workspace):

| Object | What it holds |
| --- | --- |
| `call_transcripts` | The 8 raw transcripts |
| `call_insights` | The batch extraction from `ai_query()` |
| `call_insights_eval_dataset` | The MLflow evaluation dataset: transcripts plus ground-truth labels |
| Experiment *Call Transcript Evaluation Experiment* | One evaluation run per model, its traces (stored in UC, in tables prefixed with the experiment ID) and the registered scorers |

## Notes

- The source table is recreated on every run (`CREATE OR REPLACE`); the evaluation dataset is
  merged, so re-running doesn't duplicate records.
- The `professionalism` judge and the batch extraction use the same Llama model, so that scorer
  grades the Llama run with its own model family; the Guidelines judges use the default judge model.
- With 8 records, a single changed verdict moves a metric by 12.5 points. Treat the comparison as
  illustrative.
