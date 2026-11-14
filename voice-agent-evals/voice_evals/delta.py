"""Delta table helpers for the notebooks. Every table carries an "Illustrative data" comment."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

COMMENT = "Illustrative data. Voice Agent Evals MVP demo; every product, SKU and order is made up."

SCHEMAS = {
    "golden_utterances": (
        "utterance_id STRING, category STRING, voice STRING, spoken_text STRING, gt_transcript STRING, "
        "domain_terms ARRAY<STRING>, expected_tool_calls STRING, expected_facts ARRAY<STRING>, "
        "expected_numbers ARRAY<STRING>, audio_uri STRING, audio_sha256 STRING, duration_s DOUBLE, "
        "sample_rate_hz INT, synthetic_input BOOLEAN, dataset_version STRING, reference_answer STRING"
    ),
    "domain_lexicon": "term STRING, category STRING, variants ARRAY<STRING>",
    "pipeline_outputs": (
        "run_id STRING, candidate STRING, condition STRING, utterance_id STRING, category STRING, "
        "trace_id STRING, asr_text STRING, agent_answer STRING, tool_calls STRING, tts_uri STRING, "
        "tts_duration_s DOUBLE, reasr_text STRING, latency_ms_asr DOUBLE, latency_ms_agent DOUBLE, "
        "latency_ms_tts DOUBLE, error STRING, created_at TIMESTAMP, dataset_version STRING, models STRING"
    ),
    "eval_results": (
        "run_id STRING, candidate STRING, condition STRING, utterance_id STRING, category STRING, "
        "trace_id STRING, stage STRING, metric STRING, value DOUBLE, value_text STRING, rationale STRING, "
        "source_type STRING, source_id STRING"
    ),
    "eval_scorecard": (
        "run_id STRING, candidate STRING, condition STRING, stage STRING, metric STRING, agg STRING, "
        "value DOUBLE, n BIGINT, dataset_version STRING, created_at TIMESTAMP"
    ),
    "attribution": (
        "run_id STRING, candidate STRING, utterance_id STRING, category STRING, oracle_pass BOOLEAN, "
        "as_heard_pass BOOLEAN, asr_wrong BOOLEAN, attribution STRING, tts_issue BOOLEAN, "
        "asr_wer DOUBLE, tts_roundtrip_wer DOUBLE"
    ),
}


def _columns(schema: str) -> list[tuple[str, str]]:
    return [(c.strip().split()[0], c.strip().split(None, 1)[1].upper()) for c in schema.split(", ")]


def _coerce(value, sql_type: str):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if sql_type == "TIMESTAMP" and isinstance(value, str):
        return dt.datetime.fromisoformat(value)
    if sql_type in ("DOUBLE",):
        return float(value)
    if sql_type in ("INT", "BIGINT"):
        return int(value)
    if sql_type == "BOOLEAN":
        return bool(value)
    if sql_type.startswith("ARRAY"):
        return list(value)
    if sql_type == "STRING" and not isinstance(value, str):
        return json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return value


def to_spark(spark, rows, name: str):
    schema = SCHEMAS[name]
    records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else rows
    cols = _columns(schema)
    data = [tuple(_coerce(r.get(c), t) for c, t in cols) for r in records]
    return spark.createDataFrame(data, schema=schema)


def ensure_table(spark, table: str, name: str) -> None:
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table} ({SCHEMAS[name]}) COMMENT '{COMMENT}'")


def replace_where(spark, table: str, name: str, rows, where: str) -> int:
    """Idempotent: delete the rows matching `where`, then append the new ones. New columns in the
    schema are added to an existing table (mergeSchema)."""
    ensure_table(spark, table, name)
    spark.sql(f"DELETE FROM {table} WHERE {where}")
    to_spark(spark, rows, name).write.mode("append").option("mergeSchema", "true").saveAsTable(table)
    return len(rows)


def replace_run(spark, table: str, name: str, rows, run_id: str) -> int:
    """Idempotent per run: replace this run's rows."""
    return replace_where(spark, table, name, rows, f"run_id = '{run_id}'")


def overwrite(spark, table: str, name: str, rows) -> int:
    to_spark(spark, rows, name).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    spark.sql(f"COMMENT ON TABLE {table} IS '{COMMENT}'")
    return len(rows)
