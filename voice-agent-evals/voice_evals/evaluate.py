"""Offline evaluation over logged traces, plus the per-stage scorecard and error attribution.

Inference and evaluation are separate on purpose: traces are logged first, then scored here with
`mlflow.genai.evaluate(data=<traces>)` and no predict_fn. Feedback lands on the same trace IDs,
so opening a trace shows its transcript, tool calls, scores and rationales together.
"""

from __future__ import annotations

import datetime as dt
import time

import mlflow
import pandas as pd

from .normalize import NORMALIZER_VERSION
from .scorers import LOWER_IS_BETTER, METRIC_STAGE, scorers_for

REASONING_METRICS = ["tool_name_match", "tool_args_match", "correctness", "voice_guidelines", "answer_exact_numeric"]


def load_run_traces(experiment_id: str, run_id: str, retries: int = 6, wait_s: float = 10) -> list:
    """This run's traces, latest successful one per (candidate, utterance, condition).

    Spans of UC-stored traces can arrive a little after the trace itself, so this waits until
    every trace has its spans (scoring a trace without spans would silently skip it).
    """
    for attempt in range(retries + 1):
        traces = mlflow.search_traces(
            locations=[experiment_id],
            filter_string=f"tags.voice_run_id = '{run_id}'",
            return_type="list",
            max_results=5000,
        )
        latest: dict[tuple, object] = {}
        for t in traces:
            if str(t.info.state).split(".")[-1] != "OK":
                continue
            tags = t.info.tags
            key = (tags.get("candidate"), tags.get("utterance_id"), tags.get("condition"))
            if key not in latest or t.info.request_time > latest[key].info.request_time:
                latest[key] = t
        if attempt == retries or all(t.data.spans for t in latest.values()):
            return list(latest.values())
        time.sleep(wait_s)
    return []


def run_evaluations(traces: list, *, run_id: str, judge_model: str, include_judges: bool = True,
                    log=print) -> dict[tuple, str]:
    """One MLflow evaluation run per (candidate, condition), so the runs compare side by side."""
    groups: dict[tuple, list] = {}
    for t in traces:
        groups.setdefault((t.info.tags["candidate"], t.info.tags["condition"]), []).append(t)
    eval_runs = {}
    for (candidate, condition), group in sorted(groups.items()):
        with mlflow.start_run(run_name=f"eval/{candidate}/{condition}") as run:
            mlflow.set_tags({
                "pipeline_run_id": run_id, "candidate": candidate, "condition": condition,
                "judge_model": judge_model, "normalizer_version": NORMALIZER_VERSION,
                "illustrative_data": "true",
            })
            mlflow.genai.evaluate(data=group, scorers=scorers_for(condition, judge_model, include_judges))
            eval_runs[(candidate, condition)] = run.info.run_id
        log(f"evaluated {candidate}/{condition}: {len(group)} traces")
    return eval_runs


def _numeric(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.lower() in ("yes", "no"):
        return 1.0 if value.lower() == "yes" else 0.0
    return None


def collect_results(experiment_id: str, run_id: str) -> pd.DataFrame:
    """Long table: one row per (trace, metric), latest assessment wins. Includes human feedback."""
    rows = []
    for t in load_run_traces(experiment_id, run_id):
        tags = t.info.tags
        latest = {}
        for a in t.info.assessments or []:
            if getattr(a, "feedback", None) is None or getattr(a, "valid", True) is False:
                continue
            if a.name not in latest or (a.create_time_ms or 0) > (latest[a.name].create_time_ms or 0):
                latest[a.name] = a
        for name, a in latest.items():
            value = a.feedback.value
            source = a.source
            rows.append({
                "run_id": run_id,
                "candidate": tags.get("candidate"),
                "condition": tags.get("condition"),
                "utterance_id": tags.get("utterance_id"),
                "category": tags.get("category"),
                "trace_id": t.info.trace_id,
                "stage": METRIC_STAGE.get(name, "human" if source and str(source.source_type).endswith("HUMAN") else "other"),
                "metric": name,
                "value": _numeric(value),
                "value_text": None if value is None else str(value),
                "rationale": a.rationale,
                "source_type": str(source.source_type).split(".")[-1] if source else None,
                "source_id": source.source_id if source else None,
            })
    return pd.DataFrame(rows)


def scorecard(results: pd.DataFrame, *, run_id: str, dataset_version: str) -> pd.DataFrame:
    """Long-format scorecard (the brief's eval_scorecard), easy to pivot in SQL or a dashboard."""
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    numeric = results.dropna(subset=["value"])
    out = []
    for (candidate, condition, stage, metric), g in numeric.groupby(["candidate", "condition", "stage", "metric"]):
        aggs = {"mean": g["value"].mean()}
        if metric in LOWER_IS_BETTER or metric == "tts_speaking_rate":
            aggs.update(median=g["value"].median(), p90=g["value"].quantile(0.9))
        for agg, value in aggs.items():
            out.append({
                "run_id": run_id, "candidate": candidate, "condition": condition, "stage": stage,
                "metric": metric, "agg": agg, "value": round(float(value), 4), "n": len(g),
                "dataset_version": dataset_version, "created_at": created,
            })
    return pd.DataFrame(out)


def asr_impact(results: pd.DataFrame) -> pd.DataFrame:
    """Score(2A oracle) minus score(2B as heard) per reasoning metric: the ASR error carried into reasoning."""
    r = results[results["metric"].isin(REASONING_METRICS)].dropna(subset=["value"])
    means = r.groupby(["candidate", "metric", "condition"])["value"].mean().unstack("condition")
    for col in ("oracle", "as_heard"):
        if col not in means:
            means[col] = float("nan")
    means["asr_impact"] = means["oracle"] - means["as_heard"]
    return means[["oracle", "as_heard", "asr_impact"]].round(3).reset_index()


def attribution(results: pd.DataFrame) -> pd.DataFrame:
    """Label each turn: pass, ASR-caused (2A passed, 2B failed, ASR got something wrong),
    reasoning (both failed), or flaky (passes and failures that ASR doesn't explain).
    A condition passes when the answer is correct and the right tools got the right key args."""
    wide = results.pivot_table(index=["candidate", "utterance_id", "category", "condition"],
                               columns="metric", values="value", aggfunc="last")
    rows = []
    for (candidate, utt, category), g in wide.groupby(level=[0, 1, 2]):
        def passed(cond):
            if cond not in g.index.get_level_values("condition"):
                return None
            rec = g.xs(cond, level="condition").iloc[0]
            checks = [rec.get(m) for m in ("correctness", "tool_name_match", "tool_args_match")
                      if pd.notna(rec.get(m))]
            return all(v >= 1.0 for v in checks) if checks else None

        oracle, heard = passed("oracle"), passed("as_heard")
        asr = g.xs("as_heard", level="condition").iloc[0] if "as_heard" in g.index.get_level_values("condition") else pd.Series(dtype=float)
        asr_wrong = any([
            pd.notna(asr.get("asr_term_error_rate")) and asr.get("asr_term_error_rate") > 0,
            pd.notna(asr.get("asr_number_accuracy")) and asr.get("asr_number_accuracy") < 1,
        ])
        tts_issue = any([
            pd.notna(asr.get("tts_complete")) and asr.get("tts_complete") < 1,
            pd.notna(asr.get("tts_number_accuracy")) and asr.get("tts_number_accuracy") < 1,
            pd.notna(asr.get("tts_term_accuracy")) and asr.get("tts_term_accuracy") < 1,
        ])
        if oracle and heard:
            label = "pass"
        elif oracle and heard is False:
            label = "ASR-caused" if asr_wrong else "flaky"
        elif oracle is False and heard is False:
            label = "reasoning"
        elif oracle is False and heard:
            label = "flaky"
        else:
            label = "incomplete"
        rows.append({
            "candidate": candidate, "utterance_id": utt, "category": category,
            "oracle_pass": oracle, "as_heard_pass": heard, "asr_wrong": asr_wrong,
            "attribution": label, "tts_issue": tts_issue,
            "asr_wer": asr.get("asr_wer"), "tts_roundtrip_wer": asr.get("tts_roundtrip_wer"),
        })
    return pd.DataFrame(rows)


def overview(results: pd.DataFrame, pipeline_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per candidate: the headline numbers for every stage."""
    def mean(metric, condition):
        r = results[(results["metric"] == metric) & (results["condition"] == condition)]
        return r.groupby("candidate")["value"].mean()

    cols = {
        "S1 WER": mean("asr_wer", "as_heard"),
        "S1 term err": mean("asr_term_error_rate", "as_heard"),
        "S1 number acc": mean("asr_number_accuracy", "as_heard"),
        "S2A tool match": mean("tool_name_match", "oracle"),
        "S2B tool match": mean("tool_name_match", "as_heard"),
        "S2A correct": mean("correctness", "oracle"),
        "S2B correct": mean("correctness", "as_heard"),
        "S3 round-trip WER (reference)": mean("tts_roundtrip_wer", "oracle"),
        "S3 round-trip WER": mean("tts_roundtrip_wer", "as_heard"),
        "S3 number acc": mean("tts_number_accuracy", "as_heard"),
    }
    df = pd.DataFrame(cols)
    df["ASR impact (correct)"] = df["S2A correct"] - df["S2B correct"]
    if pipeline_rows is not None and len(pipeline_rows):
        lat = pipeline_rows[pipeline_rows["condition"] == "as_heard"].groupby("candidate")[
            ["latency_ms_asr", "latency_ms_agent", "latency_ms_tts"]].median()
        lat.columns = ["p50 ms ASR", "p50 ms agent", "p50 ms TTS"]
        df = df.join(lat)
    return df.round(3).reset_index().rename(columns={"index": "candidate"})


def worst(results: pd.DataFrame, metric: str, n: int = 5, condition: str = "as_heard") -> pd.DataFrame:
    r = results[(results["metric"] == metric) & (results["condition"] == condition)].dropna(subset=["value"])
    ascending = metric not in LOWER_IS_BETTER
    return r.sort_values("value", ascending=ascending).head(n)[
        ["candidate", "utterance_id", "category", "value", "rationale", "trace_id"]]
