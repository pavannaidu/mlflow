"""Voice Eval Inspector: a FastAPI backend that serves the built React frontend.

MLflow holds the text of every voice turn; the audio stays in the UC Volume. This app puts the two
side by side: it reads the result tables through the SQL warehouse, streams clips from the Volume
through the Files API, links each turn to its trace inside its MLflow evaluation run, and writes
human listening labels back to the traces as MLflow feedback.
"""

from __future__ import annotations

import decimal
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger("voice-eval-inspector")

CATALOG = os.getenv("VOICE_CATALOG", "pavan_naidu_catalog")
SCHEMA = os.getenv("VOICE_SCHEMA", "mlflow")
VOLUME = os.getenv("VOICE_VOLUME", "voice_audio")
VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "")
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
os.environ.setdefault("MLFLOW_TRACING_SQL_WAREHOUSE_ID", WAREHOUSE_ID)

cfg = Config()
workspace = WorkspaceClient()
app = FastAPI(title="Voice Eval Inspector")

try:
    WORKSPACE_ID = str(workspace.get_workspace_id())
except Exception:  # noqa: BLE001 - only used for the optional ?o= in MLflow links
    WORKSPACE_ID = ""


def table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.voice_{name}"


# ---------------------------------------------------------------------------------------------
# SQL warehouse access (app service principal)
# ---------------------------------------------------------------------------------------------

_conn = None
_conn_lock = threading.Lock()


def _connect():
    return sql.connect(
        server_hostname=cfg.host.replace("https://", "").rstrip("/"),
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: cfg.authenticate,
    )


def _plain(value):
    """ARRAY columns come back as numpy arrays and DECIMAL as Decimal; make both JSON-friendly."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, decimal.Decimal):
        return float(value)
    return value


def query(statement: str, params: dict | None = None) -> list[dict]:
    global _conn
    with _conn_lock:
        for attempt in range(2):
            try:
                if _conn is None:
                    _conn = _connect()
                with _conn.cursor() as cur:
                    cur.execute(statement, params or {})
                    cols = [d[0] for d in cur.description]
                    return [{c: _plain(v) for c, v in zip(cols, row)} for row in cur.fetchall()]
            except Exception:
                _conn = None
                if attempt:
                    raise
    return []


# Result tables change only when a notebook runs, so reads are cached briefly; tab switches
# then don't wait on the warehouse.
CACHE_TTL_S = 120
_cache: dict[tuple, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def remember(key: tuple, load, ttl: float = CACHE_TTL_S):
    """load() once per `ttl` seconds for this key."""
    with _cache_lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = load()
    with _cache_lock:
        _cache[key] = (time.time(), value)
    return value


def cached(statement: str, params: dict | None = None, ttl: float = CACHE_TTL_S) -> list[dict]:
    """Like query(), but reuses results for `ttl` seconds. Returns copies, so callers may edit rows."""
    key = ("sql", statement, json.dumps(params or {}, sort_keys=True))
    return [dict(r) for r in remember(key, lambda: query(statement, params), ttl)]


# ---------------------------------------------------------------------------------------------
# Links into MLflow. A trace opens only from its evaluation run's details page; the experiment's
# trace list ignores the selected trace.
# ---------------------------------------------------------------------------------------------


def mlflow_url(path: str, **params) -> str:
    query_string = urlencode({**({"o": WORKSPACE_ID} if WORKSPACE_ID else {}), **params})
    return f"{cfg.host.rstrip('/')}/ml/experiments/{EXPERIMENT_ID}/{path}?{query_string}"


def eval_runs(run_id: str) -> dict[tuple[str, str], str]:
    """The MLflow evaluation run for each (candidate, condition) of a pipeline run, newest first."""
    def load():
        import mlflow

        runs = mlflow.search_runs(
            experiment_ids=[EXPERIMENT_ID], filter_string=f"tags.pipeline_run_id = '{run_id}'",
            order_by=["attributes.start_time DESC"], output_format="list", max_results=200)
        found: dict[tuple[str, str], str] = {}
        for r in runs:
            key = (r.data.tags.get("candidate"), r.data.tags.get("condition"))
            if r.info.run_name == f"eval/{key[0]}/{key[1]}":
                found.setdefault(key, r.info.run_id)
        return found

    if not EXPERIMENT_ID:
        return {}
    return remember(("eval_runs", run_id), load)


def trace_url(run_id: str, candidate: str, condition: str, trace_id: str | None) -> str | None:
    if not trace_id or not EXPERIMENT_ID:
        return None
    eval_run = eval_runs(run_id).get((candidate, condition))
    if eval_run:
        return mlflow_url(f"evaluation-runs/{eval_run}/details", selectedEvaluationId=trace_id)
    return mlflow_url("traces", startTimeLabel="ALL", selectedEvaluationId=trace_id)  # not evaluated yet


@app.on_event("startup")
def _warm_up():
    """Wake the serverless warehouse, and read the latest run's ratings, while the first visitor loads the page."""
    def warm():
        latest = next((r for r in get_runs() if r["evaluated"]), None)
        if latest:
            run_labels(latest["run_id"])
            eval_runs(latest["run_id"])

    threading.Thread(target=lambda: _safe(warm), daemon=True).start()


def _safe(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        log.warning("warm-up failed: %s", e)


@app.get("/api/config")
def get_config():
    return {
        "catalog": CATALOG, "schema": SCHEMA, "volume_root": VOLUME_ROOT,
        "experiment_id": EXPERIMENT_ID, "host": cfg.host.rstrip("/"), "workspace_id": WORKSPACE_ID,
    }


@app.get("/api/runs")
def get_runs():
    return cached(f"""
        SELECT p.run_id, min(p.created_at) AS created_at, array_sort(collect_set(p.candidate)) AS candidates,
               count(*) AS turns, count(DISTINCT p.utterance_id) AS questions, count(p.error) AS errors,
               max(coalesce(p.dataset_version, 'v1')) AS dataset_version, max(s.run_id) IS NOT NULL AS evaluated
        FROM {table('pipeline_outputs')} p
        LEFT JOIN (SELECT DISTINCT run_id FROM {table('eval_scorecard')}) s ON s.run_id = p.run_id
        GROUP BY p.run_id ORDER BY p.run_id DESC""", ttl=30)


def candidate_models(run_id: str) -> dict[str, dict | None]:
    """Each candidate's models as the pipeline recorded them, e.g. {"asr": ..., "llm": ..., "tts": ...}.
    None for runs written before the models column existed."""
    rows = cached(f"SELECT candidate, max(models) AS models FROM {table('pipeline_outputs')} "
                  "WHERE run_id = :run_id GROUP BY candidate", {"run_id": run_id})
    return {r["candidate"]: json.loads(r["models"]) if r["models"] else None for r in rows}


@app.get("/api/runs/{run_id}/scorecard")
def get_scorecard(run_id: str):
    """Per candidate: models, metric means by condition, p50 latency, failure attribution, TTS flags."""
    means = cached(
        f"SELECT candidate, condition, metric, value FROM {table('eval_scorecard')} "
        "WHERE run_id = :run_id AND agg = 'mean'", {"run_id": run_id})
    latency = cached(
        f"""SELECT candidate, percentile_approx(latency_ms_asr, 0.5) AS asr,
                   percentile_approx(latency_ms_agent, 0.5) AS agent, percentile_approx(latency_ms_tts, 0.5) AS tts
            FROM {table('pipeline_outputs')} WHERE run_id = :run_id AND condition = 'as_heard' GROUP BY candidate""",
        {"run_id": run_id})
    attribution = cached(
        f"SELECT candidate, attribution, count(*) AS n, count_if(tts_issue) AS tts_flags "
        f"FROM {table('attribution')} WHERE run_id = :run_id GROUP BY candidate, attribution", {"run_id": run_id})

    names = sorted({m["candidate"] for m in means} | {r["candidate"] for r in latency})
    models = candidate_models(run_id)
    runs = eval_runs(run_id)
    out = []
    for c in names:
        metrics: dict[str, dict[str, float]] = {}
        for m in means:
            if m["candidate"] == c:
                metrics.setdefault(m["metric"], {})[m["condition"]] = m["value"]
        lat = next((r for r in latency if r["candidate"] == c), {})
        mine = [r for r in attribution if r["candidate"] == c]
        heard, oracle = runs.get((c, "as_heard")), runs.get((c, "oracle"))
        out.append({
            "candidate": c,
            "models": models.get(c),
            # MLflow's own side-by-side view of the two evaluation runs
            "compare_url": mlflow_url("evaluation-runs", selectedRunUuid=heard, compareToRunUuid=oracle)
            if heard and oracle else None,
            "metrics": metrics,
            "latency_ms": {k: lat.get(k) for k in ("asr", "agent", "tts")},
            "attribution": {r["attribution"]: r["n"] for r in mine},
            "tts_flags": sum(r["tts_flags"] or 0 for r in mine),
        })
    return {"candidates": out}


def dataset_version(run_id: str) -> str:
    """Which input clips a run used. Rows written before dataset versions existed are v1."""
    rows = cached(f"SELECT max(coalesce(dataset_version, 'v1')) AS v FROM {table('pipeline_outputs')} "
                  "WHERE run_id = :run_id", {"run_id": run_id})
    return (rows[0]["v"] if rows else None) or "v1"


@app.get("/api/runs/{run_id}/utterances")
def get_utterances(run_id: str):
    golden = cached(
        f"SELECT utterance_id, category, voice, spoken_text, gt_transcript, audio_uri, duration_s "
        f"FROM {table('golden_utterances')} WHERE dataset_version = :v ORDER BY utterance_id",
        {"v": dataset_version(run_id)})
    attribution = cached(
        f"SELECT candidate, utterance_id, attribution, tts_issue FROM {table('attribution')} WHERE run_id = :run_id",
        {"run_id": run_id})
    for g in golden:
        g["attribution"] = {a["candidate"]: {"label": a["attribution"], "tts_issue": a["tts_issue"]}
                            for a in attribution if a["utterance_id"] == g["utterance_id"]}
    return golden


@app.get("/api/runs/{run_id}/utterances/{utterance_id}")
def get_utterance(run_id: str, utterance_id: str):
    golden = cached(f"SELECT * FROM {table('golden_utterances')} WHERE utterance_id = :u AND dataset_version = :v",
                    {"u": utterance_id, "v": dataset_version(run_id)})
    if not golden:
        raise HTTPException(404, f"Unknown utterance {utterance_id}")
    g = golden[0]
    g["expected_tool_calls"] = json.loads(g["expected_tool_calls"] or "[]")
    turns = cached(
        f"SELECT * FROM {table('pipeline_outputs')} WHERE run_id = :run_id AND utterance_id = :u "
        "ORDER BY candidate, condition DESC", {"run_id": run_id, "u": utterance_id})
    scores = cached(
        f"SELECT trace_id, stage, metric, value, value_text, rationale, source_type, source_id "
        f"FROM {table('eval_results')} WHERE run_id = :run_id AND utterance_id = :u", {"run_id": run_id, "u": utterance_id})
    attribution = cached(
        f"SELECT * FROM {table('attribution')} WHERE run_id = :run_id AND utterance_id = :u", {"run_id": run_id, "u": utterance_id})
    labels = run_labels(run_id)
    for t in turns:
        t["tool_calls"] = json.loads(t["tool_calls"] or "[]")
        t["models"] = json.loads(t["models"]) if t.get("models") else None
        t["scores"] = [s for s in scores if s["trace_id"] == t["trace_id"]]
        t["trace_url"] = trace_url(run_id, t["candidate"], t["condition"], t["trace_id"])
        t["human"] = labels.get(t["trace_id"], {})
    return {"golden": g, "turns": turns, "attribution": {a["candidate"]: a for a in attribution}}


@app.get("/api/audio")
def get_audio(path: str = Query(...)):
    """Stream a clip from the Volume. Only .wav files under this demo's Volume are served."""
    if not path.startswith(VOLUME_ROOT + "/") or ".." in path or not path.endswith(".wav"):
        raise HTTPException(400, "Only .wav files in the demo Volume can be played")
    try:
        data = workspace.files.download(path).contents.read()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, f"Could not read {path}: {e}") from e
    return Response(content=data, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})


# ---------------------------------------------------------------------------------------------
# Listening: every answer a caller would hear (the as-heard TTS clips). Labels are MLflow feedback
# on that clip's trace. Reference-answer clips are for the automatic round trip and aren't rated.
# ---------------------------------------------------------------------------------------------

HUMAN_LABELS = ("human_intelligible", "human_naturalness", "human_pronunciation_issue")
LABELS_TTL_S = 300
_labels: dict[str, tuple[float, dict[str, dict]]] = {}
_labels_lock = threading.Lock()


def _read_labels(run_id: str) -> dict[str, dict]:
    import mlflow

    traces = mlflow.search_traces(
        locations=[EXPERIMENT_ID], filter_string=f"tags.voice_run_id = '{run_id}'",
        include_spans=False, return_type="list", max_results=1000)
    labels: dict[str, dict] = {}
    for t in traces:
        latest: dict = {}
        for a in t.info.assessments or []:
            if a.name in HUMAN_LABELS and a.feedback is not None and a.valid is not False:
                if a.name not in latest or (a.create_time_ms or 0) > latest[a.name]["at"]:
                    latest[a.name] = {"value": a.feedback.value, "by": a.source.source_id if a.source else None,
                                      "at": a.create_time_ms or 0, "id": a.assessment_id}
        labels[t.info.trace_id] = latest
    return labels


def run_labels(run_id: str) -> dict[str, dict]:
    """Human labels by trace ID. Saves update this cache directly, because feedback written to a UC
    trace takes a few seconds to show up in reads."""
    if not EXPERIMENT_ID:
        return {}
    with _labels_lock:
        hit = _labels.get(run_id)
    if hit and time.time() - hit[0] < LABELS_TTL_S:
        return hit[1]
    labels = _read_labels(run_id)
    with _labels_lock:
        _labels[run_id] = (time.time(), labels)
    return labels


def _remember_saved(trace_id: str, saved: dict[str, tuple], reviewer: str):
    """saved: {label name: (value, assessment id)}"""
    with _labels_lock:
        for _, labels in _labels.values():
            if trace_id in labels:
                for name, (value, assessment_id) in saved.items():
                    labels[trace_id][name] = {"value": value, "by": reviewer, "at": int(time.time() * 1000),
                                              "id": assessment_id}


# Labels this app wrote, by (trace ID, label, reviewer), and one lock per key so that quick changes
# to the same rating apply in order.
_own_ids: dict[tuple, str] = {}
_key_locks: dict[tuple, threading.Lock] = {}


def _key_lock(key: tuple) -> threading.Lock:
    with _labels_lock:
        return _key_locks.setdefault(key, threading.Lock())


def _own_label_id(trace_id: str, name: str, reviewer: str) -> str | None:
    """This reviewer's earlier label of this name on the trace, so a changed rating updates it."""
    import mlflow

    with _labels_lock:
        if (trace_id, name, reviewer) in _own_ids:
            return _own_ids[(trace_id, name, reviewer)]
        for _, labels in _labels.values():
            hit = labels.get(trace_id, {}).get(name)
            if hit and hit.get("by") == reviewer and hit.get("id"):
                return hit["id"]
    try:
        trace = mlflow.get_trace(trace_id)
    except Exception:  # noqa: BLE001 - fall back to a new label
        return None
    mine = [a for a in (trace.info.assessments if trace else None) or []
            if a.name == name and a.feedback is not None and a.source and a.source.source_id == reviewer]
    return max(mine, key=lambda a: a.create_time_ms or 0).assessment_id if mine else None


def _update_label(trace_id: str, assessment_id: str, feedback) -> bool:
    """Update a label in place. One written seconds ago may not be visible yet (NOT_FOUND), so retry."""
    import mlflow

    for delay in (0, 1, 2, 3, 4):
        time.sleep(delay)
        try:
            mlflow.update_assessment(trace_id=trace_id, assessment_id=assessment_id, assessment=feedback)
            return True
        except Exception as e:  # noqa: BLE001
            if "NOT_FOUND" not in str(e):
                log.warning("update of %s on %s failed: %s", assessment_id, trace_id, e)
                return False
    log.warning("label %s on %s still not visible after 10 s", assessment_id, trace_id)
    return False


@app.get("/api/runs/{run_id}/listening")
def get_listening(run_id: str):
    """Each candidate's spoken answer to every question, in question order."""
    clips = cached(
        f"""SELECT p.candidate, p.condition, p.utterance_id, p.category, p.trace_id, p.agent_answer, p.reasr_text,
                   p.tts_uri, p.tts_duration_s, p.models, r.metric, r.value
            FROM {table('pipeline_outputs')} p
            LEFT JOIN {table('eval_results')} r ON r.trace_id = p.trace_id AND r.run_id = p.run_id
                 AND r.metric IN ('tts_roundtrip_wer', 'tts_complete', 'tts_speaking_rate')
            WHERE p.run_id = :run_id AND p.condition = 'as_heard' AND p.tts_uri IS NOT NULL""",
        {"run_id": run_id})
    golden = {g["utterance_id"]: g for g in cached(
        f"SELECT utterance_id, gt_transcript, reference_answer FROM {table('golden_utterances')} WHERE dataset_version = :v",
        {"v": dataset_version(run_id)})}
    by_trace: dict[str, dict] = {}
    for c in clips:
        item = by_trace.get(c["trace_id"])
        if item is None:
            g = golden.get(c["utterance_id"], {})
            item = by_trace[c["trace_id"]] = {
                **{k: c[k] for k in ("candidate", "condition", "utterance_id", "category", "trace_id",
                                     "reasr_text", "tts_uri", "tts_duration_s")},
                "question": g.get("gt_transcript"),
                "spoken_text": c["agent_answer"],
                "models": json.loads(c["models"]) if c["models"] else None,
                "trace_url": trace_url(run_id, c["candidate"], c["condition"], c["trace_id"]),
                "auto": {},
            }
        if c["metric"]:
            item["auto"][c["metric"]] = c["value"]
    labels = run_labels(run_id)
    items = sorted(by_trace.values(), key=lambda i: (i["utterance_id"], i["candidate"]))
    for item in items:
        item["human"] = labels.get(item["trace_id"], {})
    return items


class HumanLabel(BaseModel):
    # "tr-<hex>" for classic traces; traces stored in UC may use "trace:/<location>/<hex>".
    trace_id: str = Field(pattern=r"^[\w:/.\-]{3,200}$")
    intelligible: bool | None = None
    naturalness: int | None = Field(default=None, ge=1, le=5)
    pronunciation_issue: str | None = Field(default=None, max_length=500)


@app.post("/api/feedback")
def post_feedback(label: HumanLabel, request: Request):
    """Save listening labels on the trace. A reviewer changing a rating updates their own label in
    place; a different reviewer gets a label of their own."""
    import mlflow
    from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback

    reviewer = request.headers.get("X-Forwarded-Email") or request.headers.get("X-Forwarded-User") or "unknown"
    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=reviewer)
    values = {
        "human_intelligible": label.intelligible,
        "human_naturalness": label.naturalness,
        "human_pronunciation_issue": (label.pronunciation_issue or "").strip() or None,
    }
    rationale = "Listening (Voice Eval Inspector)"
    saved: dict[str, tuple] = {}
    for name, value in values.items():
        if value is None:
            continue
        key = (label.trace_id, name, reviewer)
        with _key_lock(key):
            existing = _own_label_id(*key)
            feedback = Feedback(name=name, value=value, source=source, rationale=rationale)
            if existing and _update_label(label.trace_id, existing, feedback):
                assessment_id = existing
            else:  # first rating, or the update failed: keep the rating as a new label
                assessment_id = mlflow.log_feedback(trace_id=label.trace_id, name=name, value=value,
                                                    source=source, rationale=rationale).assessment_id
            with _labels_lock:
                _own_ids[key] = assessment_id
            saved[name] = (value, assessment_id)
            _remember_saved(label.trace_id, {name: saved[name]}, reviewer)
    if not saved:
        raise HTTPException(400, "Nothing to save")
    log.info("saved %s on %s by %s", list(saved), label.trace_id, reviewer)
    return {"saved": list(saved), "by": reviewer}


@app.get("/api/health")
def health():
    return {"ok": True}


_dist = Path(__file__).resolve().parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
