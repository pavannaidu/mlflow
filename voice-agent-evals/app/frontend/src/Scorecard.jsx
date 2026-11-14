import React from "react";
import { api } from "./api.js";
import { Dot, Empty, ErrorNote, MlflowLink, Skeleton, Tip, useLoad } from "./components.jsx";
import { ATTRIBUTION, candidateLabel, candidateStack, inputAudio, pct, points, seconds } from "./format.js";

const minus = (a, b) => (a === null || a === undefined || b === null || b === undefined ? null : a - b);

const STAGES = [
  { n: "01", label: "Hearing" },
  { n: "02", label: "Reasoning" },
  { n: "03", label: "Speaking" },
];

// A metric scored under both conditions gets one header over its two columns.
const PAIRS = {
  tools: { label: "Tools", tip: "Turns where it called the expected tools, given the human-checked transcript and given its own." },
  correct: { label: "Correct", tip: "Answers with the expected facts, given the human-checked transcript and given its own." },
  rt: {
    label: "Round-trip WER",
    tip: "Word error rate between the text its voice spoke and a re-transcription of the audio: the same reference answer for every candidate, and its own answer.",
  },
};

const COLUMNS = [
  { stage: 0, key: "wer", label: "WER", lower: true, fmt: (v) => pct(v, 1),
    tip: "Word error rate of the agent's transcript against the human-checked one.",
    get: (m) => m.asr_wer?.as_heard },
  { stage: 0, key: "terms", label: "Term errors", lower: true, fmt: pct,
    tip: "Share of product names, SKUs and order numbers it misheard.",
    get: (m) => m.asr_term_error_rate?.as_heard },
  { stage: 1, key: "tools_truth", pair: "tools", label: "true transcript", fmt: pct, get: (m) => m.tool_name_match?.oracle },
  { stage: 1, key: "tools_heard", pair: "tools", label: "as heard", fmt: pct, get: (m) => m.tool_name_match?.as_heard },
  { stage: 1, key: "truth", pair: "correct", label: "true transcript", fmt: pct, get: (m) => m.correctness?.oracle },
  { stage: 1, key: "heard", pair: "correct", label: "as heard", fmt: pct, get: (m) => m.correctness?.as_heard },
  { stage: 1, key: "impact", label: "ASR impact", lower: true, fmt: points,
    tip: "Correct with the true transcript minus correct as heard: what mishearing cost.",
    get: (m) => minus(m.correctness?.oracle, m.correctness?.as_heard) },
  { stage: 2, key: "rt_ref", pair: "rt", label: "reference answer", lower: true, fmt: (v) => pct(v, 1),
    get: (m) => m.tts_roundtrip_wer?.oracle },
  { stage: 2, key: "rt_own", pair: "rt", label: "own answer", lower: true, fmt: (v) => pct(v, 1),
    get: (m) => m.tts_roundtrip_wer?.as_heard },
];
const startsStage = (i) => i > 0 && COLUMNS[i - 1].stage !== COLUMNS[i].stage;

function best(rows, col) {
  const vals = rows.map((r) => col.get(r.metrics)).filter((v) => v !== null && v !== undefined);
  if (vals.length < 2 || vals.every((v) => v === vals[0])) return undefined;
  return col.lower ? Math.min(...vals) : Math.max(...vals);
}

function StageTable({ rows }) {
  const bests = Object.fromEntries(COLUMNS.map((c) => [c.key, best(rows, c)]));
  return (
    <div className="table-wrap">
      <table className="score">
        <thead>
          <tr className="stage-row">
            <th />
            {STAGES.map((s, i) => (
              <th key={s.n} colSpan={COLUMNS.filter((c) => c.stage === i).length} className="stage-head">
                <span className="stage-num">{s.n}</span> {s.label}
              </th>
            ))}
          </tr>
          <tr className="col-row">
            <th className="cand-col" rowSpan={2}>
              Candidate
            </th>
            {COLUMNS.map((c, i) => {
              const edge = startsStage(i) ? "group-start" : "";
              if (!c.pair) {
                return (
                  <th key={c.key} rowSpan={2} className={edge}>
                    <Tip text={c.tip}>{c.label}</Tip>
                  </th>
                );
              }
              if (COLUMNS[i - 1]?.pair === c.pair) return null;
              const p = PAIRS[c.pair];
              return (
                <th key={c.pair} colSpan={COLUMNS.filter((d) => d.pair === c.pair).length} className={`pair-head ${edge}`}>
                  <Tip text={p.tip}>{p.label}</Tip>
                </th>
              );
            })}
          </tr>
          <tr className="sub-row">
            {COLUMNS.map((c, i) =>
              c.pair ? (
                <th key={c.key} className={startsStage(i) ? "group-start" : ""}>
                  {c.label}
                </th>
              ) : null,
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.candidate}>
              <th className="cand-col" scope="row">
                <span className="cand-name">{candidateLabel(r.candidate)}</span>
                <span className="cand-stack">{candidateStack(r.candidate, r.models)}</span>
                <MlflowLink
                  href={r.compare_url}
                  className="cand-link"
                  title="Its true-transcript and as-heard evaluation runs, side by side in MLflow"
                >
                  Compare in MLflow
                </MlflowLink>
              </th>
              {COLUMNS.map((c, i) => {
                const v = c.get(r.metrics);
                const isBest = bests[c.key] !== undefined && v === bests[c.key];
                return (
                  <td
                    key={c.key}
                    className={`${isBest ? "is-best" : ""} ${startsStage(i) ? "group-start" : ""}`}
                  >
                    {v === null || v === undefined ? <span className="muted">n/a</span> : c.fmt(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AttributionRow({ row, total }) {
  const counts = row.attribution || {};
  const n = total || Object.values(counts).reduce((a, b) => a + b, 0);
  return (
    <div className="attr-row">
      <div className="attr-name">{candidateLabel(row.candidate)}</div>
      <div className="bar" role="img" aria-label={ATTRIBUTION.map((a) => `${a.label} ${counts[a.key] || 0}`).join(", ")}>
        {ATTRIBUTION.filter((a) => counts[a.key]).map((a) => (
          <span
            key={a.key}
            className="seg"
            style={{ flexGrow: counts[a.key], background: a.color, color: a.ink }}
            title={`${a.label}: ${counts[a.key]} of ${n}`}
          >
            {counts[a.key]}
          </span>
        ))}
      </div>
      <div className="attr-meta">
        <span>{row.tts_flags} TTS {row.tts_flags === 1 ? "flag" : "flags"}</span>
        <span className="muted">
          p50 {row.latency_ms?.asr ? seconds(row.latency_ms.asr) : "n/a"} · {seconds(row.latency_ms?.agent)} · {seconds(row.latency_ms?.tts)}
        </span>
      </div>
    </div>
  );
}

function Loading() {
  return (
    <div className="fade-in">
      <div className="table-wrap pad">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="skeleton-row">
            <Skeleton w="18%" h={18} />
            <Skeleton w="72%" h={18} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Scorecard({ runId, run }) {
  const sc = useLoad(() => api.scorecard(runId), [runId]);
  const rows = sc.data?.candidates || [];
  return (
    <>
      <div className="page-head">
        <h1>Scorecard</h1>
        {run && (
          <p className="page-meta">
            {run.questions} questions · {run.candidates.length} candidates · {run.turns} turns · read by {inputAudio(run.dataset_version)}
          </p>
        )}
      </div>
      {sc.loading && !sc.data && <Loading />}
      {sc.error && <ErrorNote error={sc.error} />}
      {sc.data && !rows.length && <Empty>This run hasn't been scored yet. Run notebook 03_evaluate.</Empty>}
      {rows.length > 0 && (
        <div className="fade-in">
          <StageTable rows={rows} />
          <section className="section">
            <div className="section-head">
              <h2>Where failures came from</h2>
              <div className="legend">
                {ATTRIBUTION.map((a) => (
                  <span key={a.key} className="legend-item">
                    <Dot color={a.color} />
                    {a.label}
                  </span>
                ))}
              </div>
            </div>
            <div className="attr-list">
              <div className="attr-row attr-head" aria-hidden="true">
                <span />
                <span className="muted">Turns as heard, by the stage that failed</span>
                <span className="muted">Latency: hearing · reasoning · speaking</span>
              </div>
              {rows.map((r) => (
                <AttributionRow key={r.candidate} row={r} total={run?.questions} />
              ))}
            </div>
          </section>
        </div>
      )}
    </>
  );
}
