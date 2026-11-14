import React, { useState } from "react";
import { Check, ChevronDown, X } from "lucide-react";
import { api } from "./api.js";
import { AttrBadge, AudioButton, Diff, Dot, Empty, ErrorNote, MlflowLink, Skeleton, useLoad } from "./components.jsx";
import { fromRationale } from "./diff.js";
import {
  ATTRIBUTION,
  METRICS,
  attribution,
  candidateLabel,
  candidateStack,
  category,
  humanSummary,
  metricLabel,
  passes,
  scoreText,
} from "./format.js";

const callText = (c) =>
  `${c.name}(${Object.entries(c.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})`;
// Calls compare like the scorers compare them: "TLF 2210" and "TLF-2210" are the same SKU.
const callsKey = (calls) =>
  (calls || [])
    .map((c) => `${c.name}(${Object.values(c.args || {}).map((v) => String(v).toLowerCase().replace(/[^a-z0-9]/g, ""))})`)
    .join(" ");

function QuestionList({ items, selected, onSelect, names }) {
  return (
    <nav className="qlist" aria-label="Questions">
      <div className="qlist-legend">
        {ATTRIBUTION.map((a) => (
          <span key={a.key} className="legend-item">
            <Dot color={a.color} />
            {a.label}
          </span>
        ))}
      </div>
      {items.map((u) => (
        <button
          key={u.utterance_id}
          type="button"
          className="qitem"
          aria-current={u.utterance_id === selected}
          onClick={() => onSelect(u.utterance_id)}
        >
          <span className="qitem-top">
            <span className="mono">{u.utterance_id}</span>
            <span>{category(u.category)}</span>
            <span className="qitem-dots">
              {names.map((n) => {
                const a = attribution(u.attribution?.[n]?.label);
                return <span key={n} className="dot" style={{ background: a.color }} title={`${candidateLabel(n)}: ${a.label}`} />;
              })}
            </span>
          </span>
          <span className="qitem-text">{u.gt_transcript}</span>
        </button>
      ))}
    </nav>
  );
}

function Checks({ turn }) {
  const [open, setOpen] = useState(null);
  const scores = Object.keys(METRICS)
    .map((m) => turn.scores.find((s) => s.metric === m))
    .filter(Boolean);
  if (!scores.length) return <span className="muted">Not scored yet</span>;
  const failing = scores.filter((s) => passes(s.metric, s.value) === false);
  const shown = open === "all" ? scores : failing;
  const active = scores.find((s) => s.metric === open);
  return (
    <div className="checks">
      {failing.length === 0 && open !== "all" && (
        <span className="ok-line">
          <Check size={16} strokeWidth={2} aria-hidden="true" />
          All {scores.length} checks pass
        </span>
      )}
      <div className="chips">
        {shown.map((s) => {
          const ok = passes(s.metric, s.value);
          return (
            <button
              key={s.metric}
              type="button"
              className={`chip ${ok === false ? "is-bad" : ok ? "is-good" : ""}`}
              aria-pressed={open === s.metric}
              onClick={() => setOpen(open === s.metric ? null : s.metric)}
            >
              {metricLabel(s.metric)} <b>{scoreText(s.metric, s.value)}</b>
            </button>
          );
        })}
      </div>
      {active && <pre className="rationale">{active.rationale || "No rationale recorded."}</pre>}
      <button type="button" className="text-btn" onClick={() => setOpen(open === "all" ? null : "all")}>
        {open === "all" ? "Show failing only" : `Show all ${scores.length}`}
        <ChevronDown size={14} className={open === "all" ? "flip" : ""} aria-hidden="true" />
      </button>
    </div>
  );
}

function Calls({ calls, expected }) {
  if (!calls?.length) return <span className="muted">No tool call</span>;
  return (
    <div className="calls">
      {calls.map((c, i) => (
        <code key={i} className={`call${expected.has(c.name) ? "" : " is-off"}`}>
          {callText(c)}
        </code>
      ))}
    </div>
  );
}

// The same step run on the human-checked transcript, shown under the as-heard result.
function TrueTranscript({ ok, title, children }) {
  return (
    <p className="oracle" title={title}>
      {ok === true && <Check size={14} strokeWidth={2} aria-hidden="true" />}
      {ok === false && <X size={14} strokeWidth={2} aria-hidden="true" />}
      With the true transcript: {children}
    </p>
  );
}

// One spoken clip: audio and round-trip diff. Own answers also show their listening rating.
function Spoken({ turn, label, who, intended, rateHref }) {
  const rt = turn?.scores.find((s) => s.metric === "tts_roundtrip_wer");
  const pair = fromRationale(rt?.rationale, "intended", "heard back") || [intended, turn?.reasr_text];
  const rated = humanSummary(turn?.human);
  return (
    <div className="spoken">
      <span className="spoken-label">{label}</span>
      <AudioButton path={turn?.tts_uri} duration={turn?.tts_duration_s} label={`${who} ${label.toLowerCase()}`} />
      <Diff pair={pair} same="Heard back intact" />
      {turn?.tts_uri && rateHref && (
        <p className="spoken-human">
          {rated ? <>Rated: {rated}</> : <a className="link" href={rateHref}>Rate in Listening</a>}
        </p>
      )}
    </div>
  );
}

function CandidateColumn({ name, turns, golden, attr, runId }) {
  const heard = turns.as_heard;
  const oracle = turns.oracle;
  const expected = new Set(golden.expected_tool_calls.map((c) => c.name));
  const score = (t, m) => t?.scores.find((s) => s.metric === m)?.value;
  const pass = (t, metrics) => {
    const vals = metrics.map((m) => score(t, m)).filter((v) => v !== undefined && v !== null);
    return vals.length ? vals.every((v) => v >= 1) : undefined;
  };
  const asrPair = fromRationale(heard?.scores.find((s) => s.metric === "asr_wer")?.rationale, "truth", "heard") || [
    golden.gt_transcript,
    heard?.asr_text,
  ];
  const sameCalls = heard && oracle && callsKey(heard.tool_calls) === callsKey(oracle.tool_calls);
  const oracleCorrect = pass(oracle, ["correctness"]);
  const rateHref = `#/listening/${golden.utterance_id}?run=${encodeURIComponent(runId)}`;

  return (
    <article className="ccol">
      <header className="ccol-head">
        <div className="ccol-title">
          <h3>{candidateLabel(name)}</h3>
          <div className="ccol-badges">
            {attr?.tts_issue && <span className="badge badge-quiet">TTS flag</span>}
            {attr && <AttrBadge label={attr.attribution} />}
          </div>
        </div>
        <p className="cand-stack">{candidateStack(name, heard?.models || oracle?.models)}</p>
      </header>

      <section className="cell">
        <h4 className="cell-label">Heard</h4>
        {heard ? <Diff pair={asrPair} same="Heard correctly" /> : <span className="muted">n/a</span>}
      </section>

      <section className="cell">
        <h4 className="cell-label">Tools</h4>
        {heard ? <Calls calls={heard.tool_calls} expected={expected} /> : <span className="muted">n/a</span>}
        {oracle && (
          <TrueTranscript ok={pass(oracle, ["tool_name_match", "tool_args_match"])}>
            {sameCalls ? "same calls" : heard ? "different calls" : ""}
          </TrueTranscript>
        )}
        {oracle && !sameCalls && <Calls calls={oracle.tool_calls} expected={expected} />}
      </section>

      <section className="cell">
        <h4 className="cell-label">Answer</h4>
        <p className="answer">{heard?.agent_answer || <span className="muted">No answer</span>}</p>
        {oracle && (
          <TrueTranscript ok={oracleCorrect} title={oracle.agent_answer}>
            {oracleCorrect === undefined ? "not scored" : oracleCorrect ? "correct" : "not correct"}
          </TrueTranscript>
        )}
        {heard?.error && <p className="error-inline">{heard.error}</p>}
      </section>

      <section className="cell">
        <h4 className="cell-label">Spoken</h4>
        <Spoken turn={heard} label="Own answer" who={candidateLabel(name)} intended={heard?.agent_answer} rateHref={rateHref} />
        {oracle?.tts_uri && (
          <Spoken turn={oracle} label="Reference answer" who={candidateLabel(name)} intended={golden.reference_answer} />
        )}
      </section>

      <section className="cell">
        <h4 className="cell-label">Checks</h4>
        {heard ? <Checks turn={heard} /> : <span className="muted">n/a</span>}
      </section>

      <footer className="cell ccol-links">
        <MlflowLink href={heard?.trace_url}>As-heard trace</MlflowLink>
        <MlflowLink href={oracle?.trace_url}>True-transcript trace</MlflowLink>
      </footer>
    </article>
  );
}

function Detail({ runId, id }) {
  const d = useLoad(() => api.utterance(runId, id), [runId, id]);
  if (d.error) return <ErrorNote error={d.error} />;
  if (!d.data) {
    return (
      <div className="detail-loading">
        <Skeleton w="30%" h={14} />
        <Skeleton w="70%" h={28} />
        <Skeleton w="100%" h={320} />
      </div>
    );
  }
  const { golden: g, turns, attribution: attr } = d.data;
  const byCandidate = {};
  turns.forEach((t) => {
    byCandidate[t.candidate] = { ...(byCandidate[t.candidate] || {}), [t.condition]: t };
  });
  const names = Object.keys(byCandidate).sort();
  return (
    <div className={`fade-in${d.loading ? " is-stale" : ""}`} key={g.utterance_id}>
      <article className="question">
        <p className="eyebrow">
          {g.utterance_id} · {category(g.category)} · {g.voice}
          {g.synthetic_input ? " (synthetic voice)" : ""}
        </p>
        <h2 className="question-text">{g.gt_transcript}</h2>
        <div className="question-row">
          <AudioButton path={g.audio_uri} duration={g.duration_s} label="question audio" />
          {g.spoken_text !== g.gt_transcript && <span className="muted">Spoken as “{g.spoken_text}”</span>}
        </div>
        <div className="expects">
          <span className="expects-label">Expects</span>
          {g.expected_tool_calls.length ? (
            g.expected_tool_calls.map((c, i) => (
              <code key={i} className="call">
                {callText(c)}
              </code>
            ))
          ) : (
            <span className="muted">No tool call</span>
          )}
          <details className="facts">
            <summary>
              {g.expected_facts.length} expected {g.expected_facts.length === 1 ? "fact" : "facts"}
            </summary>
            <ul>
              {g.expected_facts.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </details>
          {g.reference_answer && (
            <details className="facts">
              <summary>Reference answer</summary>
              <p>{g.reference_answer}</p>
            </details>
          )}
        </div>
      </article>
      <div className="compare" style={{ "--cols": names.length }}>
        {names.map((n) => (
          <CandidateColumn key={n} name={n} turns={byCandidate[n]} golden={g} attr={attr[n]} runId={runId} />
        ))}
      </div>
    </div>
  );
}

export default function Turns({ runId, selected, onSelect }) {
  const list = useLoad(() => api.utterances(runId), [runId]);
  if (list.error) return <ErrorNote error={list.error} />;
  const items = list.data || [];
  const names = [...new Set(items.flatMap((u) => Object.keys(u.attribution || {})))].sort();
  const current = selected || items[0]?.utterance_id;
  return (
    <div className="turns">
      {list.loading && !list.data ? (
        <div className="qlist">
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} h={56} />
          ))}
        </div>
      ) : (
        <QuestionList items={items} selected={current} onSelect={onSelect} names={names} />
      )}
      <div className="turn-detail">
        {current ? <Detail runId={runId} id={current} /> : list.data && <Empty>No questions in this run.</Empty>}
      </div>
    </div>
  );
}
