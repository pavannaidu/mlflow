import React, { useEffect, useState } from "react";
import { ArrowRight, Check, Pencil } from "lucide-react";
import { api } from "./api.js";
import { AudioButton, Empty, ErrorNote, MlflowLink, Skeleton, useLoad } from "./components.jsx";
import { candidateLabel, candidateStack, category, modelName, pct } from "./format.js";

// Each clip is what a caller would hear: the candidate's own answer, spoken by its TTS voice.
// Ratings save to that turn's MLflow trace as human feedback.

function Choice({ label, options, value, onChange }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map(([text, v]) => (
        <button key={String(v)} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>
          {text}
        </button>
      ))}
    </div>
  );
}

const fromHuman = (h = {}) => ({
  intelligible: h.human_intelligible?.value ?? null,
  naturalness: h.human_naturalness?.value ?? null,
  note: h.human_pronunciation_issue?.value ?? "",
});
const isRated = (r) => r.intelligible !== null || r.naturalness !== null;
const PATCH_FIELDS = { intelligible: "intelligible", naturalness: "naturalness", pronunciation_issue: "note" };

// Every control saves on its own, straight to the clip's MLflow trace.
function ClipRow({ item, rating, onSaved }) {
  const [form, setForm] = useState(rating);
  const [savedNote, setSavedNote] = useState(rating.note);
  const [noteOpen, setNoteOpen] = useState(Boolean(rating.note));
  const [status, setStatus] = useState(isRated(rating) || rating.note ? "saved" : "idle");

  async function save(patch) {
    setStatus("saving");
    try {
      await api.saveFeedback({ trace_id: item.trace_id, ...patch });
      setStatus("saved");
      onSaved(item.trace_id, patch);
    } catch (e) {
      setStatus(`error:${e.message}`);
    }
  }

  function pick(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
    save({ [field]: value });
  }

  function saveNote() {
    const note = form.note.trim();
    if (note && note !== savedNote) {
      setSavedNote(note);
      save({ pronunciation_issue: note });
    }
  }

  const auto = item.auto || {};
  const who = `${candidateLabel(item.candidate)} answer`;
  return (
    <div className="clip-row">
      <div className="clip-meta">
        <span className="clip-who">{candidateLabel(item.candidate)}</span>
        <span className="muted">{modelName(item.models?.tts)}</span>
      </div>
      <div className="clip-body">
        <p className="clip-text">{item.spoken_text}</p>
        <p className="clip-auto">
          Round-trip WER {pct(auto.tts_roundtrip_wer, 1) ?? "n/a"}
          {auto.tts_speaking_rate ? ` · ${Math.round(auto.tts_speaking_rate)} wpm` : ""}
          {auto.tts_complete === undefined || auto.tts_complete >= 1 ? "" : " · cut off"}
          <span className="clip-trace">
            <MlflowLink href={item.trace_url} />
          </span>
        </p>
        {noteOpen && (
          <input
            className="clip-note"
            value={form.note}
            autoFocus={!form.note}
            aria-label="Pronunciation note"
            placeholder="Which word or number was said wrong?"
            onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))}
            onBlur={saveNote}
            onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
          />
        )}
      </div>
      <AudioButton path={item.tts_uri} duration={item.tts_duration_s} label={`${who}, ${item.utterance_id}`} />
      <div className="clip-rate">
        <Choice label="Intelligible" options={[["Yes", true], ["No", false]]} value={form.intelligible} onChange={(v) => pick("intelligible", v)} />
        <Choice label="Natural, 1 to 5" options={[1, 2, 3, 4, 5].map((n) => [String(n), n])} value={form.naturalness} onChange={(v) => pick("naturalness", v)} />
        <button
          type="button"
          className={`icon-btn${noteOpen ? " is-on" : ""}`}
          aria-label="Add a pronunciation note"
          aria-pressed={noteOpen}
          onClick={() => setNoteOpen((o) => !o)}
        >
          <Pencil size={14} strokeWidth={2} aria-hidden="true" />
        </button>
        <span className={`save-state${status.startsWith("error") ? " is-error" : ""}`} aria-live="polite">
          {status === "saving" && "Saving…"}
          {status === "saved" && <Check size={16} strokeWidth={2} aria-label="Saved to the trace" />}
          {status.startsWith("error") && "Not saved"}
        </span>
      </div>
    </div>
  );
}

function Summary({ items, ratingOf }) {
  const names = [...new Set(items.map((i) => i.candidate))];
  return (
    <div className="listen-summary">
      {names.map((n) => {
        const mine = items.filter((i) => i.candidate === n);
        const rated = mine.map(ratingOf).filter(isRated);
        const intel = rated.filter((r) => r.intelligible !== null);
        const nat = rated.filter((r) => r.naturalness !== null);
        return (
          <div key={n} className="listen-summary-row">
            <span className="cand-name">{candidateLabel(n)}</span>
            <span className="cand-stack">{candidateStack(n, mine[0]?.models)}</span>
            <span className="listen-summary-stats">
              <b>{rated.length}</b> of {mine.length} rated
              {intel.length > 0 && <> · intelligible <b>{pct(intel.filter((r) => r.intelligible).length / intel.length)}</b></>}
              {nat.length > 0 && <> · natural <b>{(nat.reduce((a, r) => a + r.naturalness, 0) / nat.length).toFixed(1)}</b></>}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function Listening({ runId, focus }) {
  const clips = useLoad(() => api.listening(runId), [runId]);
  const items = clips.data || [];
  const [saved, setSaved] = useState({}); // ratings made on this page, by trace ID
  const [who, setWho] = useState("all");
  const [unrated, setUnrated] = useState(null); // clips unrated when the filter was switched on

  const ratingOf = (item) => ({ ...fromHuman(item.human), ...(saved[item.trace_id] || {}) });

  function onSaved(traceId, patch) {
    const update = Object.fromEntries(Object.entries(patch).map(([k, v]) => [PATCH_FIELDS[k], v]));
    setSaved((s) => ({ ...s, [traceId]: { ...(s[traceId] || {}), ...update } }));
  }

  useEffect(() => {
    if (focus && clips.data) document.getElementById(`q-${focus}`)?.scrollIntoView({ block: "start" });
  }, [focus, clips.data]);

  const names = [...new Set(items.map((i) => i.candidate))];
  const visible = items.filter(
    (i) => (who === "all" || i.candidate === who) && (!unrated || unrated.has(i.trace_id)),
  );
  const groups = [];
  visible.forEach((i) => {
    const last = groups[groups.length - 1];
    if (last && last.id === i.utterance_id) last.items.push(i);
    else groups.push({ id: i.utterance_id, items: [i] });
  });
  const ratedCount = items.filter((i) => isRated(ratingOf(i))).length;

  return (
    <>
      <div className="page-head">
        <h1>Listening</h1>
        {clips.data && (
          <p className="page-meta">
            {items.length} spoken answers · {ratedCount} rated · each rating saves to the turn's MLflow trace
          </p>
        )}
      </div>
      {clips.error && <ErrorNote error={clips.error} />}
      {clips.loading && !clips.data && (
        <div className="clip-list pad">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="skeleton-row">
              <Skeleton w="10%" h={18} />
              <Skeleton w="60%" h={18} />
              <Skeleton w="20%" h={18} />
            </div>
          ))}
        </div>
      )}
      {clips.data && !items.length && <Empty>No spoken answers in this run.</Empty>}
      {items.length > 0 && (
        <div className="fade-in">
          <Summary items={items} ratingOf={ratingOf} />
          <div className="listen-filters">
            <Choice
              label="Candidate"
              options={[["All", "all"], ...names.map((n) => [candidateLabel(n), n])]}
              value={who}
              onChange={setWho}
            />
            <button
              type="button"
              className="toggle-btn"
              aria-pressed={unrated !== null}
              onClick={() =>
                setUnrated(unrated ? null : new Set(items.filter((i) => !isRated(ratingOf(i))).map((i) => i.trace_id)))
              }
            >
              Unrated only
            </button>
          </div>
          {!groups.length && <Empty>Every clip here has a rating.</Empty>}
          {groups.length > 0 && (
            <div className="clip-list">
              <div className="clip-row clip-cols" aria-hidden="true">
                <span>Voice</span>
                <span>Spoken answer</span>
                <span>Audio</span>
                <span className="clip-cols-rate">
                  <span>Intelligible</span>
                  <span>Natural</span>
                </span>
              </div>
              {groups.map((g) => {
                const first = g.items[0];
                return (
                  <section key={g.id} id={`q-${g.id}`} className="clip-group" aria-label={`${g.id}: ${first.question}`}>
                    <header className="clip-q">
                      <span className="mono">{g.id}</span>
                      <span className="muted">{category(first.category)}</span>
                      <span className="clip-q-text">{first.question}</span>
                      <a className="link" href={`#/turns/${g.id}?run=${encodeURIComponent(runId)}`}>
                        Open turn
                        <ArrowRight size={14} strokeWidth={2} aria-hidden="true" />
                      </a>
                    </header>
                    {g.items.map((item) => (
                      <ClipRow key={item.trace_id} item={item} rating={ratingOf(item)} onSaved={onSaved} />
                    ))}
                  </section>
                );
              })}
            </div>
          )}
        </div>
      )}
    </>
  );
}
