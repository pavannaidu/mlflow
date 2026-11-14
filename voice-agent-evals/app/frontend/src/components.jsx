import React, { useEffect, useRef, useState } from "react";
import { ArrowUpRight, Check, Info, Pause, Play } from "lucide-react";
import { audioUrl } from "./api.js";
import { diff, tokens } from "./diff.js";
import { attribution, clock } from "./format.js";

export function useLoad(load, deps) {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    load()
      .then((data) => live && setState({ loading: false, data, error: null }))
      .catch((e) => live && setState({ loading: false, data: null, error: e.message }));
    return () => {
      live = false;
    };
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}

export const Skeleton = ({ w = "100%", h = 16 }) => (
  <span className="skeleton" style={{ width: w, height: h }} aria-hidden="true" />
);

export function ErrorNote({ error }) {
  return (
    <div className="error-note" role="alert">
      Couldn't load this view. {error}
    </div>
  );
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}

export function Tip({ text, children }) {
  return (
    <span className="tip" tabIndex={0} data-tip={text}>
      {children}
      <Info size={12} strokeWidth={2} className="tip-icon" aria-hidden="true" />
    </span>
  );
}

// The backend builds MLflow URLs: a trace opens inside its evaluation run.
export function MlflowLink({ href, children = "Trace", className = "", title }) {
  if (!href) return null;
  return (
    <a className={`link ${className}`} href={href} target="_blank" rel="noreferrer" title={title}>
      {children}
      <ArrowUpRight size={14} strokeWidth={2} aria-hidden="true" />
    </a>
  );
}

export function Dot({ color }) {
  return <span className="dot" style={{ background: color }} aria-hidden="true" />;
}

export function AttrBadge({ label }) {
  const a = attribution(label);
  return (
    <span className="badge">
      <Dot color={a.color} />
      {a.label}
    </span>
  );
}

// Word diff. Words the speaker said but the transcript lacks are struck through; words the
// transcript has instead are highlighted.
export function Diff({ pair, same = "Matches" }) {
  if (!pair || pair[1] === null || pair[1] === undefined) return <span className="muted">n/a</span>;
  const ops = diff(tokens(pair[0]), tokens(pair[1]));
  if (ops.every((o) => o.op === "same")) {
    return (
      <span className="ok-line">
        <Check size={16} strokeWidth={2} aria-hidden="true" />
        {same}
      </span>
    );
  }
  return (
    <p className="diff">
      {ops.map((o, i) => (
        <React.Fragment key={i}>
          {i > 0 && " "}
          <span className={`tok tok-${o.op}`}>{o.text}</span>
        </React.Fragment>
      ))}
    </p>
  );
}

let nowPlaying = null; // only one clip plays at a time

export function AudioButton({ path, duration, label }) {
  const audio = useRef(null);
  const [st, setSt] = useState({ on: false, t: 0, d: duration || 0, err: false });
  useEffect(() => () => audio.current?.pause(), []);
  if (!path) return null;

  function ensure() {
    if (audio.current) return audio.current;
    const a = new Audio(audioUrl(path));
    a.preload = "auto";
    a.addEventListener("loadedmetadata", () => setSt((s) => ({ ...s, d: a.duration || s.d })));
    a.addEventListener("timeupdate", () => setSt((s) => ({ ...s, t: a.currentTime })));
    a.addEventListener("play", () => setSt((s) => ({ ...s, on: true })));
    a.addEventListener("pause", () => setSt((s) => ({ ...s, on: false })));
    a.addEventListener("ended", () => setSt((s) => ({ ...s, on: false, t: 0 })));
    a.addEventListener("error", () => setSt((s) => ({ ...s, on: false, err: true })));
    audio.current = a;
    return a;
  }

  function toggle() {
    const a = ensure();
    if (a.paused) {
      if (nowPlaying && nowPlaying !== a) nowPlaying.pause();
      nowPlaying = a;
      a.play().catch(() => setSt((s) => ({ ...s, err: true })));
    } else {
      a.pause();
    }
  }

  function seek(e) {
    const a = ensure();
    const r = e.currentTarget.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    if (a.duration) a.currentTime = f * a.duration;
  }

  const frac = st.d ? Math.min(1, st.t / st.d) : 0;
  return (
    <div className={`audio${st.on ? " is-on" : ""}`}>
      <button type="button" className="audio-btn" onClick={toggle} aria-label={`${st.on ? "Pause" : "Play"} ${label}`}>
        {st.on ? <Pause size={14} strokeWidth={2.25} /> : <Play size={14} strokeWidth={2.25} />}
      </button>
      <div className="audio-track" onClick={seek} title={path}>
        <span className="audio-fill" style={{ transform: `scaleX(${frac})` }} />
      </div>
      <span className="audio-time">{st.err ? "Unavailable" : clock(st.on || st.t ? st.t : st.d)}</span>
    </div>
  );
}
