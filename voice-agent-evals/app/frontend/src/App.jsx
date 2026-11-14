import React, { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import { api } from "./api.js";
import { Empty, ErrorNote, Skeleton, useLoad } from "./components.jsx";
import { inputAudio, runLabel } from "./format.js";
import Listening from "./Listening.jsx";
import Scorecard from "./Scorecard.jsx";
import Turns from "./Turns.jsx";

const TABS = [
  { key: "scorecard", label: "Scorecard" },
  { key: "turns", label: "Turns" },
  { key: "listening", label: "Listening" },
];

// Routes live in the hash, so every view (and every question) has a link: #/turns/u_006?run=...
function readRoute() {
  const [path, qs] = window.location.hash.replace(/^#\/?/, "").split("?");
  const [tab, id] = (path || "").split("/");
  return {
    tab: TABS.some((t) => t.key === tab) ? tab : "scorecard",
    id: id || null,
    run: new URLSearchParams(qs || "").get("run"),
  };
}

const href = ({ tab, id, run }) => `#/${tab}${id ? `/${id}` : ""}${run ? `?run=${run}` : ""}`;

export default function App() {
  const runs = useLoad(api.runs, []);
  const [route, setRoute] = useState(readRoute);

  useEffect(() => {
    const onHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const list = runs.data || [];
  const run = list.find((r) => r.run_id === route.run) || list.find((r) => r.evaluated) || list[0];
  const go = (next) => {
    window.location.hash = href({ ...route, run: run?.run_id, ...next });
  };

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href={href({ tab: "scorecard", run: route.run })}>
          <img src="/brand/databricks-symbol.svg" alt="" width="18" height="20" />
          <span>Voice agent evals</span>
        </a>
        <nav className="tabs" aria-label="Views">
          {TABS.map((t) => (
            <a
              key={t.key}
              className="tab"
              href={href({ tab: t.key, run: route.run })}
              aria-current={route.tab === t.key ? "page" : undefined}
            >
              {t.label}
            </a>
          ))}
        </nav>
        <div className="topbar-end">
          <span
            className="tag"
            tabIndex={0}
            data-tip="Made-up products and orders. The input audio is synthetic, which flatters ASR."
          >
            Illustrative data
          </span>
          {list.length > 0 && (
            <label className="run-select">
              <span className="sr-only">Run</span>
              <select value={run?.run_id} onChange={(e) => go({ run: e.target.value, id: null })} title={run?.run_id}>
                {list.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {runLabel(r.run_id)} · {r.candidates.length} candidates · {inputAudio(r.dataset_version)}
                    {r.evaluated ? "" : " · not scored"}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} strokeWidth={2} aria-hidden="true" />
            </label>
          )}
        </div>
      </header>

      <main className="page">
        {runs.loading && !runs.data && (
          <div className="page-head">
            <Skeleton w="180px" h={28} />
          </div>
        )}
        {runs.error && <ErrorNote error={runs.error} />}
        {runs.data && !list.length && <Empty>No runs yet. Run notebooks 02_run_pipeline and 03_evaluate.</Empty>}
        {run && route.tab === "scorecard" && <Scorecard runId={run.run_id} run={run} />}
        {run && route.tab === "turns" && (
          <Turns runId={run.run_id} selected={route.id} onSelect={(id) => go({ tab: "turns", id })} />
        )}
        {run && route.tab === "listening" && <Listening key={run.run_id} runId={run.run_id} focus={route.id} />}
      </main>
    </div>
  );
}
