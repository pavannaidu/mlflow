// Display names, metric metadata and number formats shared by every view.

const LABELS = { baseline_cascade: "Baseline", gemini_cascade: "Gemini", gemini_lite_cascade: "Gemini Lite", mock_noisy: "Mock" };
export const candidateLabel = (name) => LABELS[name] ?? name;

const MODEL_NAMES = {
  "databricks-gpt-5-4-mini": "GPT-5.4 mini",
  "databricks-meta-llama-3-1-8b-instruct": "Llama 3.1 8B",
  "databricks-meta-llama-3-3-70b-instruct": "Llama 3.3 70B",
  "databricks-gemma-3-12b": "Gemma 3 12B",
  "databricks-gemini-3-1-flash-lite": "Gemini 3.1 Flash Lite",
  "databricks-gemini-3-5-flash": "Gemini 3.5 Flash",
  "databricks-gemini-3-8-flash": "Gemini 3.8 Flash",
};
export function modelName(id) {
  if (!id) return "n/a";
  if (id.startsWith("piper:")) return `Piper ${id.split("-")[1]}`; // piper:en_US-lessac-medium
  if (id.startsWith("mock")) return "Planted errors";
  return MODEL_NAMES[id] ?? id;
}

// Runs record their models in pipeline_outputs; runs from before that used these.
const EARLIER_MODELS = {
  baseline_cascade: { asr: "ai_transcribe", llm: "databricks-gpt-5-4-mini", tts: "piper:en_US-lessac-medium" },
  gemini_cascade: { asr: "databricks-gemini-3-5-flash", llm: "databricks-gemini-3-5-flash", tts: "piper:en_GB-alan-medium" },
  mock_noisy: { asr: "mock", llm: "databricks-gpt-5-4-mini", tts: "piper:en_US-lessac-medium" },
};
export function candidateStack(name, models) {
  const m = models || EARLIER_MODELS[name];
  // Non-breaking spaces inside each name, so a narrow column wraps only between models.
  return m ? [m.asr, m.llm, m.tts].map((id) => modelName(id).replace(/ /g, "\u00a0")).join(" · ") : "";
}

export const ATTRIBUTION = [
  { key: "pass", label: "Pass", color: "var(--attr-pass)", ink: "var(--db-navy-900)" },
  { key: "ASR-caused", label: "ASR-caused", color: "var(--attr-asr)", ink: "var(--db-white)" },
  { key: "reasoning", label: "Reasoning", color: "var(--attr-reasoning)", ink: "var(--db-navy-900)" },
  { key: "flaky", label: "Flaky", color: "var(--attr-flaky)", ink: "var(--db-navy-900)" },
];
const ATTR = Object.fromEntries(ATTRIBUTION.map((a) => [a.key, a]));
ATTR.incomplete = { key: "incomplete", label: "Incomplete", color: "var(--attr-flaky)", ink: "var(--db-navy-900)" };
export const attribution = (key) => ATTR[key] ?? { key, label: key, color: "var(--attr-flaky)", ink: "var(--db-navy-900)" };

// kind: "bool" (pass when 1), "rate" (pass when 1), "error" (pass when 0), "wpm" (pass inside 120-190)
export const METRICS = {
  asr_wer: { label: "WER", stage: 1, kind: "error" },
  asr_term_error_rate: { label: "Term errors", stage: 1, kind: "error" },
  asr_number_accuracy: { label: "Numbers heard", stage: 1, kind: "rate" },
  asr_exact_match: { label: "Exact match", stage: 1, kind: "bool" },
  tool_name_match: { label: "Tools", stage: 2, kind: "bool" },
  tool_args_match: { label: "Tool arguments", stage: 2, kind: "rate" },
  correctness: { label: "Correct", stage: 2, kind: "bool" },
  voice_guidelines: { label: "Guidelines", stage: 2, kind: "bool" },
  answer_exact_numeric: { label: "Numbers in answer", stage: 2, kind: "bool" },
  spoken_suitability: { label: "Speakable", stage: 2, kind: "bool" },
  tts_roundtrip_wer: { label: "Round-trip WER", stage: 3, kind: "error" },
  tts_term_accuracy: { label: "Terms spoken", stage: 3, kind: "rate" },
  tts_number_accuracy: { label: "Numbers spoken", stage: 3, kind: "rate" },
  tts_speaking_rate: { label: "Speaking rate", stage: 3, kind: "wpm" },
  tts_complete: { label: "Complete", stage: 3, kind: "bool" },
};
export const metricLabel = (m) => METRICS[m]?.label ?? m;

export function passes(metric, value) {
  if (value === null || value === undefined) return null;
  const kind = METRICS[metric]?.kind;
  if (kind === "error") return value <= 0.0001;
  if (kind === "wpm") return value >= 120 && value <= 190;
  return value >= 1;
}

export function scoreText(metric, value) {
  if (value === null || value === undefined) return "n/a";
  const kind = METRICS[metric]?.kind;
  if (kind === "bool") return value >= 1 ? "Yes" : "No";
  if (kind === "wpm") return `${Math.round(value)} wpm`;
  return pct(value, kind === "error" ? 1 : 0);
}

export const pct = (v, digits = 0) => (v === null || v === undefined ? null : `${(v * 100).toFixed(digits)}%`);

export function points(v) {
  if (v === null || v === undefined) return null;
  const p = Math.round(v * 100);
  return `${p > 0 ? "+" : p < 0 ? "−" : ""}${Math.abs(p)} pts`;
}

export const seconds = (ms) => (ms === null || ms === undefined ? "n/a" : `${(ms / 1000).toFixed(1)} s`);

export function clock(s) {
  const t = Math.max(0, Math.round(s || 0));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
}

// Run IDs are UTC timestamps: r20261111_163143 -> "Nov 11, 16:31 UTC".
export function runLabel(runId) {
  const m = /^r(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(runId || "");
  if (!m) return runId;
  const month = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])).toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  return `${month} ${+m[3]}, ${m[4]}:${m[5]} UTC`;
}

export const category = (c) => (c || "").replace(/_/g, " ");

// Which voices read the questions in each dataset version.
const INPUT_AUDIO = { v1: "macOS voices", v2: "Kokoro voices", v3: "Kokoro voices in noise", v4: "Kokoro voices, natural speech" };
export const inputAudio = (version) => INPUT_AUDIO[version] ?? version;

// Human listening labels on a trace, as one short line: "intelligible · natural 4/5 · “note”".
export function humanSummary(h) {
  if (!h) return null;
  const parts = [];
  if (h.human_intelligible) parts.push(h.human_intelligible.value ? "intelligible" : "hard to follow");
  if (h.human_naturalness) parts.push(`natural ${h.human_naturalness.value}/5`);
  if (h.human_pronunciation_issue?.value) parts.push(`“${h.human_pronunciation_issue.value}”`);
  return parts.length ? parts.join(" · ") : null;
}
