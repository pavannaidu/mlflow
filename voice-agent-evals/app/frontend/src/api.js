async function request(path, options) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    throw new Error(`${resp.status}: ${await resp.text()}`);
  }
  return resp.json();
}

const run = (id) => `/api/runs/${encodeURIComponent(id)}`;

export const api = {
  config: () => request("/api/config"),
  runs: () => request("/api/runs"),
  scorecard: (runId) => request(`${run(runId)}/scorecard`),
  utterances: (runId) => request(`${run(runId)}/utterances`),
  utterance: (runId, id) => request(`${run(runId)}/utterances/${encodeURIComponent(id)}`),
  listening: (runId) => request(`${run(runId)}/listening`),
  saveFeedback: (body) =>
    request("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export const audioUrl = (path) => `/api/audio?path=${encodeURIComponent(path)}`;
