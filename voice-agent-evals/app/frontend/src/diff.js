// Word diffs for "truth vs heard" and "intended vs heard back".
// The scorers' rationales already carry both texts after the Python normalizer ran, so the diff
// shown here matches what WER counted. Raw text is the fallback before a run is evaluated.

export function tokens(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9.\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

// Longest-common-subsequence diff: "same", "del" (in the reference only), "ins" (in the hypothesis only).
export function diff(ref, hyp) {
  const n = ref.length;
  const m = hyp.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = ref[i] === hyp[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (ref[i] === hyp[j]) {
      ops.push({ op: "same", text: ref[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ op: "del", text: ref[i++] });
    } else {
      ops.push({ op: "ins", text: hyp[j++] });
    }
  }
  while (i < n) ops.push({ op: "del", text: ref[i++] });
  while (j < m) ops.push({ op: "ins", text: hyp[j++] });
  return ops;
}

// Pull "truth: ..." / "heard: ..." (or "intended:" / "heard back:") out of a scorer rationale.
export function fromRationale(rationale, refLabel, hypLabel) {
  if (!rationale) return null;
  const lines = rationale.split("\n");
  const pick = (label) => {
    const line = lines.find((l) => l.startsWith(`${label}:`));
    return line === undefined ? null : line.slice(label.length + 1).trim();
  };
  const ref = pick(refLabel);
  const hyp = pick(hypLabel);
  return ref === null || hyp === null ? null : [ref, hyp];
}
