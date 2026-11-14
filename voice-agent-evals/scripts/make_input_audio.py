"""Make the golden set's input audio with macOS `say`, add light noise, and upload it to the Volume.

Synthetic input flatters ASR, so every row is flagged synthetic_input=true. To use real
recordings instead, drop 16 kHz mono WAVs with the same names into the output folder and skip
straight to --upload.

    python scripts/make_input_audio.py                 # writes build/audio/input/u_*.wav
    python scripts/make_input_audio.py --upload FEVM   # also copies them to the Volume

After generating, listen to every clip once and fix any line the voice garbled. That listen is the
human check on the ground truth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden_v1.jsonl"
OUT = ROOT / "build" / "audio" / "input"
VOLUME_INPUT = "dbfs:/Volumes/pavan_naidu_catalog/mlflow/voice_audio/input"
RATE = 16000
SNR_DB = 20.0


def synthesize(text: str, voice: str, path: Path) -> None:
    cmd = ["say", "-v", voice, "-o", str(path), "--file-format=WAVE", f"--data-format=LEI16@{RATE}", text]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        print(f"  voice {voice!r} not installed, using the system default voice")
        subprocess.run(cmd[:1] + cmd[3:], check=True)


def add_noise(path: Path, seed: int) -> None:
    with wave.open(str(path)) as w:
        params = w.getparams()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    signal_power = np.mean(pcm**2) or 1.0
    noise = np.random.default_rng(seed).normal(0.0, np.sqrt(signal_power / 10 ** (SNR_DB / 10)), pcm.shape)
    noisy = np.clip(pcm + noise, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(noisy.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", metavar="PROFILE", help="copy the clips to the Volume with this CLI profile")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        path = OUT / f"{row['utterance_id']}.wav"
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.wav"
            synthesize(row["spoken_text"], row["voice"], raw)
            path.write_bytes(raw.read_bytes())
        add_noise(path, seed=i)
        with wave.open(str(path)) as w:
            seconds = w.getnframes() / w.getframerate()
        print(f"{row['utterance_id']}  {row['voice']:<9} {seconds:4.1f}s  {row['spoken_text']}")

    if args.upload:
        # Create the Volume if it doesn't exist yet (an "already exists" error is fine).
        subprocess.run(
            ["databricks", "volumes", "create", "pavan_naidu_catalog", "mlflow", "voice_audio", "MANAGED",
             "--profile", args.upload],
            capture_output=True,
        )
        subprocess.run(
            ["databricks", "fs", "cp", "--recursive", "--overwrite", str(OUT), VOLUME_INPUT, "--profile", args.upload],
            check=True,
        )
        print(f"uploaded {len(rows)} clips to {VOLUME_INPUT}")


if __name__ == "__main__":
    main()
