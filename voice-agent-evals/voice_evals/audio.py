"""WAV helpers. Audio stays in the Volume; only these small reference dicts reach a trace."""

from __future__ import annotations

import hashlib
import io
import os
import wave


def audio_ref(path: str) -> dict:
    """The text-only reference a span carries instead of audio bytes."""
    with open(path, "rb") as f:
        data = f.read()
    with wave.open(io.BytesIO(data)) as w:
        rate, frames = w.getframerate(), w.getnframes()
    return {
        "audio_uri": path,
        "audio_sha256": hashlib.sha256(data).hexdigest(),
        "duration_s": round(frames / rate, 3),
        "sample_rate_hz": rate,
    }


def write_bytes(path: str, data: bytes) -> None:
    """Write a whole file in one sequential pass (UC Volumes don't support random writes)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def download_file(path: str, url: str) -> str:
    """Stage a model file in the Volume once, so later runs don't download it again."""
    import requests

    if not os.path.exists(path):
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        write_bytes(path, resp.content)
    return path


def download_piper_voice(model_path: str, url: str) -> str:
    """Stage a Piper voice (.onnx plus .onnx.json) in the Volume."""
    for suffix in ("", ".json"):
        download_file(model_path + suffix, url + suffix)
    return model_path


def add_noise(samples, snr_db: float, seed: int):
    """Light Gaussian background noise at a fixed signal-to-noise ratio, reproducible per seed."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    power = float(np.mean(x**2)) or 1e-9
    noise = np.random.default_rng(seed).normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), x.shape)
    return x + noise


def wav_bytes(samples, rate: int) -> bytes:
    """Float samples in [-1, 1] as 16-bit mono WAV bytes."""
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
