import base64

import pytest

from voice_evals.tracing import MAX_PAYLOAD_BYTES, PayloadError, guard


def test_text_and_small_json_pass():
    payload = {"audio_uri": "/Volumes/c/s/v/input/u_001.wav", "duration_s": 3.2, "text": "hello"}
    assert guard(payload) is payload


def test_bytes_are_rejected():
    with pytest.raises(PayloadError, match="binary"):
        guard({"audio": b"RIFF\x00\x00WAVE"})


def test_audio_data_uri_is_rejected_even_when_nested():
    with pytest.raises(PayloadError, match="data URI"):
        guard({"messages": [{"content": [{"audio_url": {"url": "data:audio/wav;base64,UklGRg=="}}]}]})


def test_base64_blob_is_rejected():
    blob = base64.b64encode(bytes(range(256)) * 8).decode()
    with pytest.raises(PayloadError, match="base64"):
        guard({"text": blob})


def test_oversized_payload_is_rejected():
    with pytest.raises(PayloadError, match="limit"):
        guard({"text": "word " * (MAX_PAYLOAD_BYTES // 4)})
