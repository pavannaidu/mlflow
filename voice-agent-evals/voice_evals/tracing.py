"""The text-only trace contract: span helpers and the payload guard.

Audio lives in a UC Volume and is referenced by path. Every span input and output goes through
`guard()`, which fails the run if a payload is too big or looks like audio bytes or base64.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .normalize import NORMALIZER_VERSION

MAX_PAYLOAD_BYTES = 64 * 1024
# Long unbroken base64-alphabet runs are audio (or other binary) smuggled in as text.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{1000,}")


class PayloadError(ValueError):
    """Raised when a span payload breaks the text-only contract."""


def _check(value: Any, path: str) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise PayloadError(f"{path}: binary data ({type(value).__name__}) is not allowed in a span")
    if isinstance(value, str):
        if "data:audio" in value.lower() or ";base64," in value.lower():
            raise PayloadError(f"{path}: data URI found; reference audio by Volume path instead")
        if _BASE64_RUN.search(value):
            raise PayloadError(f"{path}: looks like base64 data; reference audio by Volume path instead")
    elif isinstance(value, dict):
        for k, v in value.items():
            _check(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check(v, f"{path}[{i}]")


def guard(payload: Any, where: str = "payload") -> Any:
    """Return the payload unchanged if it is small text/JSON, otherwise raise PayloadError."""
    size = len(json.dumps(payload, default=str).encode())
    if size > MAX_PAYLOAD_BYTES:
        raise PayloadError(f"{where}: {size} bytes is over the {MAX_PAYLOAD_BYTES}-byte span limit")
    _check(payload, where)
    return payload


def set_inputs(span, payload: Any) -> None:
    span.set_inputs(guard(payload, f"{span.name}.inputs"))


def set_outputs(span, payload: Any) -> None:
    span.set_outputs(guard(payload, f"{span.name}.outputs"))


def trace_tags(
    *,
    run_id: str,
    candidate: str,
    utterance: dict,
    condition: str,
    asr_source: str,
    dataset_version: str,
) -> dict[str, str]:
    """Tags on every trace, so runs can be filtered and split without opening spans.

    The pipeline run is tagged `voice_run_id`, not `run_id`: MLflow's trace filter treats
    `tags.run_id` specially and never matches a plain tag of that name.
    """
    return {
        "voice_run_id": run_id,
        "candidate": candidate,
        "utterance_id": utterance["utterance_id"],
        "category": utterance["category"],
        "condition": condition,
        "asr_source": asr_source,
        "dataset_version": dataset_version,
        "normalizer_version": NORMALIZER_VERSION,
        "synthetic_input": str(utterance.get("synthetic_input", True)).lower(),
        "illustrative_data": "true",
    }
