"""Fakes for the three model calls, so the pipeline, tracing and scorers run locally end to end."""

import io
import json
import re
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

GOLDEN = [json.loads(line) for line in (Path(__file__).parent.parent / "data" / "golden_v1.jsonl").read_text().splitlines()]


def silent_wav(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class FakeASR:
    """Hears whatever was "spoken" into a file (input clips or TTS output)."""

    label = "fake_asr"

    def __init__(self, spoken: dict):
        self.spoken = spoken

    def __call__(self, uri: str) -> str:
        return self.spoken[uri]


class FakeTTS:
    label = "fake_tts"

    def __init__(self, spoken: dict):
        self.spoken = spoken

    def __call__(self, text: str, out_uri: str):
        Path(out_uri).parent.mkdir(parents=True, exist_ok=True)
        seconds = max(0.5, len(text.split()) * 0.38)
        Path(out_uri).write_bytes(silent_wav(seconds))
        self.spoken[out_uri] = text
        return round(seconds, 3), 16000


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if self.content:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in self.tool_calls
            ]
        return d


def _call(i, name, args):
    return SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeLLM:
    """A rules-based stand-in for the chat model: picks tools from keywords, then answers from results."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, tools=None):
        question = messages[1]["content"].lower()
        allowed = {t["function"]["name"] for t in tools or []}
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        step = len(results)
        sku = re.search(r"\b([a-z]{3})[- ]?(\d{4})\b", question)
        if "surgeon" in question or "ibuprofen" in question:
            return self._answer("I'm sorry, I can't help with that. Please ask your care team.")
        if sku and step == 0 and "get_product" in allowed:
            return self._tools(_call(0, "get_product", {"sku": f"{sku.group(1)}-{sku.group(2)}".upper()}))
        if "orbis cup" in question and step == 0 and "search_products" in allowed:
            return self._tools(_call(0, "search_products", {"query": "Orbis cup"}))
        last = results[-1] if results else {}
        if "error" in last:
            return self._answer("I couldn't find that product. Could you repeat the product number?")
        if "list_price_usd" in last:
            return self._answer(f"The {last['name']} lists at ${last['list_price_usd']:,}.")
        if last.get("results"):
            return self._answer("The Orbis cup comes in diameters from 46 to 64 millimeters, in 2 millimeter steps.")
        return self._answer("Sorry, I couldn't find that.")

    @staticmethod
    def _tools(*calls):
        return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(tool_calls=list(calls)))])

    @staticmethod
    def _answer(text):
        return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(content=text))])


@pytest.fixture(scope="session")
def local_run(tmp_path_factory):
    """Run the real pipeline for 4 utterances x 2 candidates x 2 conditions into a local MLflow store."""
    import mlflow

    from voice_evals.audio import audio_ref
    from voice_evals.candidates import CascadeCandidate, MockNoisyCandidate, ToolCallingAgent
    from voice_evals.pipeline import run_pipeline
    from voice_evals.tools import default_tools

    root = tmp_path_factory.mktemp("voice")
    mlflow.set_tracking_uri(f"sqlite:///{root}/mlflow.db")
    experiment_id = mlflow.create_experiment("voice-evals-test", artifact_location=str(root / "artifacts"))
    mlflow.set_experiment(experiment_id=experiment_id)

    spoken: dict = {}
    golden = []
    for utt in [u for u in GOLDEN if u["utterance_id"] in {"u_005", "u_006", "u_008", "u_010"}]:
        path = root / "input" / f"{utt['utterance_id']}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(silent_wav(2.0))
        spoken[str(path)] = utt["gt_transcript"]
        golden.append({**utt, **audio_ref(str(path)), "synthetic_input": True})

    def cascade(name):
        return CascadeCandidate(name, FakeASR(spoken), ToolCallingAgent(FakeLLM(), "fake-llm"), FakeTTS(spoken))

    candidates = [cascade("baseline_cascade"), MockNoisyCandidate("mock_noisy", cascade("mock_base"))]
    rows = run_pipeline(
        candidates, golden, ["oracle", "as_heard"], run_id="r_test", tools=default_tools(),
        reference_asr=FakeASR(spoken), volume_root=str(root), dataset_version="v1", max_workers=2,
        log=lambda *_: None,
    )
    return SimpleNamespace(experiment_id=experiment_id, rows=rows, root=root, golden=golden)
