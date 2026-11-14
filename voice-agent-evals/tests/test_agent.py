"""The agent loop keeps provider extras (Gemini thought signatures) for the model, not the trace."""

import json
from types import SimpleNamespace

import mlflow

from voice_evals.candidates import ToolCallingAgent
from voice_evals.tools import default_tools
from voice_evals.tracing import guard

SIGNATURE = "Q" * 2000  # an opaque base64-looking blob, like Gemini returns


class _Msg:
    def __init__(self, content=None, tool_calls=None, dump=None):
        self.content, self.tool_calls, self._dump = content, tool_calls, dump

    def model_dump(self, exclude_none=True):
        return self._dump


class GeminiLikeClient:
    """First call: a tool call carrying a thought signature. Second call: the answer."""

    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, tools=None):
        self.requests.append(json.loads(json.dumps(messages)))
        if len(self.requests) == 1:
            call = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_product", arguments='{"sku": "TLF-2210"}'))
            dump = {"role": "assistant", "annotations": [], "tool_calls": [{
                "id": "c1", "type": "function", "function": {"name": "get_product", "arguments": '{"sku": "TLF-2210"}'},
                "extra_content": {"google": {"thought_signature": SIGNATURE}},
            }]}
            return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(tool_calls=[call], dump=dump))])
        dump = {"role": "assistant", "content": [{"type": "text", "text": "It lists at $1,840."}]}
        return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(content=dump["content"], dump=dump))])


def test_signature_goes_back_to_the_model_but_not_into_the_trace(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("agent-test")
    client = GeminiLikeClient()

    result = ToolCallingAgent(client, "gemini-like")("What's the list price of TLF-2210?", default_tools())

    assert result.answer == "It lists at $1,840."  # list-of-parts content flattened to text
    resent = client.requests[1][2]
    assert resent["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == SIGNATURE
    assert "annotations" not in resent

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    second_llm = [s for s in trace.data.spans if s.name == "llm"][1]
    logged = second_llm.inputs["messages"][2]["tool_calls"][0]
    assert logged["extra_content"] == "[provider signature omitted]"
    for span in trace.data.spans:
        guard(span.inputs)
        guard(span.outputs)
