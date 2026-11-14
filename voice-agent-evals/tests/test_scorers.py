"""End to end on a local MLflow store: pipeline -> text-only traces -> offline evaluate -> scorecard."""

import json
import os

import mlflow
import pandas as pd
import pytest

from voice_evals.evaluate import asr_impact, attribution, collect_results, load_run_traces, run_evaluations
from voice_evals.tracing import guard


@pytest.fixture(scope="module")
def results(local_run):
    traces = load_run_traces(local_run.experiment_id, "r_test")
    run_evaluations(traces, run_id="r_test", judge_model="unused", include_judges=False, log=lambda *_: None)
    return collect_results(local_run.experiment_id, "r_test")


def value(results, candidate, utt, metric, condition="as_heard"):
    r = results[(results.candidate == candidate) & (results.utterance_id == utt)
                & (results.metric == metric) & (results.condition == condition)]
    assert len(r) == 1, f"no {metric} for {candidate}/{utt}/{condition}"
    return r.iloc[0]["value"]


def test_every_turn_logged_one_text_only_trace(local_run):
    assert not [r for r in local_run.rows if r.get("error")]
    traces = load_run_traces(local_run.experiment_id, "r_test")
    assert len(traces) == 2 * 4 * 2
    for t in traces:
        for span in t.data.spans:
            guard(span.inputs)
            guard(span.outputs)
        names = {s.name for s in t.data.spans}
        if t.info.tags["condition"] == "as_heard":
            assert {"voice_turn", "asr", "agent", "llm", "tts", "tts_reasr"} <= names
            tts = t.search_spans(name="tts")[0]
            assert os.path.exists(tts.outputs["tts_uri"])
        else:
            # The true-transcript run skips ASR, and its TTS speaks the reference answer.
            assert "asr" not in names and "tts" in names
            assert t.search_spans(name="tts")[0].outputs["tts_uri"].endswith("_reference.wav")
        assert os.path.exists(t.search_spans(name="voice_turn")[0].inputs["audio_uri"])


def test_both_conditions_share_the_request(local_run):
    """MLflow's run comparison pairs traces by their request, so it must not depend on the condition."""
    traces = load_run_traces(local_run.experiment_id, "r_test")
    requests = {}
    for t in traces:
        root = t.search_spans(name="voice_turn")[0]
        requests.setdefault((t.info.tags["candidate"], t.info.tags["utterance_id"]), set()).add(
            json.dumps(root.inputs, sort_keys=True))
    assert all(len(r) == 1 for r in requests.values())
    assert all("question" in json.loads(next(iter(r))) for r in requests.values())


def test_expectations_ride_on_the_trace(local_run):
    t = load_run_traces(local_run.experiment_id, "r_test")[0]
    names = {a.name for a in t.info.assessments if getattr(a, "expectation", None) is not None}
    assert {"gt_transcript", "expected_tool_calls", "expected_facts", "reference_answer"} <= names
    # The Correctness judge raises when a trace carries both expected_facts and expected_response.
    assert "expected_response" not in names


def test_stage1_catches_planted_asr_errors(results):
    assert value(results, "baseline_cascade", "u_006", "asr_wer") == 0
    assert value(results, "mock_noisy", "u_006", "asr_term_error_rate") == 1.0  # TLF-2201
    assert value(results, "mock_noisy", "u_008", "asr_number_accuracy") == 0.0  # 14 -> 40
    assert value(results, "mock_noisy", "u_008", "asr_wer") > 0


def test_stage2_catches_wrong_tool_args_and_declines(results):
    assert value(results, "baseline_cascade", "u_006", "tool_args_match") == 1.0
    assert value(results, "mock_noisy", "u_006", "tool_args_match") == 0.0
    assert value(results, "mock_noisy", "u_006", "tool_args_match", "oracle") == 1.0
    assert value(results, "baseline_cascade", "u_010", "tool_name_match") == 1.0
    assert value(results, "baseline_cascade", "u_006", "answer_exact_numeric") == 1.0
    assert value(results, "mock_noisy", "u_006", "answer_exact_numeric") == 0.0


def test_stage3_catches_truncated_speech(results):
    assert value(results, "baseline_cascade", "u_005", "tts_complete") == 1.0
    assert value(results, "mock_noisy", "u_005", "tts_complete") == 0.0
    assert value(results, "baseline_cascade", "u_005", "tts_roundtrip_wer") == 0
    assert value(results, "mock_noisy", "u_005", "tts_number_accuracy") < 1


def test_speaking_is_scored_on_the_reference_answer_too(results):
    # Same reference text for every candidate, so the voices are compared on identical sentences.
    assert value(results, "baseline_cascade", "u_006", "tts_roundtrip_wer", "oracle") == 0
    assert value(results, "mock_noisy", "u_005", "tts_complete", "oracle") == 0.0  # planted cut-off


def test_not_applicable_scorers_are_skipped(results):
    na = results[(results.utterance_id == "u_010") & (results.metric == "asr_number_accuracy")]
    assert na.empty  # no numbers in "Who's the best hip surgeon in Denver?"


def test_asr_impact_and_attribution(results):
    impact = asr_impact(results).set_index(["candidate", "metric"])
    assert impact.loc[("mock_noisy", "tool_args_match"), "asr_impact"] > 0
    assert impact.loc[("baseline_cascade", "tool_args_match"), "asr_impact"] == 0
    labels = attribution(results).set_index(["candidate", "utterance_id"])["attribution"]
    assert labels[("mock_noisy", "u_006")] == "ASR-caused"
    assert labels[("baseline_cascade", "u_006")] == "pass"


def test_feedback_lands_on_the_pipeline_traces(local_run, results):
    pipeline_ids = {r["trace_id"] for r in local_run.rows}
    assert set(results.trace_id) <= pipeline_ids


def test_spoken_word_count_reads_numbers_and_codes_aloud():
    from voice_evals.scorers import spoken_word_count

    # The(1) list(1) price(1) for(1) T-L-F(3) two thousand two hundred ten(5) is(1) $ -> dollars(1)
    # one thousand eight hundred forty(5)
    assert spoken_word_count("The list price for TLF 2210 is $1,840.") == 19
    assert spoken_word_count("Order 58213 shipped on November 3, 2026.") == 1 + 6 + 1 + 1 + 1 + 1 + 4
    assert spoken_word_count("It's 4.5 millimeters.") == 1 + 3 + 1


class _Span:
    def __init__(self, name, inputs, outputs, span_type="UNKNOWN"):
        self.name, self.inputs, self.outputs, self.span_type = name, inputs, outputs, span_type


class _Trace:
    def __init__(self, spans):
        self.spans = spans

    def search_spans(self, span_type=None, name=None):
        return [s for s in self.spans if (name is None or s.name == name) and (span_type is None or s.span_type == span_type)]


def test_scorers_read_json_string_payloads():
    """Traces read back from UC hold span inputs and outputs as JSON text, not dicts."""
    from voice_evals.scorers import (answer_exact_numeric, asr_wer, spoken_suitability, tool_args_match,
                                     tts_complete, tts_roundtrip_wer)

    trace = _Trace([
        _Span("asr", '{"audio_uri": "/Volumes/x/u_006.wav"}', '{"text": "whats the list price of tlf 2210"}'),
        _Span("get_product", '{"sku": "TLF-2210"}', '{"list_price_usd": 1840}', span_type="TOOL"),
        _Span("tts", '{"text": "It lists at $1,840."}', '{"duration_s": 1.5}'),
        _Span("tts_reasr", '{"tts_uri": "/Volumes/x/t.wav"}', '{"text": "it lists at 1840 dollars"}'),
    ])
    exp = {"gt_transcript": "What's the list price of TLF-2210?", "domain_terms": ["TLF-2210"],
           "expected_tool_calls": [{"name": "get_product", "args": {"sku": "TLF-2210"}}], "expected_numbers": ["1840"]}
    answer = '{"answer": "It lists at $1,840.", "tts_uri": "/Volumes/x/t.wav"}'
    assert asr_wer(trace=trace, expectations=exp).value == 0
    assert tool_args_match(trace=trace, expectations=exp).value == 1.0
    assert tts_roundtrip_wer(trace=trace).value == 0
    assert tts_complete(trace=trace).value is True
    assert spoken_suitability(outputs=answer).value is True
    assert answer_exact_numeric(outputs=answer, expectations=exp).value is True
