"""Tests for durable attempt artifacts and provider-request accounting."""

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from baselines.llm.eval_utils.client import ModelResponse
from baselines.llm.eval_utils.evaluator import (
    _archive_incomplete_attempt,
    _attempt_ledger_guard,
    _episode_result_is_complete,
    _record_failed_transport,
    _record_model_response,
)


def _episode_log():
    log = {
        "stop_reason_counts": defaultdict(int),
        "incomplete_response_count": 0,
        "incomplete_response_reasons": defaultdict(int),
        "transport_error_reasons": defaultdict(int),
        "model_call_count": 0,
        "provider_request_count": 0,
        "transport_error_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "model_latency_seconds": 0.0,
        "agent_0_stop_reason_counts": defaultdict(int),
        "agent_0_incomplete_response_count": 0,
        "agent_0_model_call_count": 0,
        "agent_0_provider_request_count": 0,
        "agent_0_transport_error_count": 0,
        "agent_0_input_tokens": 0,
        "agent_0_output_tokens": 0,
        "agent_0_reasoning_tokens": 0,
        "agent_0_cached_tokens": 0,
        "agent_0_cache_write_tokens": 0,
        "agent_0_model_latency_seconds": 0.0,
    }
    return log


def test_response_and_failed_transport_attempts_are_counted():
    log = _episode_log()
    response = ModelResponse(
        model_id="gpt-test",
        completion="<action>Noop</action>",
        stop_reason="stop",
        input_tokens=100,
        output_tokens=10,
        transport_attempt_count=3,
        transport_error_count=2,
        transport_error_types=("RateLimitError", "InternalServerError"),
    )
    _record_model_response(log, response, 0, "decision")

    client = SimpleNamespace(
        last_call_exception=RuntimeError("exhausted"),
        last_transport_attempt_count=2,
        last_transport_error_count=2,
        last_transport_error_types=("APITimeoutError", "APITimeoutError"),
    )
    _record_failed_transport(log, client, 0, "decision")

    assert log["model_call_count"] == 1
    assert log["provider_request_count"] == 5
    assert log["transport_error_count"] == 4
    assert log["decision_provider_request_count"] == 5
    assert dict(log["transport_error_reasons"]) == {
        "RateLimitError": 1,
        "InternalServerError": 1,
        "APITimeoutError": 2,
    }


def test_policy_filtered_prompt_is_a_counted_non_executable_model_call():
    log = _episode_log()
    response = ModelResponse(
        model_id="gpt-5.6-luna",
        completion="",
        stop_reason="content_filter",
        input_tokens=0,
        output_tokens=0,
        status="failed",
        incomplete_reason="invalid_prompt",
        transport_attempt_count=1,
        transport_error_count=1,
        transport_error_types=("BadRequestError",),
    )

    _record_model_response(log, response, 0, "decision")

    assert log["model_call_count"] == 1
    assert log["provider_request_count"] == 1
    assert log["transport_error_count"] == 1
    assert log["stop_reason_counts"] == {"content_filter": 1}
    assert log["incomplete_response_count"] == 1
    assert log["incomplete_response_reasons"] == {"content_filter": 1}
    assert log["input_tokens"] == 0
    assert log["output_tokens"] == 0
    assert log["transport_error_reasons"] == {"BadRequestError": 1}


def test_incomplete_artifacts_are_archived_and_not_complete(tmp_path):
    task_dir = tmp_path / "alem" / "default"
    task_dir.mkdir(parents=True)
    marker = task_dir / "default_run_00.json"
    marker.write_text("{}\n", encoding="utf-8")
    debug = task_dir / "default_run_00_debug.jsonl"
    debug.write_text('{"step": 0}\n', encoding="utf-8")

    assert not _episode_result_is_complete(marker)
    archive = _archive_incomplete_attempt(tmp_path, "alem", "default", 0)

    assert archive is not None
    assert (archive / marker.name).is_file()
    assert (archive / debug.name).is_file()
    assert not marker.exists()

    marker.write_text(
        json.dumps(
            {
                "schema_version": "alem-dice-episode-v1",
                "artifact_status": "complete",
                "termination_reason": "evaluator_step_cap",
            }
        ),
        encoding="utf-8",
    )
    assert _episode_result_is_complete(marker)
    assert _archive_incomplete_attempt(tmp_path, "alem", "default", 0) is None


@pytest.mark.parametrize("payload", [[], None, "complete"])
def test_non_object_episode_marker_is_incomplete(tmp_path, payload):
    marker = tmp_path / "default_run_00.json"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    assert not _episode_result_is_complete(marker)


def test_attempt_guard_journals_usage_when_postprocessing_raises(tmp_path):
    log = _episode_log()
    log.update(
        {
            "attempt_id": "attempt-1",
            "input_tokens": 123,
            "model_call_count": 1,
            "provider_request_count": 1,
        }
    )

    with pytest.raises(RuntimeError, match="render failed"):
        with _attempt_ledger_guard(
            tmp_path,
            "alem",
            "default",
            0,
            log,
            seed=9999,
            process_num=2,
        ):
            raise RuntimeError("render failed")

    ledger = tmp_path / "alem" / "default" / "attempt_ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["attempt_id"] == "attempt-1"
    assert rows[0]["artifact_status"] == "failed"
    assert rows[0]["termination_reason"] == "error"
    assert rows[0]["input_tokens"] == 123
    assert rows[0]["seed"] == 9999
