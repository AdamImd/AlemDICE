"""Minimal safeguards for durable provider-cost and attempt accounting."""

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
from baselines.llm.eval_utils.performance_metrics import build_performance_metrics


def _episode_log():
    return {
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


def test_response_and_failed_transport_attempts_are_counted():
    log = _episode_log()
    _record_model_response(
        log,
        ModelResponse(
            model_id="gpt-test",
            completion="<action>Noop</action>",
            stop_reason="stop",
            input_tokens=100,
            output_tokens=10,
            transport_attempt_count=3,
            transport_error_count=2,
            transport_error_types=("RateLimitError", "InternalServerError"),
        ),
        0,
        "decision",
    )
    _record_failed_transport(
        log,
        SimpleNamespace(
            last_call_exception=RuntimeError("exhausted"),
            last_transport_attempt_count=2,
            last_transport_error_count=2,
            last_transport_error_types=("APITimeoutError", "APITimeoutError"),
        ),
        0,
        "decision",
    )

    assert log["model_call_count"] == 1
    assert log["provider_request_count"] == 5
    assert log["transport_error_count"] == 4
    assert dict(log["transport_error_reasons"]) == {
        "RateLimitError": 1,
        "InternalServerError": 1,
        "APITimeoutError": 2,
    }


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
    assert rows[0]["artifact_status"] == "failed"
    assert rows[0]["input_tokens"] == 123


def test_performance_metrics_preserve_score_count_and_exposure_semantics():
    log = {
        "num_steps": 10,
        "action_parse_success": 21,
        "action_parse_fail": 1,
        "action_parse_skipped_inactive": 8,
        "user_info": {
            "Team/normal_reward_pct_of_max": 0.125,
            "Team/coord_reward_pct_of_max": 0.25,
            "Team/reward_pct_of_max": 0.175,
            "Team/normal_achievement_pct": 0.2,
            "Team/coordination_achievement_pct": 0.1,
            "Team/achievement_pct": 0.16,
            "Team/normal_achievements": 4.0,
            "Team/coordination_achievements": 2.0,
            "Team/total_achievements": 6.0,
            "Agent0/normal_achievements": 3.0,
            "Agent0/coordination_achievements": 2.0,
            "Agent0/total_achievements": 5.0,
            "Agent1/normal_achievements": 2.0,
            "Agent1/coordination_achievements": 1.0,
            "Agent1/total_achievements": 3.0,
            "Agent2/normal_achievements": 1.0,
            "Agent2/coordination_achievements": 0.0,
            "Agent2/total_achievements": 1.0,
            "Coordination/total_attempts": 7.0,
            "Coordination/total_resolved_attempts": 6.0,
            "Coordination/total_successes": 3.0,
            "Cooperation/give_attempt_count": 4.0,
            "Cooperation/trade_count": 2.0,
            "Cooperation/request_count": 5.0,
            "Cooperation/revives": 1.0,
        },
    }

    metrics = build_performance_metrics(
        log,
        3,
        environment_steps_completed=10,
        agent_turns_submitted=30,
        alive_agent_turns=24,
        actionable_agent_turns=22,
    )

    assert metrics["paper_score_percent"] == {
        "base": 12.5,
        "coord": 25.0,
        "total": 17.5,
    }
    assert metrics["achievement_first_unlock_count"]["team_unique"] == {
        "base": 4,
        "coord": 2,
        "total": 6,
    }
    # The sum intentionally counts the same achievement once per attaining agent.
    assert metrics["achievement_first_unlock_count"]["summed_across_agents"] == {
        "base": 6,
        "coord": 3,
        "total": 9,
    }
    assert metrics["event_counters"]["coordination_attempts"] == 7
    assert metrics["exposure"] == {
        "environment_steps_completed": 10,
        "completed_agent_turn_capacity": 30,
        "agent_turns_submitted": 30,
        "classified_action_turns": 30,
        "alive_agent_turns": 24,
        "actionable_agent_turns": 22,
        "survival_fraction": 0.8,
        "actionable_fraction": 22 / 30,
    }

    solo = build_performance_metrics(
        {
            "num_steps": 2,
            "user_info": {
                "Team/normal_reward_pct_of_max": 0.1,
                "Team/reward_pct_of_max": 0.1,
                "Team/normal_achievements": 2,
                "Team/total_achievements": 2,
                "Agent0/normal_achievements": 2,
                "Agent0/total_achievements": 2,
            },
        },
        1,
        alive_agent_turns=2,
        actionable_agent_turns=2,
    )
    assert solo["paper_score_percent"]["coord"] is None
    assert solo["achievement_first_unlock_count"]["team_unique"]["coord"] is None
    assert solo["event_counters"]["coordination_attempts"] is None
