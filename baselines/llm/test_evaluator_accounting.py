"""Minimal safeguards for durable provider-cost and attempt accounting."""

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from baselines.llm.eval_utils.client import ModelResponse
from baselines.llm.eval_utils.evaluator import (
    TURN_ACCOUNTING_FEATURES,
    TURN_ACCOUNTING_SCHEMA_VERSION,
    TURN_ACCOUNTING_SEMANTICS,
    _archive_incomplete_attempt,
    _attempt_ledger_guard,
    _classify_action_turn,
    _episode_result_is_complete,
    _record_failed_transport,
    _record_model_response,
    _should_append_parse_feedback,
)
from baselines.llm.eval_utils.performance_metrics import build_performance_metrics
from baselines.llm.utils import (
    _accumulate_attempt_usage,
    collect_and_summarize_results,
    save_summary_stats,
)


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
        "agent_0_transport_error_reasons": defaultdict(int),
        "agent_0_input_tokens": 0,
        "agent_0_output_tokens": 0,
        "agent_0_reasoning_tokens": 0,
        "agent_0_cached_tokens": 0,
        "agent_0_cache_write_tokens": 0,
        "agent_0_model_latency_seconds": 0.0,
    }


@pytest.mark.parametrize(
    ("inactive", "submitted", "executed", "parse_failed", "expected"),
    (
        (
            False,
            "Noop",
            "Noop",
            False,
            {
                "parse_classification": "success",
                "intentional_actionable_noop": True,
                "parse_fallback_noop": False,
                "active_action_validation_fallback_noop": False,
                "inactive_submitted_turn": False,
                "canonical_submitted_noop": True,
                "effective_environment_noop": True,
            },
        ),
        (
            False,
            "Noop",
            "Noop",
            True,
            {
                "parse_classification": "failure",
                "intentional_actionable_noop": False,
                "parse_fallback_noop": True,
                "active_action_validation_fallback_noop": False,
                "inactive_submitted_turn": False,
                "canonical_submitted_noop": True,
                "effective_environment_noop": True,
            },
        ),
        (
            True,
            "Move North",
            "Move North",
            False,
            {
                "parse_classification": "skipped_inactive",
                "intentional_actionable_noop": False,
                "parse_fallback_noop": False,
                "active_action_validation_fallback_noop": False,
                "inactive_submitted_turn": True,
                "canonical_submitted_noop": False,
                "effective_environment_noop": True,
            },
        ),
        (
            False,
            "Move North",
            "Move North",
            False,
            {
                "parse_classification": "success",
                "intentional_actionable_noop": False,
                "parse_fallback_noop": False,
                "active_action_validation_fallback_noop": False,
                "inactive_submitted_turn": False,
                "canonical_submitted_noop": False,
                "effective_environment_noop": False,
            },
        ),
        (
            False,
            "Give to Agent 0",
            "Noop",
            False,
            {
                "parse_classification": "success",
                "intentional_actionable_noop": False,
                "parse_fallback_noop": False,
                "active_action_validation_fallback_noop": True,
                "inactive_submitted_turn": False,
                "canonical_submitted_noop": True,
                "effective_environment_noop": True,
            },
        ),
    ),
)
def test_action_turn_classification_uses_pre_step_state(
    inactive, submitted, executed, parse_failed, expected
):
    classification = _classify_action_turn(
        pre_step_inactive=inactive,
        submitted_action=submitted,
        executed_action=executed,
        parse_failed=parse_failed,
    )
    assert {key: classification[key] for key in expected} == expected
    partition = sum(
        int(classification[key])
        for key in (
            "intentional_actionable_noop",
            "parse_fallback_noop",
            "active_action_validation_fallback_noop",
            "inactive_effective_noop",
            "active_residual_effective_noop",
        )
    )
    assert partition == int(classification["effective_environment_noop"])


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
    assert dict(log["agent_0_transport_error_reasons"]) == {
        "RateLimitError": 1,
        "InternalServerError": 1,
        "APITimeoutError": 2,
    }
    assert log["model_usage_records"][0]["transport_attempt_count"] == 3
    assert log["model_usage_records"][0]["transport_error_count"] == 2
    assert log["model_usage_records"][0]["transport_error_types"] == [
        "RateLimitError",
        "InternalServerError",
    ]


def _versioned_turn_accounting_record():
    record = {
        "termination_reason": "environment_truncated",
        "num_steps": 4,
        "turn_accounting_schema_version": TURN_ACCOUNTING_SCHEMA_VERSION,
        "turn_accounting_features": list(TURN_ACCOUNTING_FEATURES),
        "turn_accounting_semantics": dict(TURN_ACCOUNTING_SEMANTICS),
        "turn_accounting_provenance": "evaluator_exact_pre_step",
        "turn_accounting_complete": True,
        "physical_worker_count": 2,
        "action_parse_success": 4,
        "action_parse_fail": 1,
        "action_parse_skipped_inactive": 3,
        "intentional_actionable_noop_count": 2,
        "parse_fallback_noop_count": 1,
        "active_action_validation_fallback_noop_count": 1,
        "active_residual_effective_noop_count": 0,
        "inactive_submitted_turn_count": 3,
        "inactive_effective_noop_count": 3,
        "canonical_submitted_noop_count": 4,
        "effective_environment_noop_count": 7,
        "executed_noop_count": 4,
    }
    worker_values = {
        0: {
            "parse_success": 3,
            "parse_fail": 1,
            "parse_skipped_inactive": 0,
            "intentional_actionable_noop_count": 2,
            "parse_fallback_noop_count": 1,
            "active_action_validation_fallback_noop_count": 1,
            "active_residual_effective_noop_count": 0,
            "inactive_submitted_turn_count": 0,
            "inactive_effective_noop_count": 0,
            "canonical_submitted_noop_count": 4,
            "effective_environment_noop_count": 4,
            "executed_noop_count": 4,
        },
        1: {
            "parse_success": 1,
            "parse_fail": 0,
            "parse_skipped_inactive": 3,
            "intentional_actionable_noop_count": 0,
            "parse_fallback_noop_count": 0,
            "active_action_validation_fallback_noop_count": 0,
            "active_residual_effective_noop_count": 0,
            "inactive_submitted_turn_count": 3,
            "inactive_effective_noop_count": 3,
            "canonical_submitted_noop_count": 0,
            "effective_environment_noop_count": 3,
            "executed_noop_count": 0,
        },
    }
    for worker_id, values in worker_values.items():
        for suffix, value in values.items():
            record[f"agent_{worker_id}_{suffix}"] = value
    return record


def test_source_feedback_gate_remains_post_step_while_metrics_use_pre_step():
    classification = _classify_action_turn(
        pre_step_inactive=False,
        submitted_action="Noop",
        executed_action="Noop",
        parse_failed=True,
    )
    assert classification["parse_classification"] == "failure"
    # Source historically suppresses correction when env.step reports that the
    # worker has just become inactive. Accounting must not change that behavior.
    assert not _should_append_parse_feedback(
        parse_failed=True,
        post_step_inactive=True,
    )
    assert _should_append_parse_feedback(
        parse_failed=True,
        post_step_inactive=False,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda record: record.pop("agent_1_parse_skipped_inactive"),
            "invalid or missing agent_1_parse_skipped_inactive",
        ),
        (
            lambda record: record.update(action_parse_skipped_inactive=2),
            "physical-worker sum",
        ),
        (
            lambda record: record.update(inactive_submitted_turn_count=2),
            "physical-worker sum",
        ),
        (
            lambda record: record.update(effective_environment_noop_count=6),
            "physical-worker sum",
        ),
        (
            lambda record: record.update(canonical_submitted_noop_count=2),
            "physical-worker sum",
        ),
        (
            lambda record: record.pop("turn_accounting_features"),
            "feature declaration",
        ),
        (
            lambda record: record.pop("turn_accounting_schema_version"),
            "lacks a schema declaration",
        ),
    ),
)
def test_v2_turn_accounting_rejects_missing_or_contradictory_fields(mutation, message):
    record = _versioned_turn_accounting_record()
    mutation(record)
    data = defaultdict(int)
    for field in (
        "termination_reason_counts",
        "transport_error_reasons",
        "incomplete_response_reasons",
        "stop_reason_counts",
    ):
        data[field] = defaultdict(int)
    with pytest.raises(ValueError, match=message):
        _accumulate_attempt_usage(data, record)


def _mutate_skipped_inactive_equivalence(record):
    record["action_parse_skipped_inactive"] = 2
    record["agent_1_parse_skipped_inactive"] = 2


def _mutate_effective_identity(record):
    record["effective_environment_noop_count"] = 6
    record["agent_1_effective_environment_noop_count"] = 2


def _mutate_canonical_bounds(record):
    record["canonical_submitted_noop_count"] = 2
    record["executed_noop_count"] = 2
    record["agent_0_canonical_submitted_noop_count"] = 2
    record["agent_0_executed_noop_count"] = 2


def _mutate_turn_coverage(record):
    record["action_parse_success"] = 3
    record["agent_0_parse_success"] = 2


def _mutate_parse_fallback_identity(record):
    record["parse_fallback_noop_count"] = 0
    record["agent_0_parse_fallback_noop_count"] = 0
    record["canonical_submitted_noop_count"] = 3
    record["agent_0_canonical_submitted_noop_count"] = 3
    record["executed_noop_count"] = 3
    record["agent_0_executed_noop_count"] = 3
    record["effective_environment_noop_count"] = 6
    record["agent_0_effective_environment_noop_count"] = 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (_mutate_skipped_inactive_equivalence, r"success\+fail\+skipped"),
        (_mutate_effective_identity, "effective-Noop identity"),
        (_mutate_canonical_bounds, "canonical submitted Noops"),
        (_mutate_turn_coverage, r"success\+fail\+skipped"),
        (_mutate_parse_fallback_identity, "do not equal active parse failures"),
    ),
)
def test_v2_turn_accounting_rejects_reconciled_but_impossible_counts(mutation, message):
    record = _versioned_turn_accounting_record()
    mutation(record)
    data = defaultdict(int)
    for field in (
        "termination_reason_counts",
        "transport_error_reasons",
        "incomplete_response_reasons",
        "stop_reason_counts",
    ):
        data[field] = defaultdict(int)
    with pytest.raises(ValueError, match=message):
        _accumulate_attempt_usage(data, record)


def test_versioned_noop_taxonomy_is_preserved_in_aggregate_usage():
    data = defaultdict(int)
    for field in (
        "termination_reason_counts",
        "transport_error_reasons",
        "incomplete_response_reasons",
        "stop_reason_counts",
    ):
        data[field] = defaultdict(int)

    _accumulate_attempt_usage(data, _versioned_turn_accounting_record())

    assert data["action_parse_skipped_inactive"] == 3
    assert data["intentional_actionable_noop_count"] == 2
    assert data["parse_fallback_noop_count"] == 1
    assert data["inactive_submitted_turn_count"] == 3
    assert data["canonical_submitted_noop_count"] == 4
    assert data["effective_environment_noop_count"] == 7
    assert data["executed_noop_count"] == 4
    assert data["turn_accounting_covered_attempt_count"] == 1
    assert data["turn_accounting_unavailable_attempt_count"] == 0


def test_old_attempt_ledgers_mark_turn_accounting_unavailable_not_zero():
    data = defaultdict(int)
    for field in (
        "termination_reason_counts",
        "transport_error_reasons",
        "incomplete_response_reasons",
        "stop_reason_counts",
    ):
        data[field] = defaultdict(int)
    _accumulate_attempt_usage(
        data,
        {
            "termination_reason": "environment_truncated",
            "action_parse_success": 99,
            "intentional_actionable_noop_count": 99,
        },
    )
    assert data["turn_accounting_covered_attempt_count"] == 0
    assert data["turn_accounting_unavailable_attempt_count"] == 1
    assert data["action_parse_success"] == 0
    assert data["intentional_actionable_noop_count"] == 0


def test_old_attempt_summary_serializes_turn_accounting_as_unavailable(tmp_path):
    ledger = tmp_path / "alem" / "default" / "attempt_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "alem-dice-attempt-v1",
                "episode_index": 0,
                "artifact_status": "complete",
                "termination_reason": "environment_truncated",
                "action_parse_success": 99,
                "intentional_actionable_noop_count": 99,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = collect_and_summarize_results(tmp_path)
    save_summary_stats(summary, tmp_path)
    clean = json.loads((tmp_path / "summary_stats.json").read_text(encoding="utf-8"))[
        "alem/default"
    ]
    assert clean["turn_accounting_coverage"] == "unavailable"
    assert clean["turn_accounting_covered_attempt_count"] == 0
    assert clean["turn_accounting_unavailable_attempt_count"] == 1
    assert clean["action_parse_success"] is None
    assert clean["intentional_actionable_noop_count"] is None


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
            "intentional_actionable_noop_count": 2,
            "parse_fallback_noop_count": 1,
            "inactive_submitted_turn_count": 3,
            "executed_noop_count": 6,
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
    assert rows[0]["intentional_actionable_noop_count"] == 2
    assert rows[0]["parse_fallback_noop_count"] == 1
    assert rows[0]["inactive_submitted_turn_count"] == 3
    assert rows[0]["executed_noop_count"] == 6
    assert rows[0]["turn_accounting_coverage"] == "unavailable"


def test_attempt_ledger_persists_and_verifies_physical_worker_accounting(tmp_path):
    log = _episode_log()
    log.update(_versioned_turn_accounting_record())
    log.update(
        {
            "attempt_id": "attempt-accounted",
            "logical_participant_count": 3,
            "agent_1_model_call_count": 0,
            "agent_1_provider_request_count": 0,
            "agent_1_transport_error_count": 0,
            "agent_1_input_tokens": 0,
            "agent_1_output_tokens": 0,
            "agent_1_reasoning_tokens": 0,
            "agent_1_cached_tokens": 0,
            "agent_1_cache_write_tokens": 0,
            "agent_1_model_latency_seconds": 0.0,
        }
    )
    with _attempt_ledger_guard(
        tmp_path,
        "alem",
        "default",
        0,
        log,
        seed=9999,
        process_num=0,
    ):
        pass

    row = json.loads(
        (tmp_path / "alem" / "default" / "attempt_ledger.jsonl").read_text(encoding="utf-8").strip()
    )
    assert row["turn_accounting_coverage"] == "complete"
    assert row["physical_worker_count"] == 2
    assert row["agent_0_canonical_submitted_noop_count"] == 4
    assert row["agent_1_canonical_submitted_noop_count"] == 0
    assert (
        sum(
            row[f"agent_{worker}_effective_environment_noop_count"]
            for worker in range(row["physical_worker_count"])
        )
        == row["effective_environment_noop_count"]
    )


def test_v2_attempt_emitter_persists_decision_provider_and_per_call_cache_usage(
    tmp_path,
):
    log = _episode_log()
    log.update(_versioned_turn_accounting_record())
    log["attempt_id"] = "attempt-real-emitter"
    _record_model_response(
        log,
        ModelResponse(
            model_id="gpt-5.4-nano-2026-03-17",
            completion="<action>Noop</action>",
            stop_reason="stop",
            input_tokens=11,
            cached_tokens=4,
            output_tokens=3,
            reasoning_tokens=2,
            cache_write_tokens=1,
            response_id="response-real-emitter",
            status="completed",
            transport_attempt_count=2,
            transport_error_count=1,
            transport_error_types=("APIConnectionError",),
        ),
        0,
        "decision",
    )
    with _attempt_ledger_guard(
        tmp_path,
        "alem",
        "default",
        0,
        log,
        seed=9999,
        process_num=0,
    ):
        pass

    row = json.loads(
        (tmp_path / "alem" / "default" / "attempt_ledger.jsonl").read_text(encoding="utf-8").strip()
    )
    assert row["turn_accounting_coverage"] == "complete"
    assert row["decision_provider_request_count"] == 2
    assert row["transport_error_count"] == 1
    assert row["transport_error_reasons"] == {"APIConnectionError": 1}
    assert row["agent_0_provider_request_count"] == 2
    assert row["agent_0_transport_error_count"] == 1
    assert row["agent_0_transport_error_reasons"] == {"APIConnectionError": 1}
    assert row["decision_input_tokens"] == 11
    assert row["decision_cached_tokens"] == 4
    assert row["decision_output_tokens"] == 3
    assert row["decision_reasoning_tokens"] == 2
    assert row["decision_cache_write_tokens"] == 1
    assert row["model_usage_records"] == [
        {
            "participant_id": 0,
            "phase": "decision",
            "model_id": "gpt-5.4-nano-2026-03-17",
            "response_id": "response-real-emitter",
            "provider_status": "completed",
            "stop_reason": "stop",
            "incomplete_reason": None,
            "input_tokens": 11,
            "cached_tokens": 4,
            "output_tokens": 3,
            "reasoning_tokens": 2,
            "cache_write_tokens": 1,
            "latency_seconds": 0.0,
            "transport_attempt_count": 2,
            "transport_error_count": 1,
            "transport_error_types": ["APIConnectionError"],
        }
    ]


def test_attempt_ledger_rejects_inconsistent_worker_sum(tmp_path):
    log = _episode_log()
    log.update(_versioned_turn_accounting_record())
    log["agent_1_effective_environment_noop_count"] = 2
    with pytest.raises(ValueError, match="effective-Noop identity"):
        with _attempt_ledger_guard(
            tmp_path,
            "alem",
            "default",
            0,
            log,
            seed=9999,
            process_num=0,
        ):
            pass


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
