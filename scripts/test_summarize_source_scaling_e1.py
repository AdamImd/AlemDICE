"""Focused synthetic checks for the E1 analysis contract."""

import json

import pytest

from scripts.summarize_source_scaling_e1 import (
    _debug_noop_metrics,
    _episode_noop_metrics,
    discover_rows,
    episode_row,
    paired_population_contrasts,
    validate_manifest_grid,
)


def _episode_payload(*, seed=7):
    return {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "termination_reason": "environment_truncated",
        "physical_worker_count": 2,
        "seed": seed,
        "episode_return": 2.5,
        "input_tokens": 100,
        "cached_tokens": 40,
        "output_tokens": 20,
        "reasoning_tokens": 10,
        "model_call_count": 20,
        "provider_request_count": 20,
        "model_latency_seconds": 30.0,
        "episode_wall_seconds": 20.0,
        "mean_tick_wall_seconds": 2.0,
        "intentional_actionable_noop_count": 2,
        "parse_fallback_noop_count": 1,
        "inactive_submitted_turn_count": 10,
        "executed_noop_count": 13,
        "action_frequency": {"Noop": 13, "Do": 7},
        "communication_metrics": {"worker_peer": {"delivery_bytes": 50}},
        "performance_metrics": {
            "schema_version": "alem-dice-performance-v1",
            "paper_score_percent": {"base": 2.0, "coord": 4.0, "total": 3.0},
            "achievement_coverage_percent": {"base": 5.0, "coord": 6.0, "total": 5.5},
            "achievement_first_unlock_count": {
                "team_unique": {"base": 2, "coord": 1, "total": 3},
                "summed_across_agents": {"base": 3, "coord": 1, "total": 4},
            },
            "event_counters": {},
            "exposure": {
                "environment_steps_completed": 10,
                "completed_agent_turn_capacity": 20,
                "agent_turns_submitted": 20,
                "classified_action_turns": 20,
                "alive_agent_turns": 12,
                "actionable_agent_turns": 10,
                "survival_fraction": 0.6,
                "actionable_fraction": 0.5,
            },
        },
    }


def test_episode_row_labels_cache_rates_and_latency(tmp_path):
    root = tmp_path
    path = root / "n2" / "easy" / "alem" / "default" / "default_run_00.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_episode_payload()), encoding="utf-8")

    row = episode_row(path, root, requested_environment_steps=10)

    assert row["cached_input_tokens"] == 40
    assert row["uncached_input_tokens"] == 60
    assert row["input_cache_fraction"] == 0.4
    assert row["input_tokens_per_submitted_turn"] == 5
    assert row["input_tokens_per_actionable_turn"] == 10
    assert row["total_tokens_per_submitted_turn"] == 6
    assert row["total_tokens_per_actionable_turn"] == 12
    assert row["delivered_bytes_per_submitted_turn"] == 2.5
    assert row["delivered_bytes_per_actionable_turn"] == 5
    assert row["summed_model_latency_seconds"] == 30
    assert row["step_completion_fraction"] == 1
    assert row["noop_metrics_provenance"] == "episode_exact_pre_step"


def test_legacy_debug_reconstructs_death_transition_noops(tmp_path):
    episode = tmp_path / "default_run_00.json"
    debug = tmp_path / "default_run_00_debug.jsonl"
    records = [
        {
            "step": 0,
            "agents": {
                "0": {
                    "llm_raw_output": "<action>Noop</action>",
                    "parsed_action": "Noop",
                    "stop_reason": "stop",
                },
                "1": {
                    "llm_raw_output": "<action><action>Do</action></action>",
                    "parsed_action": "Noop",
                    "stop_reason": "stop",
                },
            },
            # Agent 1 dies during this step. The legacy evaluator called its
            # final actionable turn "skipped" using the post-step state.
            "action_parse_stats": {
                "0": {"success": 1, "fail": 0, "skipped_inactive": 0},
                "1": {"success": 0, "fail": 0, "skipped_inactive": 1},
            },
        },
        {
            "step": 1,
            "agents": {
                "0": {
                    "llm_raw_output": "<action>Move North</action>",
                    "parsed_action": "Move North",
                    "stop_reason": "stop",
                },
                "1": {
                    "llm_raw_output": "<action>Noop</action>",
                    "parsed_action": "Noop",
                    "stop_reason": "stop",
                },
            },
            "action_parse_stats": {
                "0": {"success": 2, "fail": 0, "skipped_inactive": 0},
                "1": {"success": 0, "fail": 0, "skipped_inactive": 2},
            },
        },
    ]
    debug.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    metrics = _debug_noop_metrics(
        episode,
        num_agents=2,
        expected_submitted_turns=4,
    )

    assert metrics is not None
    assert metrics["noop_metrics_provenance"] == "debug_jsonl_legacy_reconstruction"
    assert metrics["noop_metrics_complete"] is True
    assert metrics["intentional_actionable_noops"] == 1
    assert metrics["parse_fallback_noops"] == 1
    assert metrics["inactive_billed_turns"] == 1
    assert metrics["executed_noops"] == 3


def test_missing_debug_preserves_only_recoverable_aggregate_noop_fields(tmp_path):
    payload = _episode_payload()
    for field in (
        "intentional_actionable_noop_count",
        "parse_fallback_noop_count",
        "inactive_submitted_turn_count",
        "executed_noop_count",
    ):
        payload.pop(field)

    metrics = _episode_noop_metrics(
        payload,
        tmp_path / "default_run_00.json",
        num_agents=2,
        expected_submitted_turns=20,
    )

    assert metrics["noop_metrics_provenance"] == "episode_aggregate_partial"
    assert metrics["intentional_actionable_noops"] is None
    assert metrics["parse_fallback_noops"] is None
    assert metrics["inactive_billed_turns"] == 10
    assert metrics["executed_noops"] == 13


@pytest.mark.parametrize("allow_incomplete", (False, True))
def test_manifest_grid_requires_explicit_incomplete_watermark(allow_incomplete):
    manifest = {
        "planned_counts": [1, 2],
        "seeds": [10, 11],
        "episodes_per_count": 2,
    }
    rows = [
        {"num_agents": 1, "seed": 10, "episode_index": 0},
        {"num_agents": 1, "seed": 11, "episode_index": 1},
        {"num_agents": 2, "seed": 10, "episode_index": 0},
    ]
    if not allow_incomplete:
        with pytest.raises(ValueError, match="--allow-incomplete"):
            validate_manifest_grid(rows, manifest, allow_incomplete=False)
        return

    audit = validate_manifest_grid(rows, manifest, allow_incomplete=True)
    assert audit["complete"] is False
    assert audit["missing_pairs"] == [{"num_agents": 2, "seed": 11}]
    assert audit["watermark"].startswith("INCOMPLETE INTERIM ANALYSIS")


def test_absent_manifest_also_requires_explicit_incomplete_mode():
    rows = [{"num_agents": 1, "seed": 10, "episode_index": 0}]
    with pytest.raises(ValueError, match="--allow-incomplete"):
        validate_manifest_grid(rows, None, allow_incomplete=False)
    audit = validate_manifest_grid(rows, None, allow_incomplete=True)
    assert audit["complete"] is None
    assert audit["watermark"].startswith("MANIFEST ABSENT")


def test_discovery_rejects_duplicate_population_seed(tmp_path):
    for episode_index in (0, 1):
        path = (
            tmp_path
            / "n2"
            / "easy"
            / "alem"
            / f"task{episode_index}"
            / f"default_run_{episode_index:02d}.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_episode_payload(seed=7)), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Duplicate \(population, seed\)"):
        discover_rows(tmp_path, requested_environment_steps=10)


def test_paired_contrast_resamples_common_seed_differences():
    rows = [
        {"num_agents": 1, "seed": 10, "paper_total_percent": 1.0},
        {"num_agents": 1, "seed": 11, "paper_total_percent": 3.0},
        {"num_agents": 2, "seed": 10, "paper_total_percent": 4.0},
        {"num_agents": 2, "seed": 11, "paper_total_percent": 8.0},
    ]
    contrasts = paired_population_contrasts(rows, reps=100, bootstrap_seed=123)
    estimate = contrasts["n2_minus_n1"]["metrics"]["paper_total_percent"]

    assert estimate["common_seeds"] == [10, 11]
    assert estimate["common_seed_count"] == 2
    assert estimate["mean_difference"] == 4.0
