"""Focused tests for the paired three-seed signal-study summarizer."""

import csv
import json

import pytest

from scripts.run_signal_scale_study import (
    ENDPOINTS,
    arm_command,
    episode_complete,
)
from scripts.summarize_signal_scale_study import ARMS, EXPECTED_SEEDS, summarize


def _episode(
    seed,
    *,
    parse_rate,
    achievements,
    coordination_successes,
    episode_return,
    input_tokens=100,
    output_tokens=10,
    model_call_count=600,
    num_steps=200,
    attempt_id=None,
):
    payload = {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "termination_reason": "environment_truncated",
        "seed": seed,
        "num_steps": num_steps,
        "episode_return": episode_return,
        "action_parse_rate": parse_rate,
        "action_parse_success": 600,
        "action_parse_fail": 0,
        "action_parse_skipped_inactive": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "model_call_count": model_call_count,
        "provider_request_count": model_call_count,
        "transport_error_count": 0,
        "model_latency_seconds": 120.0,
        "episode_wall_seconds": 50.0,
        "user_info": {
            "Team/total_achievements": achievements,
            "Coordination/total_attempts": 4,
            "Coordination/total_successes": coordination_successes,
            "Coordination/coordination_success_rate": coordination_successes / 4,
        },
        "coordination_protocol": {
            "strategy": "cohesion",
            "protocol_parse_rate": 1.0,
            "status_window_coverage": 1.0,
            "status": 600,
            "protocol_messages": 600,
        },
        "communication_metrics": {
            "worker_peer": {"delivery_bytes": 1000},
        },
    }
    if attempt_id is not None:
        payload["attempt_id"] = attempt_id
    return payload


def _write_episode(root, arm, episode_index, payload):
    task_dir = root / arm / "alem" / "default"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"default_run_{episode_index:02d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_scale_launcher_command_pins_parallel_seeds_and_equal_budget(tmp_path):
    command = arm_command(tmp_path, tmp_path / "run", "free_concise")
    assert "eval.num_episodes.alem=3" in command
    assert "eval.num_workers=3" in command
    assert "eval.max_steps_per_episode=200" in command
    assert "experiment.seeds=[9999,10000,10001]" in command
    assert "EVAL_SEED=9999" in command
    for index, endpoint in enumerate(ENDPOINTS):
        assert f"clients.{index}.base_url={endpoint}" in command
        assert f"clients.{index}.generate_kwargs.max_tokens=2048" in command


def test_scale_launcher_completion_requires_expected_seed(tmp_path):
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "alem-dice-episode-v1",
                "artifact_status": "complete",
                "termination_reason": "environment_truncated",
                "seed": 9999,
            }
        ),
        encoding="utf-8",
    )
    assert episode_complete(path, 9999) is True
    assert episode_complete(path, 10000) is False


def _complete_study(root):
    settings = {
        "free_thinking": {
            "parse": (0.7, 0.7, 0.7),
            "achievements": (1, 2, 3),
            "successes": (0, 0, 0),
            "output_tokens": 20,
            "model_call_count": 600,
        },
        "free_concise": {
            "parse": (0.8, 0.8, 0.7),
            "achievements": (2, 2, 2),
            "successes": (0, 0, 0),
            "output_tokens": 10,
            "model_call_count": 500,
        },
        "cohesion_concise": {
            "parse": (0.8, 0.8, 0.7),
            "achievements": (3, 3, 2),
            "successes": (1, 1, 0),
            "output_tokens": 10,
            "model_call_count": 500,
        },
    }
    for arm in ARMS:
        for episode_index, seed in enumerate(EXPECTED_SEEDS):
            _write_episode(
                root,
                arm,
                episode_index,
                _episode(
                    seed,
                    parse_rate=settings[arm]["parse"][episode_index],
                    achievements=settings[arm]["achievements"][episode_index],
                    coordination_successes=settings[arm]["successes"][episode_index],
                    episode_return=settings[arm]["achievements"][episode_index],
                    output_tokens=settings[arm]["output_tokens"],
                    model_call_count=settings[arm]["model_call_count"],
                    attempt_id=(
                        "concise-final" if arm == "free_concise" and episode_index == 0 else None
                    ),
                ),
            )


def test_summary_writes_paired_outputs_and_counts_retry_usage(tmp_path):
    run_root = tmp_path / "run"
    _complete_study(run_root)
    concise_task = run_root / "free_concise" / "alem" / "default"
    failed_attempt = {
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 0,
        "attempt_id": "concise-failed",
        "artifact_status": "failed",
        "termination_reason": "error",
        "seed": 9999,
        "input_tokens": 10,
        "output_tokens": 1,
        "model_call_count": 3,
        "provider_request_count": 3,
        "episode_wall_seconds": 5,
    }
    final_attempt = {
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 0,
        "attempt_id": "concise-final",
        "artifact_status": "complete",
        "termination_reason": "environment_truncated",
        "seed": 9999,
        "input_tokens": 100,
        "output_tokens": 10,
        "model_call_count": 500,
        "provider_request_count": 500,
        "episode_wall_seconds": 50,
    }
    concise_task.joinpath("attempt_ledger.jsonl").write_text(
        json.dumps(failed_attempt) + "\n" + json.dumps(final_attempt) + "\n",
        encoding="utf-8",
    )
    events = []
    for offset, arm in enumerate(ARMS):
        events.extend(
            (
                {
                    "event": "arm_started",
                    "arm": arm,
                    "at": f"2026-07-22T0{offset}:00:00+00:00",
                },
                {
                    "event": "arm_finished",
                    "arm": arm,
                    "at": f"2026-07-22T0{offset}:00:10+00:00",
                },
            )
        )
    run_root.joinpath("events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    artifact_dir = tmp_path / "artifacts"
    results_md = tmp_path / "report" / "results.md"
    outputs = summarize(run_root, artifact_dir, results_md)

    assert all(path.is_file() for path in outputs)
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["decisions"]["response_mode"]["status"] == "positive_signal"
    assert summary["decisions"]["cohesion"]["status"] == "positive_signal"
    assert summary["decisions"]["response_mode"]["criteria_evaluated"] is True
    assert summary["decisions"]["cohesion"]["criteria_evaluated"] is True
    concise_9999 = next(
        row
        for row in summary["episodes"]
        if row["arm"] == "free_concise" and row["expected_seed"] == 9999
    )
    assert concise_9999["attempt_count"] == 2
    assert concise_9999["failed_attempt_count"] == 1
    assert concise_9999["cumulative_attempt_usage"]["input_tokens"] == 110
    assert concise_9999["cumulative_attempt_usage"]["output_tokens"] == 11
    concise_aggregate = summary["arm_aggregates"]["free_concise"]
    assert concise_aggregate["cumulative_attempt_usage"]["input_tokens"] == 310
    assert concise_aggregate["arm_elapsed_seconds"] == 10
    response_seed = summary["contrasts"]["response_mode"]["seed_rows"][0]
    assert response_seed["deltas"]["action_parse_rate"] == pytest.approx(0.1)
    assert response_seed["deltas"]["total_achievements"] == 1
    assert response_seed["deltas"]["total_tokens"] == -10
    assert response_seed["deltas"]["model_call_count"] == -100
    assert concise_9999["normalized"]["input_tokens_per_model_call"] == 0.2
    assert concise_9999["normalized"]["output_tokens_per_model_call"] == 0.02
    assert concise_9999["normalized"]["wall_seconds_per_step"] == 0.25
    assert concise_9999["normalized"]["delivery_bytes_per_step"] == 5

    with (artifact_dir / "episodes.csv").open(encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    with (artifact_dir / "paired_contrasts.csv").open(encoding="utf-8") as handle:
        contrast_rows = list(csv.DictReader(handle))
    assert len(episode_rows) == 9
    assert len(contrast_rows) == 6
    assert "positive_signal" in results_md.read_text(encoding="utf-8")
    inventory = json.loads(
        (artifact_dir / "artifact_inventory.json").read_text(encoding="utf-8")
    )
    archived = {
        row["archived_relative_path"]: row for row in inventory["files"]
    }
    raw_path = "audit/raw_episodes/free_concise/default_run_00.json"
    assert archived[raw_path]["present"] is True
    assert archived[raw_path]["sha256"]
    assert (artifact_dir / raw_path).is_file()
    assert archived["audit/study/manifest.json"]["present"] is False


def test_invalid_artifacts_are_not_imputed_or_paired(tmp_path):
    run_root = tmp_path / "run"
    _complete_study(run_root)
    early_natural = _episode(
        9999,
        parse_rate=0.7,
        achievements=1,
        coordination_successes=0,
        episode_return=1,
        num_steps=17,
    )
    early_natural["termination_reason"] = "environment_terminated"
    _write_episode(run_root, "free_thinking", 0, early_natural)
    malformed = run_root / "free_thinking" / "alem" / "default" / "default_run_02.json"
    malformed.write_text("{not-json}", encoding="utf-8")
    failed = _episode(
        10000,
        parse_rate=0.8,
        achievements=2,
        coordination_successes=0,
        episode_return=2,
    )
    failed.update(
        artifact_status="failed",
        error="provider unavailable",
        termination_reason="error",
    )
    _write_episode(run_root, "free_concise", 1, failed)
    mismatched_seed = _episode(
        12345,
        parse_rate=0.8,
        achievements=3,
        coordination_successes=1,
        episode_return=3,
    )
    _write_episode(run_root, "cohesion_concise", 0, mismatched_seed)

    artifact_dir = tmp_path / "artifacts"
    results_md = tmp_path / "results.md"
    summarize(run_root, artifact_dir, results_md)
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))

    natural_row = next(
        row
        for row in summary["episodes"]
        if row["arm"] == "free_thinking" and row["expected_seed"] == 9999
    )
    assert natural_row["valid"] is True
    assert natural_row["execution"] == {
        "num_steps": 17,
        "reached_step_cap": False,
    }
    malformed_row = next(
        row
        for row in summary["episodes"]
        if row["arm"] == "free_thinking" and row["expected_seed"] == 10001
    )
    assert malformed_row["valid"] is False
    assert malformed_row["invalid_reasons"] == ["malformed_json"]
    failed_row = next(
        row
        for row in summary["episodes"]
        if row["arm"] == "free_concise" and row["expected_seed"] == 10000
    )
    assert failed_row["valid"] is False
    assert "artifact_not_complete" in failed_row["invalid_reasons"]
    assert "episode_error" in failed_row["invalid_reasons"]
    mismatch_row = next(
        row
        for row in summary["episodes"]
        if row["arm"] == "cohesion_concise" and row["expected_seed"] == 9999
    )
    assert mismatch_row["valid"] is False
    assert "seed_mismatch" in mismatch_row["invalid_reasons"]

    response = summary["contrasts"]["response_mode"]
    assert response["aggregate"]["valid_pair_count"] == 1
    invalid_pair = next(row for row in response["seed_rows"] if row["seed"] == 10000)
    assert invalid_pair["paired_valid"] is False
    assert all(value is None for value in invalid_pair["deltas"].values())
    assert response["decision"]["status"] == "indeterminate_incomplete_pairs"
    assert response["decision"]["criteria_evaluated"] is False
    assert all(value is None for value in response["decision"]["criteria"].values())
    assert summary["decisions"]["cohesion"]["status"] == "indeterminate_incomplete_pairs"


def test_partial_ledger_uses_ledger_only_for_covered_seed(tmp_path):
    run_root = tmp_path / "run"
    _complete_study(run_root)
    task_dir = run_root / "free_thinking" / "alem" / "default"
    covered = {
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 0,
        "artifact_status": "complete",
        "termination_reason": "environment_truncated",
        "input_tokens": 123,
        "output_tokens": 12,
    }
    task_dir.joinpath("attempt_ledger.jsonl").write_text(
        "{broken}\n" + json.dumps(covered) + "\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    summarize(run_root, artifact_dir, tmp_path / "results.md")
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    rows = {
        row["expected_seed"]: row for row in summary["episodes"] if row["arm"] == "free_thinking"
    }

    assert rows[9999]["cumulative_usage_source"] == "attempt_ledger"
    assert rows[9999]["cumulative_attempt_usage"]["input_tokens"] == 123
    assert rows[10000]["cumulative_usage_source"] == "stable_artifact_fallback"
    assert rows[10000]["cumulative_attempt_usage"]["input_tokens"] == 100
    assert any("malformed ledger row" in warning for warning in summary["audit_warnings"])


def test_missing_episode_reports_and_archives_partial_debug_as_non_efficacy(tmp_path):
    run_root = tmp_path / "run"
    _complete_study(run_root)
    task_dir = run_root / "free_thinking" / "alem" / "default"
    task_dir.joinpath("default_run_00.json").unlink()
    debug_records = [
        {
            "step": 0,
            "rewards": [1.0, 2.0],
            "action_parse_stats": {
                "0": {"success": 0, "fail": 1},
                "1": {"success": 1, "fail": 0},
            },
            "agents": {
                "0": {
                    "llm_raw_output": "",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "latency_seconds": 1.0,
                    "stop_reason": "length",
                    "transport_attempt_count": 2,
                    "transport_error_count": 1,
                },
                "1": {
                    "llm_raw_output": "<action>Noop</action>",
                    "input_tokens": 20,
                    "output_tokens": 3,
                    "latency_seconds": 3.0,
                    "stop_reason": "stop",
                    "transport_attempt_count": 1,
                    "transport_error_count": 0,
                },
            },
            "communication_routes": [
                {"content": "hi", "recipients": [1, 2]},
            ],
        },
        {
            "step": 1,
            "rewards": [3.0, 4.0],
            "action_parse_stats": {
                "0": {"success": 1, "fail": 1},
                "1": {"success": 2, "fail": 0},
            },
            "agents": {
                "0": {
                    "llm_raw_output": "<action>Noop</action>",
                    "input_tokens": 30,
                    "output_tokens": 4,
                    "latency_seconds": 5.0,
                    "incomplete_reason": "max_completion_tokens",
                    "transport_attempt_count": 1,
                    "transport_error_count": 0,
                },
                "1": {
                    "llm_raw_output": "<action>Noop</action>",
                    "input_tokens": 40,
                    "output_tokens": 5,
                    "latency_seconds": 7.0,
                    "stop_reason": "stop",
                    "transport_attempt_count": 1,
                    "transport_error_count": 0,
                },
            },
            "communication_routes": [
                {"content": "é", "recipients": [0]},
            ],
        },
    ]
    task_dir.joinpath("default_run_00_debug.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in debug_records),
        encoding="utf-8",
    )
    concise_debug = (
        run_root
        / "free_concise"
        / "alem"
        / "default"
        / "default_run_00_debug.jsonl"
    )
    concise_debug.write_text(
        "".join(json.dumps(record) + "\n" for record in debug_records),
        encoding="utf-8",
    )
    task_dir.joinpath("default_run_00.csv").write_text("Step\n0\n1\n", encoding="utf-8")
    run_root.joinpath("free_thinking", "eval.log").write_text("interrupted\n", encoding="utf-8")
    run_root.joinpath("free_thinking.attempt_01.console.log").write_text(
        "stopped\n", encoding="utf-8"
    )

    artifact_dir = tmp_path / "artifacts"
    results_md = tmp_path / "results.md"
    summarize(run_root, artifact_dir, results_md)
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    row = next(
        row
        for row in summary["episodes"]
        if row["arm"] == "free_thinking" and row["expected_seed"] == 9999
    )
    partial = row["partial_debug"]
    assert row["valid"] is False
    assert partial["classification"] == "partial_non_efficacy"
    assert partial["eligible_for_efficacy"] is False
    assert partial["observed_steps"] == 2
    assert partial["observed_step_values"] == [0, 1]
    assert partial["reward_sum"] == 10
    assert partial["model_call_count"] == 4
    assert partial["provider_request_count"] == 5
    assert partial["input_tokens"] == 100
    assert partial["output_tokens"] == 14
    assert partial["mean_model_latency_seconds"] == 4
    assert partial["length_stop_count"] == 2
    assert partial["empty_output_count"] == 1
    assert partial["action_parse_success"] == 3
    assert partial["action_parse_fail"] == 1
    assert partial["action_parse_rate"] == 0.75
    assert partial["transport_error_count"] == 1
    assert partial["communications"] == {
        "emitted_messages": 2,
        "delivered_messages": 3,
        "payload_bytes": 4,
        "delivery_bytes": 6,
    }
    assert summary["decisions"]["response_mode"]["status"] == (
        "indeterminate_incomplete_pairs"
    )
    assert summary["decisions"]["response_mode"]["criteria_evaluated"] is False
    assert all(
        value is None
        for value in summary["decisions"]["response_mode"]["criteria"].values()
    )
    assert summary["arm_aggregates"]["free_thinking"]["partial_debug_non_efficacy"][
        "episode_count"
    ] == 1
    matched = summary["matched_prefix_response"]
    assert matched["eligible_for_registered_efficacy"] is False
    assert len(matched["rows"]) == 1
    assert matched["rows"][0]["seed"] == 9999
    assert matched["rows"][0]["steps"] == 2
    assert matched["rows"][0]["matched_steps_complete"] is True
    assert matched["rows"][0]["concise_reward_sum"] == 10
    with (artifact_dir / "matched_prefix_response.csv").open(encoding="utf-8") as handle:
        matched_csv = list(csv.DictReader(handle))
    assert len(matched_csv) == 1
    assert matched_csv[0]["classification"] == (
        "matched_prefix_descriptive_non_efficacy"
    )
    markdown = results_md.read_text(encoding="utf-8")
    assert "Interrupted partial telemetry (non-efficacy)" in markdown
    assert "excluded from efficacy aggregates" in markdown
    assert (
        "Decision criteria were not evaluated because the required three valid seed "
        "pairs were unavailable."
    ) in markdown

    inventory = json.loads(
        (artifact_dir / "artifact_inventory.json").read_text(encoding="utf-8")
    )
    archived = {row["archived_relative_path"]: row for row in inventory["files"]}
    expected_archives = (
        "audit/partial_runs/free_thinking/default_run_00_debug.jsonl",
        "audit/partial_runs/free_thinking/default_run_00.csv",
        "audit/arms/free_thinking/eval.log",
        "audit/study/console_logs/free_thinking.attempt_01.console.log",
        "audit/matched_prefix_sources/concise/seed_9999_debug.jsonl",
        "audit/matched_prefix_sources/thinking/seed_9999_debug.jsonl",
    )
    for archived_path in expected_archives:
        assert archived[archived_path]["present"] is True
        assert archived[archived_path]["sha256"]
        assert (artifact_dir / archived_path).is_file()
    assert (
        archived["audit/matched_prefix_sources/concise/seed_9999_debug.jsonl"][
            "selected_record_count"
        ]
        == 2
    )
    derived = {row["relative_path"]: row for row in inventory["derived_files"]}
    assert derived["matched_prefix_response.csv"]["present"] is True
    assert derived["matched_prefix_response.csv"]["sha256"]
