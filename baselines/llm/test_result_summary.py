"""Regression tests for local episode aggregation and resumable artifacts."""

import json

from baselines.llm.utils import collect_and_summarize_results, save_summary_stats


def test_summary_aggregates_parse_rate_and_termination_reasons(tmp_path):
    episode_dir = tmp_path / "alem" / "default"
    episode_dir.mkdir(parents=True)
    episodes = (
        {
            "schema_version": "alem-dice-episode-v1",
            "artifact_status": "complete",
            "episode_return": 1.0,
            "num_steps": 2,
            "done": False,
            "termination_reason": "evaluator_step_cap",
            "action_parse_success": 4,
            "action_parse_fail": 1,
            "model_call_count": 7,
            "decision_model_call_count": 6,
            "debrief_model_call_count": 1,
        },
        {
            "schema_version": "alem-dice-episode-v1",
            "artifact_status": "complete",
            "episode_return": 2.0,
            "num_steps": 3,
            "done": True,
            "termination_reason": "environment_terminated",
            "action_parse_success": 5,
            "action_parse_fail": 0,
            "model_call_count": 9,
            "decision_model_call_count": 9,
            "debrief_model_call_count": 0,
        },
    )
    for index, episode in enumerate(episodes):
        (episode_dir / f"default_run_{index:02d}.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )

    summary = collect_and_summarize_results(str(tmp_path))
    task = summary["alem/default"]

    assert task["action_parse_success"] == 9
    assert task["action_parse_fail"] == 1
    assert task["action_parse_rate"] == 0.9
    assert task["model_call_count"] == 16
    assert task["decision_model_call_count"] == 15
    assert task["debrief_model_call_count"] == 1
    assert dict(task["termination_reason_counts"]) == {
        "evaluator_step_cap": 1,
        "environment_terminated": 1,
    }

    save_summary_stats(summary, str(tmp_path))
    saved = json.loads((tmp_path / "summary_stats.json").read_text(encoding="utf-8"))
    assert saved["alem/default"]["action_parse_rate"] == 0.9
    assert saved["alem/default"]["model_call_count"] == 16
    assert saved["alem/default"]["decision_model_call_count"] == 15
    assert saved["alem/default"]["debrief_model_call_count"] == 1
    assert saved["alem/default"]["termination_reason_counts"] == {
        "evaluator_step_cap": 1,
        "environment_terminated": 1,
    }


def test_failed_episode_usage_is_counted_but_reward_metrics_are_excluded(tmp_path):
    episode_dir = tmp_path / "alem" / "default"
    episode_dir.mkdir(parents=True)
    (episode_dir / "default_run_00.json").write_text(
        json.dumps(
            {
                "schema_version": "alem-dice-episode-v1",
                "artifact_status": "failed",
                "error": "provider unavailable",
                "termination_reason": "error",
                "episode_return": 99.0,
                "num_steps": 1,
                "model_call_count": 3,
                "decision_model_call_count": 3,
                "input_tokens": 1200,
                "output_tokens": 30,
                "action_parse_success": 3,
                "action_parse_fail": 0,
            }
        ),
        encoding="utf-8",
    )

    task = collect_and_summarize_results(str(tmp_path))["alem/default"]

    assert len(task["failed_episodes"]) == 1
    assert task["model_call_count"] == 3
    assert task["input_tokens"] == 1200
    assert task["action_parse_rate"] == 1.0
    assert task["termination_reason_counts"]["error"] == 1
    assert task["total_reward"] == 0.0
    assert task["total_steps"] == 0


def test_attempt_ledger_preserves_failed_usage_without_double_counting(tmp_path):
    episode_dir = tmp_path / "alem" / "default"
    episode_dir.mkdir(parents=True)
    completed = {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "termination_reason": "evaluator_step_cap",
        "episode_return": 2.0,
        "num_steps": 2,
        "model_call_count": 6,
        "provider_request_count": 6,
        "decision_model_call_count": 6,
        "input_tokens": 1800,
        "action_parse_success": 6,
        "action_parse_fail": 0,
    }
    (episode_dir / "default_run_00.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    failed_attempt = {
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 0,
        "artifact_status": "failed",
        "termination_reason": "error",
        "model_call_count": 3,
        "provider_request_count": 4,
        "transport_error_count": 1,
        "decision_model_call_count": 3,
        "input_tokens": 900,
        "action_parse_success": 3,
        "action_parse_fail": 0,
    }
    completed_attempt = {
        **completed,
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 0,
    }
    (episode_dir / "attempt_ledger.jsonl").write_text(
        "\n".join((json.dumps(failed_attempt), json.dumps(completed_attempt))) + "\n",
        encoding="utf-8",
    )
    archive = episode_dir / "attempt_archive" / "episode_00" / "old"
    archive.mkdir(parents=True)
    (archive / "default_run_00.json").write_text(
        json.dumps({**completed, "episode_return": 99.0}), encoding="utf-8"
    )

    task = collect_and_summarize_results(str(tmp_path))["alem/default"]

    assert len(task["episodes"]) == 1
    assert task["attempt_count"] == 2
    assert task["failed_attempt_count"] == 1
    assert task["model_call_count"] == 9
    assert task["provider_request_count"] == 10
    assert task["transport_error_count"] == 1
    assert task["input_tokens"] == 2700
    assert task["total_reward"] == 2.0


def test_partial_ledger_falls_back_only_for_uncovered_episode(tmp_path):
    episode_dir = tmp_path / "alem" / "default"
    episode_dir.mkdir(parents=True)
    for episode_index, input_tokens in ((0, 100), (1, 200)):
        episode = {
            "schema_version": "alem-dice-episode-v1",
            "artifact_status": "complete",
            "termination_reason": "evaluator_step_cap",
            "episode_return": 1.0,
            "num_steps": 1,
            "model_call_count": 3,
            "provider_request_count": 3,
            "input_tokens": input_tokens,
            "action_parse_success": 3,
            "action_parse_fail": 0,
        }
        (episode_dir / f"default_run_{episode_index:02d}.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )

    covered_attempt = {
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 1,
        "artifact_status": "complete",
        "termination_reason": "evaluator_step_cap",
        "model_call_count": 3,
        "provider_request_count": 3,
        "input_tokens": 200,
        "action_parse_success": 3,
        "action_parse_fail": 0,
    }
    (episode_dir / "attempt_ledger.jsonl").write_text(
        json.dumps(covered_attempt) + "\n", encoding="utf-8"
    )

    task = collect_and_summarize_results(str(tmp_path))["alem/default"]

    assert len(task["episodes"]) == 2
    assert task["attempt_count"] == 2
    assert task["model_call_count"] == 6
    assert task["provider_request_count"] == 6
    assert task["input_tokens"] == 300
    assert task["action_parse_success"] == 6
    assert task["total_reward"] == 2.0


def test_empty_or_malformed_ledger_does_not_suppress_fallback(tmp_path):
    ledger_contents = (
        "",
        "{not-json}\n",
        json.dumps({"schema_version": "alem-dice-attempt-v1"}) + "\n",
        "[]\n",
    )
    for case_index, contents in enumerate(ledger_contents):
        root = tmp_path / f"case-{case_index}"
        episode_dir = root / "alem" / "default"
        episode_dir.mkdir(parents=True)
        episode = {
            "schema_version": "alem-dice-episode-v1",
            "artifact_status": "complete",
            "termination_reason": "evaluator_step_cap",
            "episode_return": 1.0,
            "num_steps": 1,
            "model_call_count": 3,
            "provider_request_count": 4,
            "input_tokens": 100,
            "action_parse_success": 3,
            "action_parse_fail": 0,
        }
        (episode_dir / "default_run_00.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
        (episode_dir / "attempt_ledger.jsonl").write_text(contents, encoding="utf-8")

        task = collect_and_summarize_results(str(root))["alem/default"]

        assert task["attempt_count"] == 1
        assert task["model_call_count"] == 3
        assert task["provider_request_count"] == 4
        assert task["input_tokens"] == 100


def test_attempt_id_requires_exact_ledger_match(tmp_path):
    episode_dir = tmp_path / "alem" / "default"
    episode_dir.mkdir(parents=True)
    completed = {
        "schema_version": "alem-dice-episode-v1",
        "attempt_id": "new-attempt",
        "artifact_status": "complete",
        "termination_reason": "evaluator_step_cap",
        "episode_return": 1.0,
        "num_steps": 1,
        "model_call_count": 3,
        "provider_request_count": 3,
        "input_tokens": 200,
        "action_parse_success": 3,
        "action_parse_fail": 0,
    }
    (episode_dir / "default_run_00.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    stale_attempt = {
        "schema_version": "alem-dice-attempt-v1",
        "attempt_id": "old-attempt",
        "episode_index": 0,
        "artifact_status": "complete",
        "termination_reason": "evaluator_step_cap",
        "model_call_count": 2,
        "provider_request_count": 2,
        "input_tokens": 100,
        "action_parse_success": 2,
        "action_parse_fail": 0,
    }
    (episode_dir / "attempt_ledger.jsonl").write_text(
        json.dumps(stale_attempt) + "\n", encoding="utf-8"
    )

    task = collect_and_summarize_results(str(tmp_path))["alem/default"]

    assert task["attempt_count"] == 2
    assert task["model_call_count"] == 5
    assert task["provider_request_count"] == 5
    assert task["input_tokens"] == 300


def test_matching_attempt_id_suppresses_result_fallback(tmp_path):
    episode_dir = tmp_path / "alem" / "default"
    episode_dir.mkdir(parents=True)
    completed = {
        "schema_version": "alem-dice-episode-v1",
        "attempt_id": "same-attempt",
        "artifact_status": "complete",
        "termination_reason": "evaluator_step_cap",
        "episode_return": 1.0,
        "num_steps": 1,
        "model_call_count": 3,
        "provider_request_count": 3,
        "input_tokens": 200,
        "action_parse_success": 3,
        "action_parse_fail": 0,
    }
    (episode_dir / "default_run_00.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    matching_attempt = {
        **completed,
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 0,
    }
    (episode_dir / "attempt_ledger.jsonl").write_text(
        json.dumps(matching_attempt) + "\n", encoding="utf-8"
    )

    task = collect_and_summarize_results(str(tmp_path))["alem/default"]

    assert task["attempt_count"] == 1
    assert task["model_call_count"] == 3
    assert task["provider_request_count"] == 3
    assert task["input_tokens"] == 200


def test_legacy_coverage_requires_matching_episode_status(tmp_path):
    episode_dir = tmp_path / "alem" / "default"
    episode_dir.mkdir(parents=True)
    completed = {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "termination_reason": "evaluator_step_cap",
        "episode_return": 1.0,
        "num_steps": 1,
        "model_call_count": 3,
        "provider_request_count": 3,
        "input_tokens": 200,
        "action_parse_success": 3,
        "action_parse_fail": 0,
    }
    (episode_dir / "default_run_00.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    failed_attempt = {
        "schema_version": "alem-dice-attempt-v1",
        "episode_index": 0,
        "artifact_status": "failed",
        "termination_reason": "error",
        "model_call_count": 2,
        "provider_request_count": 2,
        "input_tokens": 100,
        "action_parse_success": 2,
        "action_parse_fail": 0,
    }
    (episode_dir / "attempt_ledger.jsonl").write_text(
        json.dumps(failed_attempt) + "\n", encoding="utf-8"
    )

    task = collect_and_summarize_results(str(tmp_path))["alem/default"]

    assert task["attempt_count"] == 2
    assert task["failed_attempt_count"] == 1
    assert task["model_call_count"] == 5
    assert task["input_tokens"] == 300
