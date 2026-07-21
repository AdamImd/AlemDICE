"""Tests for the bodyless-team-leader study launcher and summary."""

import json

from scripts import run_team_leader_study
from scripts.summarize_team_leader_study import summarize


def test_study_dry_run_has_exact_call_cap_without_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_team_leader_study, "_git_status_lines", lambda: ())
    output = tmp_path / "study"
    result = run_team_leader_study.run_study(
        ["--output-root", str(output), "--dry-run"]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Nominal logical-call cap: 1,880" in captured.out
    assert "Maximum concurrent provider calls: 3" in captured.out
    assert not output.exists()


def test_arm_commands_pin_topology_and_isolated_cache_keys(tmp_path):
    commands = {
        arm: run_team_leader_study._command(arm, tmp_path / arm / "easy")
        for arm in run_team_leader_study.ARMS
    }
    for arm, command in commands.items():
        assert f"team.topology={arm}" in command
        cache_args = [value for value in command if "prompt_cache_key=" in value]
        assert len(cache_args) == 4
        arm_code = {"baseline": ":b:", "leader_peer": ":lp:", "leader_no_peer": ":ln:"}[
            arm
        ]
        assert all(arm_code in value for value in cache_args)
        assert all(len(value.split("=", 1)[1] + ":traffic-0") <= 64 for value in cache_args)


def _episode_payload(arm):
    leader = None
    if arm != "baseline":
        leader = {
            "plan_calls": 1,
            "valid_plans": 1,
            "invalid_plans": 0,
            "plan_parse_rate": 1.0,
        }
    return {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "termination_reason": "evaluator_step_cap",
        "seed": 9999,
        "num_steps": 1,
        "episode_return": 1.0,
        "action_parse_rate": 1.0,
        "input_tokens": 100,
        "output_tokens": 10,
        "reasoning_tokens": 5,
        "cached_tokens": 20,
        "model_call_count": 3 + int(arm != "baseline"),
        "provider_request_count": 3 + int(arm != "baseline"),
        "model_latency_seconds": 2.0,
        "user_info": {
            "Team/normal_reward_pct_of_max": 0.01,
            "Team/coord_reward_pct_of_max": 0.02,
            "Team/reward_pct_of_max": 0.015,
        },
        "communication_metrics": {
            "worker_peer": {"emitted_messages": 1, "delivery_bytes": 20}
        },
        "leader": leader,
    }


def test_summary_writes_root_and_publishable_artifacts(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "study_manifest.json").write_text(
        json.dumps({"schema_version": "alem-dice-team-leader-study-v1"}), encoding="utf-8"
    )
    for arm in run_team_leader_study.ARMS:
        episode_dir = root / arm / "easy" / "alem" / "default"
        episode_dir.mkdir(parents=True)
        (episode_dir / "default_run_00.json").write_text(
            json.dumps(_episode_payload(arm)), encoding="utf-8"
        )
    results = tmp_path / "Results"
    report, payload = summarize(root, results)
    assert report.is_file()
    assert payload.is_file()
    assert (root / "episodes.csv").is_file()
    summary = json.loads((root / "study_summary.json").read_text(encoding="utf-8"))
    assert len(summary["arms"]) == 3
    assert "leader_peer_minus_leader_no_peer" in summary["contrasts"]
