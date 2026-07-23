"""Tests for embodied-commander study launch and gate tooling."""

import json

from scripts import run_embodied_commander_study as launcher
from scripts import summarize_embodied_commander_study as summarizer


def test_commands_use_three_clients_and_arm_isolated_cache_keys(tmp_path):
    source = launcher._command("100", "baseline", tmp_path / "source")
    treatment = launcher._command("100", "embodied_commander_broadcast", tmp_path / "treatment")

    assert "experiment=embodied_commander_100" in source
    assert "team.topology=baseline" in source
    assert "team.topology=embodied_commander_broadcast" in treatment
    assert sum(value.startswith("clients.") for value in source) == 3
    assert sum(value.startswith("clients.") for value in treatment) == 3
    assert set(source).isdisjoint({value for value in treatment if "prompt_cache_key" in value})
    config = launcher.compose_experiment("embodied_commander_100")
    manifest = launcher._manifest(config, tmp_path, "100", launcher.ARMS)
    assert manifest["logical_call_cap"] == 630
    assert manifest["commander_plan_call_cap_per_treatment_arm"] == 30


def _episode(seed, *, commander=None):
    return {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "error": None,
        "seed": seed,
        "num_steps": 100,
        "termination_reason": "evaluator_step_cap",
        "episode_return": 1.0,
        "action_parse_rate": 1.0,
        "input_tokens": 100,
        "output_tokens": 20,
        "model_call_count": 300,
        "provider_request_count": 300,
        "transport_error_count": 0,
        "episode_wall_seconds": 30.0,
        "user_info": {"Team/achievement_pct": 0.1},
        **({"squad_commander": commander} if commander else {}),
    }


def test_stage_100_summary_passes_structural_gate(tmp_path):
    root = tmp_path / "study"
    manifest = {
        "stage": "100",
        "arms": ["baseline", "embodied_commander_broadcast"],
        "seeds": [12000],
    }
    root.mkdir()
    (root / launcher.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    commander = {
        "plan_calls": 6,
        "valid_plans": 6,
        "plan_parse_rate": 1.0,
        "active_plan_coverage": 1.0,
        "status_attempts": 90,
        "valid_status": 90,
        "status_parse_rate": 1.0,
        "status_attempt_rate": 0.9,
        "valid_status_coverage": 0.9,
        "assignment_ack_rate": 1.0,
        "accepted_unauthorized_plans": 0,
        "accepted_stale_statuses": 0,
        "hidden_state_leak_guard_violations": 0,
    }
    for arm in manifest["arms"]:
        directory = root / arm / "easy" / "alem" / "default"
        directory.mkdir(parents=True)
        payload = _episode(
            12000,
            commander=commander if arm == "embodied_commander_broadcast" else None,
        )
        (directory / "default_run_00.json").write_text(json.dumps(payload), encoding="utf-8")

    report, summary_path = summarizer.summarize(root, tmp_path / "Results")
    summary = json.loads((root / "commander_study_summary.json").read_text(encoding="utf-8"))
    assert report.is_file()
    assert summary_path.is_file()
    assert summary["preregistered_gate"]["passed"] is True
    assert summary["preregistered_gate"]["manual_semantic_trace_review"]["available"] is False
    treatment = next(
        row for row in summary["episodes"] if row["arm"] == "embodied_commander_broadcast"
    )
    treatment["valid_status_coverage"] = 0.79
    assert summarizer._stage_100_gate(summary["episodes"])["passed"] is False
