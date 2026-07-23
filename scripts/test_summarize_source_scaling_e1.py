"""Focused synthetic checks for the E1 analysis contract."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.summarize_source_scaling_e1 import (
    CANONICAL_POPULATIONS,
    CANONICAL_SEEDS,
    CANONICAL_TREATMENT,
    MANIFEST_SCHEMA,
    TURN_ACCOUNTING_SCHEMA,
    _debug_noop_metrics,
    _episode_noop_metrics,
    _validate_episode_treatment,
    bootstrap_mean_ci,
    discover_rows,
    episode_row,
    paired_population_contrasts,
    validate_manifest_contract,
    validate_manifest_grid,
    validate_population_run_binding,
)


def _episode_payload(*, seed=7):
    return {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "termination_reason": "environment_truncated",
        "physical_worker_count": 2,
        "logical_participant_count": 2,
        "seed": seed,
        "num_steps": 10,
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
        "active_action_validation_fallback_noop_count": 0,
        "active_residual_effective_noop_count": 0,
        "inactive_submitted_turn_count": 10,
        "inactive_effective_noop_count": 10,
        "canonical_submitted_noop_count": 13,
        "effective_environment_noop_count": 13,
        "executed_noop_count": 13,
        "action_parse_success": 9,
        "action_parse_fail": 1,
        "action_parse_skipped_inactive": 10,
        "action_parse_rate": 0.9,
        "turn_accounting_schema_version": TURN_ACCOUNTING_SCHEMA,
        "turn_accounting_complete": True,
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


def _grid_manifest():
    return {
        "schema_version": MANIFEST_SCHEMA,
        "profile": CANONICAL_TREATMENT["profile"],
        "planned_counts": list(CANONICAL_POPULATIONS),
        "seeds": list(CANONICAL_SEEDS),
        "episodes_per_count": len(CANONICAL_SEEDS),
    }


def _strict_manifest(root):
    file_hashes = {}
    normalized_hashes = {}
    cache_keys = {}
    for population in CANONICAL_POPULATIONS:
        config = root / f"n{population}" / "resolved_config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(f"alem:\n  num_agents: {population}\n", encoding="utf-8")
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        file_hashes[str(population)] = digest
        normalized_hashes[str(population)] = "a" * 64
        cache_keys[str(population)] = [
            f"alem:e1:g54n:n{population}:a{worker_id}" for worker_id in range(population)
        ]
    return {
        **_grid_manifest(),
        **CANONICAL_TREATMENT,
        "stage_definitions": {
            "all": list(CANONICAL_POPULATIONS),
            "e1a": [1, 2, 3, 4],
            "e1b": [6],
        },
        "output_root": str(root.resolve()),
        "source_commit": "b" * 40,
        "uv_lock_sha256": "c" * 64,
        "resolved_config_sha256": normalized_hashes,
        "resolved_config_file_sha256": file_hashes,
        "cache_keys": cache_keys,
    }


def _strip_versioned_turn_accounting(payload):
    for field in (
        "turn_accounting_schema_version",
        "turn_accounting_complete",
        "action_parse_success",
        "action_parse_fail",
        "action_parse_skipped_inactive",
        "intentional_actionable_noop_count",
        "parse_fallback_noop_count",
        "active_action_validation_fallback_noop_count",
        "active_residual_effective_noop_count",
        "inactive_submitted_turn_count",
        "inactive_effective_noop_count",
        "canonical_submitted_noop_count",
        "effective_environment_noop_count",
        "executed_noop_count",
    ):
        payload.pop(field, None)


def _treatment_payload(manifest):
    payload = _episode_payload(seed=CANONICAL_SEEDS[0])
    payload.update(
        {
            "task": "default",
            "team_topology": "baseline",
            "coordination_strategy": "free",
            "team": {"topology": "baseline"},
            "agent": {
                "type": "robust_all",
                "prompt_mode": "specific_collaborative",
            },
            "model_call_count": 2,
            "model_usage_records": [
                {
                    "participant_id": worker_id,
                    "phase": "decision",
                    "model_id": "gpt-5.4-nano-2026-03-17",
                }
                for worker_id in range(2)
            ],
            "clients": [
                {
                    "client_name": "openai_responses",
                    "model_id": "gpt-5.4-nano",
                    "generate_kwargs": {
                        "reasoning_effort": "high",
                        "prompt_cache_key": manifest["cache_keys"]["2"][worker_id],
                    },
                    "prompt_cache_key_resolved": (
                        f"{manifest['cache_keys']['2'][worker_id]}:traffic-0"
                    ),
                    "prompt_cache_traffic_shard_resolved": 0,
                }
                for worker_id in range(2)
            ],
        }
    )
    return payload


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
    assert row["canonical_submitted_noops"] == 13
    assert row["effective_environment_noops"] == 13
    assert row["executed_noops"] == row["canonical_submitted_noops"]
    assert row["survival_fraction"] == row["alive_turn_fraction"]
    assert row["input_tokens_per_agent_turn"] == row["input_tokens_per_submitted_turn"]


def test_episode_row_rejects_inconsistent_performance_exposure(tmp_path):
    root = tmp_path
    path = root / "n2" / "easy" / "alem" / "default" / "default_run_00.json"
    path.parent.mkdir(parents=True)
    payload = _episode_payload()
    payload["performance_metrics"]["exposure"]["agent_turns_submitted"] = 19
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="physical-worker exposure"):
        episode_row(path, root, requested_environment_steps=10)


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
    assert metrics["canonical_submitted_noops"] == 3
    assert metrics["effective_environment_noops"] == 3
    assert metrics["executed_noops"] == 3
    assert metrics["action_parse_success"] == 2
    assert metrics["action_parse_fail"] == 1
    assert metrics["action_parse_skipped_inactive"] == 1
    assert metrics["action_parse_rate"] == pytest.approx(2 / 3)


def test_inactive_non_noop_is_effective_but_not_canonical_noop(tmp_path):
    episode = tmp_path / "default_run_00.json"
    debug = tmp_path / "default_run_00_debug.jsonl"
    records = [
        {
            "step": 0,
            "agents": {
                "0": {
                    "llm_raw_output": "<action>Move North</action>",
                    "parsed_action": "Move North",
                    "stop_reason": "stop",
                }
            },
            "action_parse_stats": {"0": {"success": 0, "fail": 0, "skipped_inactive": 1}},
        },
        {
            "step": 1,
            "agents": {
                "0": {
                    "llm_raw_output": "<action>Move East</action>",
                    "parsed_action": "Move East",
                    "stop_reason": "stop",
                }
            },
            "action_parse_stats": {"0": {"success": 0, "fail": 0, "skipped_inactive": 2}},
        },
    ]
    debug.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    metrics = _debug_noop_metrics(
        episode,
        num_agents=1,
        expected_submitted_turns=2,
    )
    assert metrics["canonical_submitted_noops"] == 0
    assert metrics["effective_environment_noops"] == 1
    assert metrics["inactive_effective_noops"] == 1
    assert metrics["executed_noops"] == 0


def test_active_action_validation_fallback_has_own_noop_cause(tmp_path):
    episode = tmp_path / "default_run_00.json"
    debug = tmp_path / "default_run_00_debug.jsonl"
    debug.write_text(
        json.dumps(
            {
                "step": 0,
                "agents": {
                    "0": {
                        "llm_raw_output": "<action>Give to Agent 0</action>",
                        "parsed_action": "Noop",
                        "stop_reason": "stop",
                    }
                },
                "action_parse_stats": {"0": {"success": 1, "fail": 0, "skipped_inactive": 0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = _debug_noop_metrics(
        episode,
        num_agents=1,
        expected_submitted_turns=1,
    )
    assert metrics["active_action_validation_fallback_noops"] == 1
    assert metrics["parse_fallback_noops"] == 0
    assert metrics["effective_environment_noops"] == 1


def test_complete_legacy_debug_overrides_biased_episode_parse_rate(tmp_path):
    root = tmp_path
    path = root / "n1" / "easy" / "alem" / "default" / "default_run_00.json"
    path.parent.mkdir(parents=True)
    payload = _episode_payload()
    _strip_versioned_turn_accounting(payload)
    payload.update(
        {
            "physical_worker_count": 1,
            "logical_participant_count": 1,
            "num_steps": 2,
            "action_parse_rate": 0.5,
            "action_frequency": {"Noop": 2},
        }
    )
    payload["performance_metrics"]["exposure"] = {
        "environment_steps_completed": 2,
        "completed_agent_turn_capacity": 2,
        "agent_turns_submitted": 2,
        "classified_action_turns": 2,
        "alive_agent_turns": 1,
        "actionable_agent_turns": 1,
        "survival_fraction": 0.5,
        "actionable_fraction": 0.5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    debug = path.with_name("default_run_00_debug.jsonl")
    debug.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "step": 0,
                    "agents": {
                        "0": {
                            "llm_raw_output": "unparseable prose",
                            "parsed_action": "Noop",
                            "stop_reason": "stop",
                        }
                    },
                    "action_parse_stats": {"0": {"success": 0, "fail": 0, "skipped_inactive": 1}},
                },
                {
                    "step": 1,
                    "agents": {
                        "0": {
                            "llm_raw_output": "<action>Noop</action>",
                            "parsed_action": "Noop",
                            "stop_reason": "stop",
                        }
                    },
                    "action_parse_stats": {"0": {"success": 0, "fail": 0, "skipped_inactive": 2}},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    row = episode_row(path, root, requested_environment_steps=2)
    assert row["legacy_recorded_action_parse_rate"] == 0.5
    assert row["action_parse_success"] == 0
    assert row["action_parse_fail"] == 1
    assert row["action_parse_skipped_inactive"] == 1
    assert row["action_parse_rate"] == 0


def test_missing_debug_preserves_only_recoverable_aggregate_noop_fields(tmp_path):
    payload = _episode_payload()
    _strip_versioned_turn_accounting(payload)

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
    assert metrics["canonical_submitted_noops"] == 13
    assert metrics["effective_environment_noops"] is None
    assert metrics["executed_noops"] == 13
    assert metrics["action_parse_rate"] is None


@pytest.mark.parametrize("allow_incomplete", (False, True))
def test_manifest_grid_requires_explicit_incomplete_watermark(allow_incomplete):
    manifest = _grid_manifest()
    rows = [
        {
            "num_agents": population,
            "seed": seed,
            "episode_index": seed_index,
        }
        for population in CANONICAL_POPULATIONS
        for seed_index, seed in enumerate(CANONICAL_SEEDS)
        if (population, seed) != (6, CANONICAL_SEEDS[-1])
    ]
    if not allow_incomplete:
        with pytest.raises(ValueError, match="--allow-incomplete"):
            validate_manifest_grid(rows, manifest, allow_incomplete=False)
        return

    audit = validate_manifest_grid(rows, manifest, allow_incomplete=True)
    assert audit["complete"] is False
    assert audit["missing_pairs"] == [{"num_agents": 6, "seed": CANONICAL_SEEDS[-1]}]
    assert audit["watermark"].startswith("INCOMPLETE INTERIM ANALYSIS")


@pytest.mark.parametrize("allow_incomplete", (False, True))
def test_absent_manifest_is_never_relaxed(allow_incomplete):
    rows = [{"num_agents": 1, "seed": CANONICAL_SEEDS[0], "episode_index": 0}]
    with pytest.raises(ValueError, match="relaxes only missing cells"):
        validate_manifest_grid(rows, None, allow_incomplete=allow_incomplete)


def test_manifest_profile_and_episodes_per_count_fail_closed():
    manifest = _grid_manifest()
    manifest["profile"] = "wrong-profile"
    with pytest.raises(ValueError, match="wrong E1 profile"):
        validate_manifest_grid([], manifest, allow_incomplete=True)

    manifest = _grid_manifest()
    manifest["episodes_per_count"] = "three"
    with pytest.raises(ValueError, match="episodes_per_count"):
        validate_manifest_grid([], manifest, allow_incomplete=True)


def test_manifest_contract_binds_study_config_hashes(tmp_path):
    manifest = _strict_manifest(tmp_path)
    binding = validate_manifest_contract(tmp_path, manifest)
    assert binding["source_commit"] == "b" * 40

    (tmp_path / "n2" / "resolved_config.yaml").write_text(
        "alem:\n  num_agents: 99\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Study config hash mismatch"):
        validate_manifest_contract(tmp_path, manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["clients"][0].update(model_id="wrong-model"), "wrong client"),
        (lambda payload: payload["team"].update(topology="leader_peer"), "contamination"),
    ),
)
def test_episode_treatment_rejects_wrong_model_or_topology(tmp_path, mutation, message):
    manifest = _strict_manifest(tmp_path)
    payload = _treatment_payload(manifest)
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        _validate_episode_treatment(
            payload,
            tmp_path / "episode.json",
            manifest,
            num_agents=2,
        )


def test_population_run_manifest_and_runtime_config_are_bound(tmp_path):
    manifest = _strict_manifest(tmp_path)
    arm = tmp_path / "n2" / "easy"
    arm.mkdir(parents=True)
    runtime_config = arm / "resolved_config.yaml"
    runtime_config.write_text("alem:\n  num_agents: 2\n", encoding="utf-8")
    runtime_hash = hashlib.sha256(runtime_config.read_bytes()).hexdigest()
    run_manifest = {
        "schema_version": "alem-dice-run-manifest-v1",
        "profile": CANONICAL_TREATMENT["profile"],
        "difficulty": "easy",
        "source_commit": manifest["source_commit"],
        "uv_lock_sha256": manifest["uv_lock_sha256"],
        "resolved_config_file": "resolved_config.yaml",
        "resolved_config_sha256": runtime_hash,
        "models": ["gpt-5.4-nano"] * 6,
    }
    (arm / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    validate_population_run_binding(tmp_path, manifest, 2)

    run_manifest["profile"] = "contaminated"
    (arm / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="profile treatment binding mismatch"):
        validate_population_run_binding(tmp_path, manifest, 2)


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


def test_bootstrap_rejects_one_unstable_replication():
    with pytest.raises(ValueError, match="at least 100"):
        bootstrap_mean_ci([1.0, 2.0], reps=1, rng=np.random.default_rng(1))


def test_summarizer_import_is_cache_absent_and_side_effect_free(tmp_path):
    isolated_home = tmp_path / "home"
    isolated_cache = tmp_path / "cache"
    isolated_cwd = tmp_path / "cwd"
    for path in (isolated_home, isolated_cache, isolated_cwd):
        path.mkdir()
    env = {
        **os.environ,
        "HOME": str(isolated_home),
        "XDG_CACHE_HOME": str(isolated_cache),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.summarize_source_scaling_e1; "
                "assert 'alem' not in sys.modules; "
                "assert not any(name.startswith('alem.alem_coop') for name in sys.modules)"
            ),
        ],
        cwd=isolated_cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Loading textures" not in result.stdout + result.stderr
    assert not any(any(path.iterdir()) for path in (isolated_home, isolated_cache, isolated_cwd))
