"""Focused synthetic checks for the E1 analysis contract."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from alem_turn_accounting import (
    TURN_ACCOUNTING_AGENT_SUFFIXES,
    TURN_ACCOUNTING_FEATURES,
    TURN_ACCOUNTING_SEMANTICS,
)
from baselines.llm.eval_utils.evaluator import _attempt_ledger_guard
from baselines.llm.eval_utils.performance_metrics import build_performance_metrics
from scripts import run_source_scaling_study as source_scaling_launcher
from scripts.summarize_source_scaling_e1 import (
    CANONICAL_CLIENT_SLOTS,
    CANONICAL_POPULATIONS,
    CANONICAL_SEEDS,
    CANONICAL_TREATMENT,
    MANIFEST_SCHEMA,
    TRUSTED_PREFLIGHT_SHA256,
    TRUSTED_RESOLVED_MODEL,
    TRUSTED_SOURCE_COMMIT,
    TRUSTED_UV_LOCK_SHA256,
    TURN_ACCOUNTING_SCHEMA,
    _canonical_source_config_payload,
    _debug_noop_metrics,
    _episode_noop_metrics,
    _read_manifest,
    _study_config_semantics_sha256,
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
    payload = {
        "schema_version": "alem-dice-episode-v1",
        "artifact_status": "complete",
        "termination_reason": "environment_truncated",
        "physical_worker_count": 2,
        "logical_participant_count": 2,
        "seed": seed,
        "num_steps": 10,
        "episode_return": 2.5,
        "agent_0_return": 2.0,
        "agent_1_return": 3.0,
        "input_tokens": 100,
        "cached_tokens": 40,
        "output_tokens": 20,
        "reasoning_tokens": 10,
        "cache_write_tokens": 0,
        "decision_input_tokens": 100,
        "decision_cached_tokens": 40,
        "decision_output_tokens": 20,
        "decision_reasoning_tokens": 10,
        "decision_cache_write_tokens": 0,
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
        "resolved_model_id": TRUSTED_RESOLVED_MODEL,
        "resolved_model_ids": [TRUSTED_RESOLVED_MODEL],
        "environment_steps_completed": 10,
        "agent_turns_submitted": 20,
        "alive_agent_turns": 12,
        "actionable_agent_turns": 10,
        "action_parse_success": 9,
        "action_parse_fail": 1,
        "action_parse_skipped_inactive": 10,
        "action_parse_rate": 0.9,
        "turn_accounting_schema_version": TURN_ACCOUNTING_SCHEMA,
        "turn_accounting_features": list(TURN_ACCOUNTING_FEATURES),
        "turn_accounting_semantics": dict(TURN_ACCOUNTING_SEMANTICS),
        "turn_accounting_provenance": "evaluator_exact_pre_step",
        "turn_accounting_complete": True,
        "action_frequency": {"Noop": 13, "Do": 7},
        "communication_metrics": {"worker_peer": {"delivery_bytes": 50}},
        "user_info": {
            "Team/normal_reward_pct_of_max": 0.02,
            "Team/coord_reward_pct_of_max": 0.04,
            "Team/reward_pct_of_max": 0.03,
            "Team/normal_achievement_pct": 0.05,
            "Team/coordination_achievement_pct": 0.06,
            "Team/achievement_pct": 0.055,
            "Team/normal_achievements": 2,
            "Team/coordination_achievements": 1,
            "Team/total_achievements": 3,
            "Agent0/normal_achievements": 2,
            "Agent0/coordination_achievements": 1,
            "Agent0/total_achievements": 3,
            "Agent1/normal_achievements": 1,
            "Agent1/coordination_achievements": 0,
            "Agent1/total_achievements": 1,
            "Coordination/total_attempts": 0,
            "Coordination/total_resolved_attempts": 0,
            "Coordination/total_successes": 0,
            "Cooperation/give_attempt_count": 0,
            "Cooperation/trade_count": 0,
            "Cooperation/request_count": 0,
            "Cooperation/revives": 0,
            "Cooperation/alive_agent_steps": 12,
            "Cooperation/actionable_agent_steps": 10,
        },
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
    worker_values = {
        0: {
            "parse_success": 9,
            "parse_fail": 1,
            "parse_skipped_inactive": 0,
            "intentional_actionable_noop_count": 2,
            "parse_fallback_noop_count": 1,
            "active_action_validation_fallback_noop_count": 0,
            "active_residual_effective_noop_count": 0,
            "inactive_submitted_turn_count": 0,
            "inactive_effective_noop_count": 0,
            "canonical_submitted_noop_count": 3,
            "effective_environment_noop_count": 3,
            "executed_noop_count": 3,
        },
        1: {
            "parse_success": 0,
            "parse_fail": 0,
            "parse_skipped_inactive": 10,
            "intentional_actionable_noop_count": 0,
            "parse_fallback_noop_count": 0,
            "active_action_validation_fallback_noop_count": 0,
            "active_residual_effective_noop_count": 0,
            "inactive_submitted_turn_count": 10,
            "inactive_effective_noop_count": 10,
            "canonical_submitted_noop_count": 10,
            "effective_environment_noop_count": 10,
            "executed_noop_count": 10,
        },
    }
    for worker_id, values in worker_values.items():
        for suffix, value in values.items():
            payload[f"agent_{worker_id}_{suffix}"] = value
        payload[f"agent_{worker_id}_input_tokens"] = 50
        payload[f"agent_{worker_id}_cached_tokens"] = 20
        payload[f"agent_{worker_id}_output_tokens"] = 10
        payload[f"agent_{worker_id}_reasoning_tokens"] = 5
        payload[f"agent_{worker_id}_cache_write_tokens"] = 0
        payload[f"agent_{worker_id}_resolved_model_id"] = TRUSTED_RESOLVED_MODEL
        payload[f"agent_{worker_id}_resolved_model_ids"] = [TRUSTED_RESOLVED_MODEL]
    payload["performance_metrics"] = build_performance_metrics(
        payload,
        2,
        environment_steps_completed=10,
        agent_turns_submitted=20,
        alive_agent_turns=12,
        actionable_agent_turns=10,
    )
    return payload


def _grid_manifest():
    return {
        "schema_version": MANIFEST_SCHEMA,
        "profile": CANONICAL_TREATMENT["profile"],
        "planned_counts": list(CANONICAL_POPULATIONS),
        "seeds": list(CANONICAL_SEEDS),
        "episodes_per_count": len(CANONICAL_SEEDS),
    }


def _canonical_preflight_payload():
    return {
        "attempt": 1,
        "cached_tokens": 0,
        "completed_at_utc": "2026-07-23T08:52:29.225479+00:00",
        "completion": "<action>Noop</action>",
        "configured_prompt_cache_key": "alem:e1:g54n:preflight",
        "effective_prompt_cache_key": "alem:e1:g54n:preflight:traffic-0",
        "incomplete_reason": None,
        "input_tokens": 35,
        "latency_seconds": 2.993951339041814,
        "logical_response_count": 1,
        "model_id": TRUSTED_RESOLVED_MODEL,
        "output_tokens": 31,
        "parse_success": True,
        "parsed_action": "Noop",
        "prompt_cache_traffic_shard": 0,
        "provider_status": "completed",
        "reasoning_effort": "high",
        "reasoning_tokens": 18,
        "requested_model": "gpt-5.4-nano",
        "resolved_config_sha256": (
            "cb0be6dcb1c811eb573e80521605fd4ba609fb79a834a0be29049d86d7d48868"
        ),
        "response_id": "resp_0e417cee7cbc0ae0016a61d64aa5ac819a91d0baaeb9f094e4",
        "schema_version": "alem-dice-source-scaling-preflight-v1",
        "source_commit": TRUSTED_SOURCE_COMMIT,
        "started_at_utc": "2026-07-23T08:52:26.215589+00:00",
        "status": "passed",
        "stop_reason": "stop",
        "transport_attempt_count": 1,
        "transport_error_count": 0,
        "transport_error_types": [],
        "uv_lock_sha256": TRUSTED_UV_LOCK_SHA256,
    }


def _strict_manifest(root):
    file_hashes = {}
    normalized_hashes = {}
    cache_keys = {}
    for population in CANONICAL_POPULATIONS:
        config = root / f"n{population}" / "resolved_config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(
            OmegaConf.to_yaml(
                OmegaConf.create(_canonical_source_config_payload(population)),
                resolve=True,
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        file_hashes[str(population)] = digest
        normalized_hashes[str(population)] = _study_config_semantics_sha256(config)
        cache_keys[str(population)] = [
            f"alem:e1:g54n:n{population}:a{worker_id}" for worker_id in range(population)
        ]
    preflight = root / "preflight.json"
    preflight.write_text(
        json.dumps(_canonical_preflight_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight_hash = hashlib.sha256(preflight.read_bytes()).hexdigest()
    assert preflight_hash == TRUSTED_PREFLIGHT_SHA256
    return {
        **_grid_manifest(),
        **CANONICAL_TREATMENT,
        "stage_definitions": {
            "all": list(CANONICAL_POPULATIONS),
            "e1a": [1, 2, 3, 4],
            "e1b": [6],
        },
        "output_root": str(root.resolve()),
        "source_commit": TRUSTED_SOURCE_COMMIT,
        "uv_lock_sha256": TRUSTED_UV_LOCK_SHA256,
        "preflight_sha256": preflight_hash,
        "resolved_config_sha256": normalized_hashes,
        "resolved_config_file_sha256": file_hashes,
        "cache_keys": cache_keys,
    }


def _strip_versioned_turn_accounting(payload):
    for field in (
        "turn_accounting_schema_version",
        "turn_accounting_features",
        "turn_accounting_semantics",
        "turn_accounting_provenance",
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
    for worker_id in range(2):
        for suffix in TURN_ACCOUNTING_AGENT_SUFFIXES.values():
            payload.pop(f"agent_{worker_id}_{suffix}", None)


def _treatment_payload(manifest):
    payload = _episode_payload(seed=CANONICAL_SEEDS[0])
    trusted = _canonical_source_config_payload(2)
    payload.update(
        {
            "task": "default",
            "team_topology": "baseline",
            "coordination_strategy": "free",
            "team": trusted["team"],
            "agent": trusted["agent"],
            "model_call_count": 20,
            "decision_model_call_count": 20,
            "decision_provider_request_count": 20,
            "transport_error_count": 0,
            "transport_error_reasons": {},
            "debrief_model_call_count": 0,
            "commander_plan_model_call_count": 0,
            "agent_0_model_call_count": 10,
            "agent_1_model_call_count": 10,
            "agent_0_provider_request_count": 10,
            "agent_1_provider_request_count": 10,
            "agent_0_transport_error_count": 0,
            "agent_1_transport_error_count": 0,
            "model_usage_records": [
                {
                    "participant_id": worker_id,
                    "phase": "decision",
                    "model_id": TRUSTED_RESOLVED_MODEL,
                    "response_id": f"response-{worker_id}-{call_index}",
                    "input_tokens": 5,
                    "cached_tokens": 2,
                    "output_tokens": 1,
                    "reasoning_tokens": 1 if call_index < 5 else 0,
                    "cache_write_tokens": 0,
                }
                for worker_id in range(2)
                for call_index in range(10)
            ],
            "clients": [
                {
                    **trusted["clients"][worker_id],
                    "enable_thinking_resolved": False,
                    "model_id_resolved": (TRUSTED_RESOLVED_MODEL if worker_id < 2 else None),
                    "model_ids_resolved": ([TRUSTED_RESOLVED_MODEL] if worker_id < 2 else []),
                    "prompt_cache_key_resolved": (
                        f"alem:e1:g54n:n2:a{worker_id}:traffic-0" if worker_id < 2 else None
                    ),
                    "prompt_cache_traffic_shard_resolved": 0 if worker_id < 2 else None,
                }
                for worker_id in range(CANONICAL_CLIENT_SLOTS)
            ],
        }
    )
    return payload


def _write_population_run_binding(root, manifest, *, population=2, resume_count=0):
    arm = root / f"n{population}" / "easy"
    arm.mkdir(parents=True, exist_ok=True)

    def write_runtime(name, run_id):
        path = arm / name
        payload = _canonical_source_config_payload(population, arm)
        payload.setdefault("wandb", {})["run_id"] = run_id
        path.write_text(
            OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True),
            encoding="utf-8",
        )
        return path

    initial = write_runtime("resolved_config.yaml", "initial")
    resume_history = []
    for index in range(resume_count):
        name = f"resolved_config.resume-20260723T01010{index}123456.yaml"
        config = write_runtime(name, f"resume-{index}")
        resume_history.append(
            {
                "source_commit": manifest["source_commit"],
                "uv_lock_sha256": manifest["uv_lock_sha256"],
                "resolved_config_file": name,
                "resolved_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
        )
    run_manifest = {
        "schema_version": "alem-dice-run-manifest-v1",
        "profile": CANONICAL_TREATMENT["profile"],
        "difficulty": "easy",
        "source_commit": manifest["source_commit"],
        "uv_lock_sha256": manifest["uv_lock_sha256"],
        "resolved_config_file": initial.name,
        "resolved_config_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
        "models": ["gpt-5.4-nano"] * CANONICAL_CLIENT_SLOTS,
        "resume_history": resume_history,
    }
    (arm / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    return arm, run_manifest


def _write_exact_v2_debug(path, *, all_noops=False):
    records = []
    for step in range(10):
        agents = {}
        for worker_id in range(2):
            inactive = worker_id == 1
            if all_noops:
                parse_classification = "skipped_inactive" if inactive else "success"
                intentional = not inactive
                parse_fallback = False
                canonical_noop = True
            else:
                parse_classification = (
                    "skipped_inactive" if inactive else ("failure" if step == 2 else "success")
                )
                intentional = not inactive and step < 2
                parse_fallback = not inactive and step == 2
                canonical_noop = inactive or step <= 2
            classification = {
                "pre_step_inactive": inactive,
                "submitted_action": "Noop" if canonical_noop else "Do",
                "canonical_submitted_action": "Noop" if canonical_noop else "Do",
                "parse_classification": parse_classification,
                "intentional_actionable_noop": intentional,
                "parse_fallback_noop": parse_fallback,
                "active_action_validation_fallback_noop": False,
                "active_residual_effective_noop": False,
                "inactive_submitted_turn": inactive,
                "inactive_effective_noop": inactive,
                "canonical_submitted_noop": canonical_noop,
                "effective_environment_noop": inactive or canonical_noop,
                "executed_noop": canonical_noop,
            }
            agents[str(worker_id)] = {"action_turn_classification": classification}
        records.append(
            {
                "step": step,
                "turn_accounting_schema_version": TURN_ACCOUNTING_SCHEMA,
                "turn_accounting_features": list(TURN_ACCOUNTING_FEATURES),
                "turn_accounting_semantics": TURN_ACCOUNTING_SEMANTICS,
                "turn_accounting_provenance": "evaluator_exact_pre_step",
                "agents": agents,
                "action_parse_stats": {},
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_canonical_episode_bundle(root, manifest):
    arm, _ = _write_population_run_binding(root, manifest, population=2)
    task_dir = arm / "alem" / "default"
    task_dir.mkdir(parents=True)
    path = task_dir / "default_run_00.json"
    payload = _treatment_payload(manifest)
    payload["attempt_id"] = "1" * 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    stem = path.name.removesuffix(".json")
    for companion, content in (
        (path.with_name(f"{stem}.csv"), b"step,reward\n"),
        (path.with_name(f"{stem}_states.pkl.gz"), b"gzip"),
    ):
        companion.write_bytes(content)
    rewards = np.zeros((10, 2), dtype=np.float32)
    rewards[0] = (2.0, 3.0)
    np.savez_compressed(
        path.with_name(f"{stem}_trajectory.npz"),
        rewards=rewards,
    )
    _write_exact_v2_debug(path.with_name(f"{stem}_debug.jsonl"))

    # Exercise the production emitter rather than hand-writing fields that
    # could mask a missing ledger field.
    with _attempt_ledger_guard(
        arm,
        "alem",
        "default",
        0,
        payload,
        seed=payload["seed"],
        process_num=0,
    ):
        pass
    return path, payload


def test_episode_row_labels_cache_rates_and_latency(tmp_path):
    root = tmp_path
    path = root / "n2" / "easy" / "alem" / "default" / "default_run_00.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_episode_payload()), encoding="utf-8")
    _write_exact_v2_debug(path.with_name("default_run_00_debug.jsonl"))

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
    assert row["noop_metrics_provenance"] == "episode_debug_reconciled_exact_pre_step"
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
    _write_exact_v2_debug(path.with_name("default_run_00_debug.jsonl"))
    with pytest.raises(ValueError, match="rebuilt raw counters"):
        episode_row(path, root, requested_environment_steps=10)


def _v2_missing_worker_field(payload):
    payload.pop("agent_1_parse_skipped_inactive")


def _v2_missing_features(payload):
    payload.pop("turn_accounting_features")


def _v2_skipped_inactive_disagreement(payload):
    payload["inactive_submitted_turn_count"] = 9
    payload["agent_1_inactive_submitted_turn_count"] = 9


def _v2_canonical_bound_violation(payload):
    payload["canonical_submitted_noop_count"] = 12
    payload["executed_noop_count"] = 12
    payload["agent_0_canonical_submitted_noop_count"] = 2
    payload["agent_0_executed_noop_count"] = 2


def _v2_effective_identity_violation(payload):
    payload["effective_environment_noop_count"] = 12
    payload["agent_0_effective_environment_noop_count"] = 2


def _v2_turn_coverage_violation(payload):
    payload["action_parse_success"] = 8
    payload["agent_0_parse_success"] = 8


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (_v2_missing_worker_field, "invalid or missing agent_1_parse_skipped_inactive"),
        (_v2_missing_features, "feature declaration"),
        (_v2_skipped_inactive_disagreement, "inactive-effective counters disagree"),
        (_v2_canonical_bound_violation, "canonical submitted Noops"),
        (_v2_effective_identity_violation, "effective-Noop identity"),
        (_v2_turn_coverage_violation, r"success\+fail\+skipped"),
    ),
)
def test_declared_v2_episode_never_downgrades_to_legacy_reconstruction(
    tmp_path,
    mutation,
    message,
):
    root = tmp_path
    path = root / "n2" / "easy" / "alem" / "default" / "default_run_00.json"
    path.parent.mkdir(parents=True)
    payload = _episode_payload()
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    # A complete-looking legacy journal must not rescue a malformed v2 marker.
    path.with_name("default_run_00_debug.jsonl").write_text(
        json.dumps(
            {
                "step": 0,
                "agents": {
                    "0": {
                        "llm_raw_output": "<action>Noop</action>",
                        "parsed_action": "Noop",
                        "stop_reason": "stop",
                    },
                    "1": {
                        "llm_raw_output": "<action>Noop</action>",
                        "parsed_action": "Noop",
                        "stop_reason": "stop",
                    },
                },
                "action_parse_stats": {
                    "0": {"success": 1, "fail": 0, "skipped_inactive": 0},
                    "1": {"success": 1, "fail": 0, "skipped_inactive": 0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        episode_row(path, root, requested_environment_steps=10)


def test_debug_record_declaring_v2_requires_features_and_exact_fields(tmp_path):
    episode = tmp_path / "default_run_00.json"
    debug = tmp_path / "default_run_00_debug.jsonl"
    debug.write_text(
        json.dumps(
            {
                "step": 0,
                "turn_accounting_schema_version": TURN_ACCOUNTING_SCHEMA,
                "turn_accounting_semantics": TURN_ACCOUNTING_SEMANTICS,
                "turn_accounting_provenance": "evaluator_exact_pre_step",
                "agents": {
                    "0": {
                        "action_turn_classification": {
                            "pre_step_inactive": False,
                        }
                    }
                },
                "action_parse_stats": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="feature declaration"):
        _debug_noop_metrics(
            episode,
            num_agents=1,
            expected_submitted_turns=1,
        )


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
            "episode_return": 2.0,
            "agent_0_return": 2.0,
            "environment_steps_completed": 2,
            "agent_turns_submitted": 2,
            "alive_agent_turns": 1,
            "actionable_agent_turns": 1,
            "action_parse_rate": 0.5,
            "action_parse_success": 0,
            "action_parse_fail": 1,
            "action_parse_skipped_inactive": 1,
            "action_frequency": {"Noop": 2},
        }
    )
    payload["user_info"]["Cooperation/alive_agent_steps"] = 1
    payload["user_info"]["Cooperation/actionable_agent_steps"] = 1
    payload["performance_metrics"] = build_performance_metrics(
        payload,
        1,
        environment_steps_completed=2,
        agent_turns_submitted=2,
        alive_agent_turns=1,
        actionable_agent_turns=1,
    )
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
    assert binding["source_commit"] == TRUSTED_SOURCE_COMMIT
    assert binding["resolved_model_id"] == TRUSTED_RESOLVED_MODEL

    (tmp_path / "n2" / "resolved_config.yaml").write_text(
        "alem:\n  num_agents: 99\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Study config hash mismatch"):
        validate_manifest_contract(tmp_path, manifest)


def test_manifest_contract_recomputes_normalized_semantics(tmp_path):
    manifest = _strict_manifest(tmp_path)
    manifest["resolved_config_sha256"]["2"] = "d" * 64
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        validate_manifest_contract(tmp_path, manifest)


def test_self_consistent_untrusted_source_commit_is_rejected(tmp_path):
    manifest = _strict_manifest(tmp_path)
    untrusted_commit = "7d377a668197e1124d33d9b7a5b4161a455d6199"
    manifest["source_commit"] = untrusted_commit
    preflight_path = tmp_path / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["source_commit"] = untrusted_commit
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["preflight_sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="trusted E1 source commit"):
        validate_manifest_contract(tmp_path, manifest)


def test_paid_launcher_is_pinned_to_trusted_source(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(source_scaling_launcher, "_blocking_source_status", lambda: ())
    with pytest.raises(
        source_scaling_launcher.SourceScalingLaunchError,
        match="pinned to trusted source commit",
    ):
        source_scaling_launcher._validate_paid_run(
            {
                "source_commit": "7d377a668197e1124d33d9b7a5b4161a455d6199",
                "uv_lock_sha256": TRUSTED_UV_LOCK_SHA256,
            }
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(status="failed"), "status mismatch"),
        (
            lambda payload: payload.update(model_id="gpt-5.4-nano-2099-01-01"),
            "status mismatch",
        ),
        (lambda payload: payload.update(provider_status="failed"), "status mismatch"),
    ),
)
def test_preflight_model_and_status_are_bound_even_with_updated_hash(
    tmp_path,
    mutation,
    message,
):
    manifest = _strict_manifest(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    mutation(preflight)
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["preflight_sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        validate_manifest_contract(tmp_path, manifest)


def test_preflight_file_hash_is_bound_to_manifest(tmp_path):
    manifest = _strict_manifest(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        preflight_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="preflight hash mismatch"):
        validate_manifest_contract(tmp_path, manifest)


def test_copied_or_moved_study_root_fails_output_binding(tmp_path):
    original = tmp_path / "original"
    copied = tmp_path / "copied"
    manifest = _strict_manifest(original)
    (original / "study_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copytree(original, copied)
    copied_manifest = json.loads((copied / "study_manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="output_root"):
        validate_manifest_contract(copied, copied_manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["clients"][0].update(model_id="wrong-model"),
            "immutable client treatment",
        ),
        (lambda payload: payload["team"].update(topology="leader_peer"), "contamination"),
        (
            lambda payload: payload["agent"].update(use_communication=False),
            "contamination",
        ),
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["clients"].append(dict(payload["clients"][0])), "six-slot"),
        (
            lambda payload: payload["clients"][5]["generate_kwargs"].update(
                prompt_cache_key="poison"
            ),
            "immutable client treatment",
        ),
        (
            lambda payload: payload["clients"][0].update(prompt_cache_key_resolved=None),
            "unresolved physical-worker cache route",
        ),
        (
            lambda payload: payload["clients"][0].update(enable_thinking_resolved=True),
            "resolved thinking mode",
        ),
        (
            lambda payload: payload["clients"][0].update(behavior_override="poison"),
            "unexpected client treatment fields",
        ),
        (
            lambda payload: payload["model_usage_records"].pop(),
            "decision-call coverage",
        ),
        (
            lambda payload: payload["model_usage_records"][-1].update(participant_id=0),
            "decision-call coverage",
        ),
        (
            lambda payload: payload.update(agent_1_model_call_count=9),
            "worker 1 model-call counter",
        ),
    ),
)
def test_episode_treatment_requires_exact_clients_cache_and_worker_calls(
    tmp_path,
    mutation,
    message,
):
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
    arm, run_manifest = _write_population_run_binding(tmp_path, manifest, population=2)
    validate_population_run_binding(tmp_path, manifest, 2)

    run_manifest["profile"] = "contaminated"
    (arm / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="profile treatment binding mismatch"):
        validate_population_run_binding(tmp_path, manifest, 2)


@pytest.mark.parametrize(
    ("invocation_index", "field", "poison", "message"),
    (
        (0, "source_commit", "f" * 40, "initial invocation source_commit mismatch"),
        (0, "uv_lock_sha256", "d" * 64, "initial invocation uv_lock_sha256 mismatch"),
        (0, "resolved_config_sha256", "e" * 64, "initial invocation runtime config hash"),
        (2, "source_commit", "f" * 40, "resume 2 source_commit mismatch"),
        (2, "uv_lock_sha256", "d" * 64, "resume 2 uv_lock_sha256 mismatch"),
        (2, "resolved_config_sha256", "e" * 64, "resume 2 runtime config hash"),
    ),
)
def test_every_initial_and_resume_invocation_binds_source_lock_and_raw_hash(
    tmp_path,
    invocation_index,
    field,
    poison,
    message,
):
    manifest = _strict_manifest(tmp_path)
    arm, run_manifest = _write_population_run_binding(
        tmp_path,
        manifest,
        population=2,
        resume_count=2,
    )
    validate_population_run_binding(tmp_path, manifest, 2)

    invocation = (
        run_manifest
        if invocation_index == 0
        else run_manifest["resume_history"][invocation_index - 1]
    )
    invocation[field] = poison
    (arm / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_population_run_binding(tmp_path, manifest, 2)


@pytest.mark.parametrize(
    ("invocation_index", "message"),
    (
        (0, "initial invocation runtime semantics mismatch"),
        (2, "resume 2 runtime semantics mismatch"),
    ),
)
def test_runtime_raw_hash_update_cannot_hide_semantic_poison(
    tmp_path,
    invocation_index,
    message,
):
    manifest = _strict_manifest(tmp_path)
    arm, run_manifest = _write_population_run_binding(
        tmp_path,
        manifest,
        population=2,
        resume_count=2,
    )
    invocation = (
        run_manifest
        if invocation_index == 0
        else run_manifest["resume_history"][invocation_index - 1]
    )
    runtime = arm / invocation["resolved_config_file"]
    runtime.write_text(
        runtime.read_text(encoding="utf-8").replace("num_agents: 2", "num_agents: 99"),
        encoding="utf-8",
    )
    invocation["resolved_config_sha256"] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    (arm / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_population_run_binding(tmp_path, manifest, 2)


def test_undeclared_resume_config_is_rejected(tmp_path):
    manifest = _strict_manifest(tmp_path)
    arm, _ = _write_population_run_binding(tmp_path, manifest, population=2)
    (arm / "resolved_config.resume-20260723T999999999999.yaml").write_text(
        (arm / "resolved_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="undeclared or missing"):
        validate_population_run_binding(tmp_path, manifest, 2)


def test_canonical_discovery_requires_companions_and_attempt_binding(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, _ = _write_canonical_episode_bundle(tmp_path, manifest)
    rows = discover_rows(
        tmp_path,
        requested_environment_steps=10,
        manifest=manifest,
    )
    assert [(row["num_agents"], row["seed"]) for row in rows] == [(2, CANONICAL_SEEDS[0])]

    path.with_name("default_run_00_states.pkl.gz").unlink()
    with pytest.raises(ValueError, match="canonical episode companion"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_self_consistent_config_poison_cannot_redefine_source_treatment(tmp_path):
    manifest = _strict_manifest(tmp_path)
    arm, run_manifest = _write_population_run_binding(tmp_path, manifest, population=2)

    study_config = tmp_path / "n2" / "resolved_config.yaml"
    poisoned_study = OmegaConf.load(study_config)
    poisoned_study.agent.use_communication = False
    study_config.write_text(
        OmegaConf.to_yaml(poisoned_study, resolve=True),
        encoding="utf-8",
    )
    manifest["resolved_config_file_sha256"]["2"] = hashlib.sha256(
        study_config.read_bytes()
    ).hexdigest()
    manifest["resolved_config_sha256"]["2"] = _study_config_semantics_sha256(study_config)

    runtime_config = arm / "resolved_config.yaml"
    poisoned_runtime = OmegaConf.load(runtime_config)
    poisoned_runtime.agent.use_communication = False
    runtime_config.write_text(
        OmegaConf.to_yaml(poisoned_runtime, resolve=True),
        encoding="utf-8",
    )
    run_manifest["resolved_config_sha256"] = hashlib.sha256(runtime_config.read_bytes()).hexdigest()
    (arm / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    # Runtime validation must not derive its expectation from the now-poisoned
    # study config, even though every mutable hash agrees with the poison.
    with pytest.raises(ValueError, match="runtime semantics mismatch"):
        validate_population_run_binding(tmp_path, manifest, 2)
    with pytest.raises(ValueError, match="immutable Source profile semantics"):
        validate_manifest_contract(tmp_path, manifest)


def test_finalized_v2_episode_reconciles_complete_debug_journal(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, _ = _write_canonical_episode_bundle(tmp_path, manifest)
    _write_exact_v2_debug(
        path.with_name("default_run_00_debug.jsonl"),
        all_noops=True,
    )

    with pytest.raises(ValueError, match="v2 debug journal disagrees"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_finalized_v2_debug_reconciles_each_worker_not_only_team_totals(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, payload = _write_canonical_episode_bundle(tmp_path, manifest)
    ledger_path = path.parent / "attempt_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for suffix in TURN_ACCOUNTING_AGENT_SUFFIXES.values():
        worker_0 = f"agent_0_{suffix}"
        worker_1 = f"agent_1_{suffix}"
        payload[worker_0], payload[worker_1] = payload[worker_1], payload[worker_0]
        ledger[worker_0], ledger[worker_1] = payload[worker_0], payload[worker_1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="agent_0_|agent_1_"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_finalized_v2_ledger_requires_complete_exact_coverage(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, _ = _write_canonical_episode_bundle(tmp_path, manifest)
    ledger_path = path.parent / "attempt_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["turn_accounting_coverage"] = "unavailable"
    ledger["turn_accounting_unavailable_reason"] = "counter snapshot unavailable"
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coverage is not complete"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_finalized_v2_ledger_requires_decision_provider_counter(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, _ = _write_canonical_episode_bundle(tmp_path, manifest)
    ledger_path = path.parent / "attempt_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger.pop("decision_provider_request_count")
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="decision_provider_request_count"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_episode_return_is_rebuilt_from_workers_and_raw_trajectory(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, payload = _write_canonical_episode_bundle(tmp_path, manifest)
    ledger_path = path.parent / "attempt_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "agent_0_return": 4.0,
            "agent_1_return": 5.0,
            "episode_return": 4.5,
        }
    )
    for field in ("agent_0_return", "agent_1_return", "episode_return"):
        ledger[field] = payload[field]
    path.write_text(json.dumps(payload), encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="return disagrees with trajectory rewards"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_recorded_performance_headline_is_rebuilt_from_raw_counters(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, payload = _write_canonical_episode_bundle(tmp_path, manifest)
    ledger_path = path.parent / "attempt_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["performance_metrics"]["paper_score_percent"]["total"] = 999.0
    ledger["performance_metrics"] = payload["performance_metrics"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rebuilt raw counters"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


@pytest.mark.parametrize("field", ("episode_return", "performance_metrics", "user_info"))
def test_finalized_v2_ledger_requires_headline_and_raw_evidence(tmp_path, field):
    manifest = _strict_manifest(tmp_path)
    path, _ = _write_canonical_episode_bundle(tmp_path, manifest)
    ledger_path = path.parent / "attempt_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger.pop(field)
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_legacy_ledger_may_precede_redundant_decision_provider_counter(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, payload = _write_canonical_episode_bundle(tmp_path, manifest)
    _strip_versioned_turn_accounting(payload)
    payload.update(
        {
            "action_parse_success": 9,
            "action_parse_fail": 1,
            "action_parse_skipped_inactive": 10,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    ledger_path = path.parent / "attempt_ledger.jsonl"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger.pop("decision_provider_request_count")
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")

    rows = discover_rows(
        tmp_path,
        requested_environment_steps=10,
        manifest=manifest,
    )
    assert len(rows) == 1
    assert rows[0]["provider_request_count"] == 20


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update(provider_request_count=0),
        lambda payload: payload.update(provider_request_count=21),
        lambda payload: payload.update(
            transport_error_count=1,
            transport_error_reasons={"timeout": 1},
        ),
        lambda payload: payload.update(agent_1_provider_request_count=9),
    ),
)
def test_episode_treatment_requires_exact_successful_provider_attempts(
    tmp_path,
    mutation,
):
    manifest = _strict_manifest(tmp_path)
    payload = _treatment_payload(manifest)
    mutation(payload)
    with pytest.raises(ValueError, match="provider-request|provider requests"):
        _validate_episode_treatment(
            payload,
            tmp_path / "episode.json",
            manifest,
            num_agents=2,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["model_usage_records"][0].update(
                model_id="gpt-5.4-nano-2099-01-01"
            ),
            "contaminated model usage",
        ),
        (
            lambda payload: payload["clients"][0].update(
                model_id_resolved="gpt-5.4-nano-2099-01-01"
            ),
            "trusted resolved model snapshot",
        ),
        (
            lambda payload: payload["model_usage_records"][0].pop("cache_write_tokens"),
            "missing cache_write_tokens",
        ),
        (
            lambda payload: payload["model_usage_records"][0].update(input_tokens=6),
            "per-call usage disagrees",
        ),
        (
            lambda payload: (
                payload["model_usage_records"][0].update(input_tokens=6),
                payload.update(input_tokens=101, decision_input_tokens=101),
            ),
            "worker 0 aggregates",
        ),
        (
            lambda payload: payload["model_usage_records"][1].update(
                response_id=payload["model_usage_records"][0]["response_id"]
            ),
            "duplicate response_id",
        ),
    ),
)
def test_episode_treatment_binds_resolved_model_and_per_call_usage(
    tmp_path,
    mutation,
    message,
):
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


@pytest.mark.parametrize(
    ("artifact", "message"),
    (
        ("study_manifest", "E1 study manifest must not use symlinks"),
        ("preflight", "canonical E1 preflight must not use symlinks"),
        ("study_config", "N=2 study config must not use symlinks"),
        ("run_manifest", "N=2 run manifest must not use symlinks"),
        ("runtime_config", "runtime config must not use symlinks"),
        ("episode", "Non-canonical episode artifact"),
        ("companion", "canonical episode companion .* must not use symlinks"),
        ("ledger", "canonical attempt ledger must not use symlinks"),
    ),
)
def test_external_symlinked_managed_artifacts_are_rejected(
    tmp_path,
    artifact,
    message,
):
    root = tmp_path / "study"
    manifest = _strict_manifest(root)
    path, _ = _write_canonical_episode_bundle(root, manifest)
    study_manifest_path = root / "study_manifest.json"
    study_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    targets = {
        "study_manifest": study_manifest_path,
        "preflight": root / "preflight.json",
        "study_config": root / "n2" / "resolved_config.yaml",
        "run_manifest": root / "n2" / "easy" / "run_manifest.json",
        "runtime_config": root / "n2" / "easy" / "resolved_config.yaml",
        "episode": path,
        "companion": path.with_name("default_run_00_states.pkl.gz"),
        "ledger": path.parent / "attempt_ledger.jsonl",
    }
    target = targets[artifact]
    external = tmp_path / f"external-{artifact}"
    shutil.copy2(target, external)
    target.unlink()
    target.symlink_to(external.resolve())

    with pytest.raises(ValueError, match=message):
        if artifact == "study_manifest":
            _read_manifest(root)
        elif artifact in {"preflight", "study_config"}:
            validate_manifest_contract(root, manifest)
        else:
            discover_rows(
                root,
                requested_environment_steps=10,
                manifest=manifest,
            )


@pytest.mark.parametrize(
    "artifact",
    (
        "study_manifest",
        "preflight",
        "study_config",
        "run_manifest",
        "runtime_config",
        "episode",
        "companion",
        "ledger",
    ),
)
def test_multiply_linked_managed_artifacts_are_rejected(tmp_path, artifact):
    root = tmp_path / "study"
    manifest = _strict_manifest(root)
    path, _ = _write_canonical_episode_bundle(root, manifest)
    study_manifest_path = root / "study_manifest.json"
    study_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    targets = {
        "study_manifest": study_manifest_path,
        "preflight": root / "preflight.json",
        "study_config": root / "n2" / "resolved_config.yaml",
        "run_manifest": root / "n2" / "easy" / "run_manifest.json",
        "runtime_config": root / "n2" / "easy" / "resolved_config.yaml",
        "episode": path,
        "companion": path.with_name("default_run_00_states.pkl.gz"),
        "ledger": path.parent / "attempt_ledger.jsonl",
    }
    target = targets[artifact]
    os.link(target, tmp_path / f"external-hardlink-{artifact}")

    with pytest.raises(ValueError, match="exactly one filesystem link"):
        if artifact == "study_manifest":
            _read_manifest(root)
        elif artifact in {"preflight", "study_config"}:
            validate_manifest_contract(root, manifest)
        else:
            discover_rows(
                root,
                requested_environment_steps=10,
                manifest=manifest,
            )


def test_canonical_discovery_rejects_poison_copy_and_unbound_attempt(tmp_path):
    manifest = _strict_manifest(tmp_path)
    path, payload = _write_canonical_episode_bundle(tmp_path, manifest)
    payload["attempt_id"] = "2" * 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="attempt_id does not bind"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )

    # Restore the marker, then add a plausible copy outside the one canonical
    # task path. Discovery must reject rather than silently select one.
    payload["attempt_id"] = "1" * 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    poison = path.parent.parent / "poison" / path.name
    poison.parent.mkdir()
    shutil.copy2(path, poison)
    with pytest.raises(ValueError, match="Non-canonical episode artifact path"):
        discover_rows(
            tmp_path,
            requested_environment_steps=10,
            manifest=manifest,
        )


def test_discovery_rejects_noncanonical_episode_paths(tmp_path):
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

    with pytest.raises(ValueError, match="Non-canonical episode artifact path"):
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
