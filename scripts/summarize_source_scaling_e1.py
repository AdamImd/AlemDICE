#!/usr/bin/env python3
"""Extract and plot E1 Source-baseline performance versus physical agent count.

Expected layout (additional directories such as ``easy`` are allowed):

    RUN_ROOT/n1/.../alem/default/default_run_00.json
    RUN_ROOT/n2/.../alem/default/default_run_00.json

The canonical ``physical_worker_count`` field is authoritative; the ``n<N>``
path component is checked when present. Only complete canonical episode
artifacts are accepted, and a manifest-declared population-by-seed grid must be
complete unless ``--allow-incomplete`` is explicitly selected. Older Noop
categories are reconstructed from per-turn debug journals with recorded
provenance; categories that cannot be recovered remain missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import stat
import sys
from collections import defaultdict
from datetime import UTC, datetime
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_UTILS = ROOT / "baselines" / "llm" / "eval_utils"
if str(EVAL_UTILS) not in sys.path:
    sys.path.insert(0, str(EVAL_UTILS))

from performance_metrics import (  # noqa: E402
    PERFORMANCE_METRICS_SCHEMA,
    build_performance_metrics,
)

from alem_action_parser import (  # noqa: E402
    extract_action_multistrategy,
)
from alem_turn_accounting import (  # noqa: E402
    TURN_ACCOUNTING_AGENT_SUFFIXES,
    TURN_ACCOUNTING_FEATURES,
    TURN_ACCOUNTING_SCHEMA_VERSION,
    TURN_ACCOUNTING_SEMANTICS,
    validate_turn_accounting,
)

EPISODE_PATTERN = re.compile(r".+_run_(\d+)\.json$")
POPULATION_PATTERN = re.compile(r"n(\d+)$", re.IGNORECASE)
MANIFEST_NAME = "study_manifest.json"
MANIFEST_SCHEMA = "alem-dice-source-scaling-e1-v1"
RUN_MANIFEST_SCHEMA = "alem-dice-run-manifest-v1"
PREFLIGHT_NAME = "preflight.json"
PREFLIGHT_SCHEMA = "alem-dice-source-scaling-preflight-v1"
TURN_ACCOUNTING_SCHEMA = TURN_ACCOUNTING_SCHEMA_VERSION
LEGACY_TURN_RECONSTRUCTION_SCHEMA = "alem-dice-turn-accounting-legacy-reconstruction-v1"
CSV_SCHEMA_VERSION = "alem-dice-e1-episodes-csv-v4"
MIN_BOOTSTRAP_REPS = 100
CANONICAL_CLIENT_SLOTS = 6
TRUSTED_SOURCE_COMMIT = "49bc152e2b70609aa9a4518b86b1f8f1fced5a14"
TRUSTED_UV_LOCK_SHA256 = "d75773f66d8a5af4ea339ef8be9c4f9a2cec08e088a74dcacd9f4e746654b128"
TRUSTED_PREFLIGHT_SHA256 = "b9e43822765d250a33348d968e3c6a50147e90aa20f3a714b995beca75e17794"
TRUSTED_RESOLVED_MODEL = "gpt-5.4-nano-2026-03-17"
TRUSTED_NOMINAL_DECISION_CALL_CAP = 9_600
TRUSTED_CAMPAIGN_LOGICAL_CALL_CAP = 12_001
TRUSTED_CAMPAIGN_PROVIDER_ATTEMPT_CAP = 15_000
CANONICAL_STUDY_CONFIG_SEMANTIC_SHA256 = {
    1: "cb0be6dcb1c811eb573e80521605fd4ba609fb79a834a0be29049d86d7d48868",
    2: "f3655f4e2c5c70b6f6cca51ab78c51463911a84d74fb8bff0a37bcab509a70ae",
    3: "a1d17eb17786d114ff2d107bfbbed22bacd4c63ebe21a2e214d0d758c3655e71",
    4: "0cb23010f132637d6fdb0daa83bd18fc5dd47d02be268bef65777f75fd93d32e",
    6: "1f3c786a40188ac05add7944afe9d9ea2191d4bc1b2089357a201e20947d3efc",
}
CANONICAL_POPULATIONS = [1, 2, 3, 4, 6]
CANONICAL_SEEDS = [13100, 13101, 13102]
CANONICAL_TREATMENT = {
    "profile": "source_scaling_200",
    "difficulty": "easy",
    "max_steps_per_episode": 200,
    "episodes_per_count": 3,
    "episode_workers": 1,
    "agent_calls_within_tick": "concurrent",
    "model": "gpt-5.4-nano",
    "reasoning_effort": "high",
    "topology": "baseline",
    "coordination_strategy": "free",
    "agent_type": "robust_all",
    "prompt_mode": "specific_collaborative",
}
SUMMARY_METRICS = (
    "paper_base_percent",
    "paper_coord_percent",
    "paper_total_percent",
    "episode_return",
    "team_unique_base_achievements",
    "team_unique_coord_achievements",
    "team_unique_total_achievements",
    "summed_agent_base_achievements",
    "summed_agent_coord_achievements",
    "summed_agent_total_achievements",
    "coordination_attempts",
    "coordination_resolved_attempts",
    "coordination_successes",
    "give_attempts",
    "successful_transfers",
    "requests",
    "revives",
    "deaths",
    "action_parse_rate",
    "action_parse_success",
    "action_parse_fail",
    "action_parse_skipped_inactive",
    "environment_steps_completed",
    "agent_turns_submitted",
    "alive_agent_turns",
    "actionable_agent_turns",
    "alive_turn_fraction",
    "actionable_turn_fraction",
    "requested_environment_steps",
    "step_completion_fraction",
    "intentional_actionable_noops",
    "parse_fallback_noops",
    "active_action_validation_fallback_noops",
    "active_residual_effective_noops",
    "inactive_billed_turns",
    "inactive_effective_noops",
    "canonical_submitted_noops",
    "effective_environment_noops",
    "executed_noops",
    "input_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "input_cache_fraction",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "model_call_count",
    "provider_request_count",
    "recovered_transport_retry_count",
    "transport_error_count",
    "summed_model_latency_seconds",
    "input_tokens_per_submitted_turn",
    "input_tokens_per_actionable_turn",
    "total_tokens_per_submitted_turn",
    "total_tokens_per_actionable_turn",
    "delivered_bytes",
    "delivered_bytes_per_submitted_turn",
    "delivered_bytes_per_actionable_turn",
    "episode_wall_seconds",
    "mean_tick_wall_seconds",
)
CSV_FIELDS = (
    "csv_schema_version",
    "csv_compatibility_note",
    "analysis_status",
    "analysis_watermark",
    "artifact_path",
    "attempt_id",
    "num_agents",
    "seed",
    "episode_index",
    "termination_reason",
    "used_legacy_metric_fallback",
    "noop_metrics_provenance",
    "noop_metrics_complete",
    "noop_metrics_note",
    "turn_accounting_schema_version",
    "turn_accounting_provenance",
    "turn_accounting_complete",
    "turn_accounting_note",
    "retry_accounting_provenance",
    "retry_accounting_complete",
    "transport_error_reasons_json",
    "legacy_recorded_action_parse_rate",
    *SUMMARY_METRICS,
    "achievement_base_percent",
    "achievement_coord_percent",
    "achievement_total_percent",
    "completed_agent_turn_capacity",
    "classified_action_turns",
    "survival_fraction",
    "actionable_fraction",
    "input_tokens_per_agent_turn",
    "delivered_bytes_per_agent_turn",
)
HEADLINE_METRICS = (
    ("paper_total_percent", "Reward-weighted Total score (%)"),
    ("episode_return", "Mean per-agent return"),
    ("team_unique_total_achievements", "Unique first-unlock types (team)"),
)


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return value


def _count(value: Any) -> int | None:
    number = _number(value)
    if number is None or float(number) < 0 or not float(number).is_integer():
        return None
    return int(number)


def _transport_reason_counts(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: transport error reasons must be an object")
    normalized = {}
    for reason, count_value in value.items():
        count = _count(count_value)
        if not isinstance(reason, str) or not reason or count is None or count < 1:
            raise ValueError(f"{context}: invalid transport error reason count")
        normalized[reason] = count
    return normalized


def _transport_reason_list_counts(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, list) or any(
        not isinstance(reason, str) or not reason for reason in value
    ):
        raise ValueError(f"{context}: invalid per-call transport error types")
    counts: dict[str, int] = defaultdict(int)
    for reason in value:
        counts[reason] += 1
    return dict(counts)


def _nested(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _path_population(path: Path, root: Path) -> int | None:
    for part in path.relative_to(root).parts:
        match = POPULATION_PATTERN.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def _delivered_bytes(payload: dict[str, Any]) -> int | float | None:
    """Return Source's ordinary peer-broadcast fan-out bytes.

    ``CommunicationTracker.as_dict`` currently contains one mapping per
    channel. Reading the registered Source channel directly avoids ever
    double-counting a future aggregate/summary mapping.
    """

    communication = payload.get("communication_metrics")
    if not isinstance(communication, dict):
        return None
    worker_peer = communication.get("worker_peer")
    if not isinstance(worker_peer, dict):
        return None
    return _number(worker_peer.get("delivery_bytes"))


def _divide(numerator: Any, denominator: Any) -> float | None:
    numerator_value = _number(numerator)
    denominator_value = _number(denominator)
    if numerator_value is None or denominator_value is None or denominator_value <= 0:
        return None
    return float(numerator_value) / float(denominator_value)


def _require_managed_regular_file(
    path: Path,
    *,
    managed_root: Path,
    label: str,
    nonempty: bool = False,
) -> Path:
    """Require a regular, non-symlinked file contained by ``managed_root``."""

    root_absolute = managed_root.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"{label} is outside its managed root: {path}") from exc

    current = root_absolute
    try:
        for component in relative.parts:
            current = current / component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} must not use symlinks: {path}")
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required {label}: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot inspect required {label} {path}: {exc}") from exc

    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(
            f"{label} must have exactly one filesystem link, found {metadata.st_nlink}: {path}"
        )
    try:
        resolved_root = root_absolute.resolve(strict=True)
        resolved_path = path_absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Cannot resolve required {label} {path}: {exc}") from exc
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes its managed root: {path}")
    if nonempty and metadata.st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def _read_manifest(root: Path) -> dict[str, Any] | None:
    path = root / MANIFEST_NAME
    if not path.exists() and not path.is_symlink():
        return None
    _require_managed_regular_file(
        path,
        managed_root=root,
        label="E1 study manifest",
        nonempty=True,
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read E1 study manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"E1 study manifest is not a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Cannot hash required E1 binding file {path}: {exc}") from exc
    return digest.hexdigest()


def _resolved_config_payload(path: Path) -> dict[str, Any]:
    """Load one resolved Hydra config without importing the simulator."""

    try:
        from omegaconf import OmegaConf

        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise ValueError(f"Cannot normalize required E1 config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Resolved E1 config is not a mapping: {path}")
    return payload


def _config_semantics_sha256(
    payload: dict[str, Any],
    *,
    drop_wandb_run_id: bool,
) -> str:
    normalized = json.loads(json.dumps(payload, default=str))
    if drop_wandb_run_id:
        wandb_payload = normalized.get("wandb")
        if isinstance(wandb_payload, dict):
            wandb_payload.pop("run_id", None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _study_config_semantics_sha256(path: Path) -> str:
    return _config_semantics_sha256(
        _resolved_config_payload(path),
        drop_wandb_run_id=False,
    )


def _canonical_overrides(population: int) -> tuple[str, ...]:
    return (
        f"alem.num_agents={population}",
        *(
            f"clients.{worker_id}.generate_kwargs.prompt_cache_key="
            f"alem:e1:g54n:n{population}:a{worker_id}"
            for worker_id in range(CANONICAL_CLIENT_SLOTS)
        ),
    )


@cache
def _canonical_source_config_json(
    population: int,
    runtime_arm_root: str | None = None,
) -> str:
    """Compose the trusted Source profile independently of output artifacts."""

    if population not in CANONICAL_POPULATIONS:
        raise ValueError(f"Unsupported canonical E1 population N={population}")
    try:
        from omegaconf import OmegaConf

        from baselines.llm.experiment_config import compose_experiment

        overrides = list(_canonical_overrides(population))
        if runtime_arm_root is not None:
            overrides.extend(
                (
                    f"alem.coordination_difficulty={CANONICAL_TREATMENT['difficulty']}",
                    f"eval.resume_from={runtime_arm_root}",
                )
            )
        config = compose_experiment(
            CANONICAL_TREATMENT["profile"],
            overrides=overrides,
        )
        payload = OmegaConf.to_container(config, resolve=True)
    except Exception as exc:
        raise ValueError(
            f"Cannot compose trusted Source profile for N={population}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Trusted Source profile for N={population} is not a mapping")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_source_config_payload(
    population: int,
    runtime_arm_root: Path | None = None,
) -> dict[str, Any]:
    runtime_text = str(runtime_arm_root.resolve()) if runtime_arm_root is not None else None
    return json.loads(_canonical_source_config_json(population, runtime_text))


def _canonical_study_semantics_sha256(population: int) -> str:
    payload = _canonical_source_config_payload(population)
    digest = _config_semantics_sha256(payload, drop_wandb_run_id=False)
    immutable_digest = CANONICAL_STUDY_CONFIG_SEMANTIC_SHA256[population]
    if digest != immutable_digest:
        raise ValueError(f"Trusted Source profile semantic fingerprint drifted for N={population}")
    return digest


def _expected_runtime_semantics_sha256(population: int, arm_root: Path) -> str:
    # Composed from the checked-in trusted profile, never from the mutable
    # manifest-bound study config.
    payload = _canonical_source_config_payload(population, arm_root)
    return _config_semantics_sha256(payload, drop_wandb_run_id=True)


def _require_hash(value: Any, *, length: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"Study manifest has invalid {label}")
    return value


def _validate_preflight_contract(root: Path, manifest: dict[str, Any]) -> str:
    """Bind the canonical compatibility call to the immutable E1 campaign."""

    declared_hash = _require_hash(
        manifest.get("preflight_sha256"),
        length=64,
        label="preflight_sha256",
    )
    path = root / PREFLIGHT_NAME
    _require_managed_regular_file(
        path,
        managed_root=root,
        label="canonical E1 preflight",
        nonempty=True,
    )
    actual_hash = _sha256_file(path)
    if actual_hash != declared_hash:
        raise ValueError(f"Canonical E1 preflight hash mismatch: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read canonical E1 preflight {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Canonical E1 preflight is not an object: {path}")
    config_hashes = manifest.get("resolved_config_sha256")
    expected = {
        "schema_version": PREFLIGHT_SCHEMA,
        "source_commit": TRUSTED_SOURCE_COMMIT,
        "uv_lock_sha256": TRUSTED_UV_LOCK_SHA256,
        "resolved_config_sha256": (
            config_hashes.get("1") if isinstance(config_hashes, dict) else None
        ),
        "requested_model": CANONICAL_TREATMENT["model"],
        "model_id": TRUSTED_RESOLVED_MODEL,
        "reasoning_effort": CANONICAL_TREATMENT["reasoning_effort"],
        "configured_prompt_cache_key": "alem:e1:g54n:preflight",
        "effective_prompt_cache_key": "alem:e1:g54n:preflight:traffic-0",
        "prompt_cache_traffic_shard": 0,
        "attempt": 1,
        "status": "passed",
        "provider_status": "completed",
        "parse_success": True,
        "parsed_action": "Noop",
        "incomplete_reason": None,
        "stop_reason": "stop",
        "logical_response_count": 1,
        "transport_attempt_count": 1,
        "transport_error_count": 0,
        "transport_error_types": [],
    }
    mismatches = [
        field for field, expected_value in expected.items() if payload.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"{path}: canonical preflight binding/status mismatch for " + ", ".join(mismatches)
        )
    response_id = payload.get("response_id")
    if not isinstance(response_id, str) or not response_id:
        raise ValueError(f"{path}: canonical preflight lacks a provider response_id")
    token_fields = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
    token_values = {field: _count(payload.get(field)) for field in token_fields}
    if any(value is None for value in token_values.values()):
        raise ValueError(f"{path}: canonical preflight has invalid token accounting")
    if token_values["cached_tokens"] > token_values["input_tokens"]:
        raise ValueError(f"{path}: canonical preflight cached tokens exceed input tokens")
    return TRUSTED_RESOLVED_MODEL


def validate_manifest_contract(root: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the immutable E1 treatment and its study-config bindings."""

    if manifest is None:
        raise ValueError(
            "Canonical E1 analysis requires study_manifest.json; "
            "--allow-incomplete relaxes only missing population/seed cells."
        )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Study manifest has unsupported schema_version")
    if manifest.get("planned_counts") != CANONICAL_POPULATIONS:
        raise ValueError("Study manifest planned_counts are not the canonical E1 grid")
    if manifest.get("seeds") != CANONICAL_SEEDS:
        raise ValueError("Study manifest seeds are not the canonical paired E1 seeds")
    expected_stages = {
        "all": CANONICAL_POPULATIONS,
        "e1a": [1, 2, 3, 4],
        "e1b": [6],
    }
    if manifest.get("stage_definitions") != expected_stages:
        raise ValueError("Study manifest has invalid stage_definitions")
    for field, expected in CANONICAL_TREATMENT.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Study manifest treatment mismatch for {field}: "
                f"expected {expected!r}, found {manifest.get(field)!r}"
            )
    expected_call_caps = {
        "nominal_decision_call_cap": TRUSTED_NOMINAL_DECISION_CALL_CAP,
        "campaign_logical_call_cap": TRUSTED_CAMPAIGN_LOGICAL_CALL_CAP,
        "campaign_provider_attempt_cap": TRUSTED_CAMPAIGN_PROVIDER_ATTEMPT_CAP,
        "nominal_call_cap_by_count": {
            str(population): population
            * len(CANONICAL_SEEDS)
            * CANONICAL_TREATMENT["max_steps_per_episode"]
            for population in CANONICAL_POPULATIONS
        },
    }
    cap_mismatches = [
        field
        for field, expected_value in expected_call_caps.items()
        if manifest.get(field) != expected_value
    ]
    if cap_mismatches:
        raise ValueError(
            "Study manifest has invalid frozen campaign call cap(s): " + ", ".join(cap_mismatches)
        )
    output_root = manifest.get("output_root")
    if not isinstance(output_root, str) or output_root != str(root.resolve()):
        raise ValueError("Study manifest output_root does not bind to the analyzed root")
    source_commit = _require_hash(manifest.get("source_commit"), length=40, label="source_commit")
    uv_lock_sha256 = _require_hash(
        manifest.get("uv_lock_sha256"), length=64, label="uv_lock_sha256"
    )
    if source_commit != TRUSTED_SOURCE_COMMIT:
        raise ValueError(
            f"Study manifest is not pinned to the trusted E1 source commit {TRUSTED_SOURCE_COMMIT}"
        )
    if uv_lock_sha256 != TRUSTED_UV_LOCK_SHA256:
        raise ValueError("Study manifest uv.lock hash is not the trusted E1 dependency lock")

    normalized_hashes = manifest.get("resolved_config_sha256")
    file_hashes = manifest.get("resolved_config_file_sha256")
    cache_keys = manifest.get("cache_keys")
    expected_keys = {str(population) for population in CANONICAL_POPULATIONS}
    for mapping, label in (
        (normalized_hashes, "resolved_config_sha256"),
        (file_hashes, "resolved_config_file_sha256"),
        (cache_keys, "cache_keys"),
    ):
        if not isinstance(mapping, dict) or set(mapping) != expected_keys:
            raise ValueError(f"Study manifest has invalid {label} population keys")
    for population in CANONICAL_POPULATIONS:
        population_key = str(population)
        expected_normalized_hash = _require_hash(
            normalized_hashes[population_key],
            length=64,
            label=f"resolved_config_sha256[{population_key}]",
        )
        expected_file_hash = _require_hash(
            file_hashes[population_key],
            length=64,
            label=f"resolved_config_file_sha256[{population_key}]",
        )
        expected_cache_keys = [
            f"alem:e1:g54n:n{population}:a{worker_id}" for worker_id in range(population)
        ]
        if cache_keys[population_key] != expected_cache_keys:
            raise ValueError(f"Study manifest has invalid cache_keys[{population_key}]")
        config_path = root / f"n{population}" / "resolved_config.yaml"
        _require_managed_regular_file(
            config_path,
            managed_root=root,
            label=f"N={population} study config",
            nonempty=True,
        )
        if _sha256_file(config_path) != expected_file_hash:
            raise ValueError(f"Study config hash mismatch: {config_path}")
        actual_semantics = _study_config_semantics_sha256(config_path)
        if actual_semantics != expected_normalized_hash:
            raise ValueError(f"Study config semantic hash mismatch: {config_path}")
        if actual_semantics != _canonical_study_semantics_sha256(population):
            raise ValueError(
                f"Study config violates immutable Source profile semantics: {config_path}"
            )
    resolved_model_id = _validate_preflight_contract(root, manifest)
    return {
        "source_commit": source_commit,
        "uv_lock_sha256": uv_lock_sha256,
        "resolved_model_id": resolved_model_id,
    }


def validate_population_run_binding(
    root: Path,
    manifest: dict[str, Any],
    population: int,
) -> None:
    """Bind an observed population arm to its runtime config and run manifest."""

    arm_root = root / f"n{population}" / CANONICAL_TREATMENT["difficulty"]
    run_manifest_path = arm_root / "run_manifest.json"
    _require_managed_regular_file(
        run_manifest_path,
        managed_root=root,
        label=f"N={population} run manifest",
        nonempty=True,
    )
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read required run manifest {run_manifest_path}: {exc}") from exc
    if not isinstance(run_manifest, dict):
        raise ValueError(f"Run manifest is not an object: {run_manifest_path}")
    exact = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "profile": CANONICAL_TREATMENT["profile"],
        "difficulty": CANONICAL_TREATMENT["difficulty"],
        "resolved_config_file": "resolved_config.yaml",
    }
    for field, expected in exact.items():
        if run_manifest.get(field) != expected:
            raise ValueError(
                f"{run_manifest_path}: {field} treatment binding mismatch "
                f"(expected {expected!r}, found {run_manifest.get(field)!r})"
            )
    models = run_manifest.get("models")
    if models != [CANONICAL_TREATMENT["model"]] * CANONICAL_CLIENT_SLOTS:
        raise ValueError(f"{run_manifest_path}: wrong model declaration")
    resume_history = run_manifest.get("resume_history")
    if not isinstance(resume_history, list):
        raise ValueError(f"{run_manifest_path}: resume_history must be a list")

    invocations = [
        {
            "source_commit": run_manifest.get("source_commit"),
            "uv_lock_sha256": run_manifest.get("uv_lock_sha256"),
            "resolved_config_file": run_manifest.get("resolved_config_file"),
            "resolved_config_sha256": run_manifest.get("resolved_config_sha256"),
        },
        *resume_history,
    ]
    expected_runtime_semantics = _expected_runtime_semantics_sha256(population, arm_root)
    declared_config_names = []
    for invocation_index, invocation in enumerate(invocations):
        label = "initial invocation" if invocation_index == 0 else f"resume {invocation_index}"
        if not isinstance(invocation, dict):
            raise ValueError(f"{run_manifest_path}: {label} is not an object")
        if invocation.get("source_commit") != manifest["source_commit"]:
            raise ValueError(f"{run_manifest_path}: {label} source_commit mismatch")
        if invocation.get("uv_lock_sha256") != manifest["uv_lock_sha256"]:
            raise ValueError(f"{run_manifest_path}: {label} uv_lock_sha256 mismatch")
        config_name = invocation.get("resolved_config_file")
        if (
            not isinstance(config_name, str)
            or Path(config_name).name != config_name
            or not config_name.endswith(".yaml")
        ):
            raise ValueError(f"{run_manifest_path}: {label} has an unsafe config filename")
        if invocation_index == 0 and config_name != "resolved_config.yaml":
            raise ValueError(
                f"{run_manifest_path}: initial runtime config filename is not canonical"
            )
        if (
            invocation_index
            and re.fullmatch(
                r"resolved_config\.resume-\d{8}T\d{12}\.yaml",
                config_name,
            )
            is None
        ):
            raise ValueError(f"{run_manifest_path}: {label} config filename is not canonical")
        if config_name in declared_config_names:
            raise ValueError(f"{run_manifest_path}: duplicate runtime config declaration")
        declared_config_names.append(config_name)
        expected_raw_hash = _require_hash(
            invocation.get("resolved_config_sha256"),
            length=64,
            label=f"{run_manifest_path} {label} resolved_config_sha256",
        )
        runtime_config = arm_root / config_name
        _require_managed_regular_file(
            runtime_config,
            managed_root=root,
            label=f"N={population} {label} runtime config",
            nonempty=True,
        )
        if _sha256_file(runtime_config) != expected_raw_hash:
            raise ValueError(f"{run_manifest_path}: {label} runtime config hash mismatch")
        runtime_semantics = _config_semantics_sha256(
            _resolved_config_payload(runtime_config),
            drop_wandb_run_id=True,
        )
        if runtime_semantics != expected_runtime_semantics:
            raise ValueError(f"{run_manifest_path}: {label} runtime semantics mismatch")

    actual_config_names = sorted(path.name for path in arm_root.glob("resolved_config*.yaml"))
    if actual_config_names != sorted(declared_config_names):
        raise ValueError(f"{run_manifest_path}: undeclared or missing runtime config provenance")


def _validate_episode_treatment(
    payload: dict[str, Any],
    path: Path,
    manifest: dict[str, Any],
    *,
    num_agents: int,
    resolved_model_id: str = TRUSTED_RESOLVED_MODEL,
) -> dict[str, Any]:
    """Reject any episode whose runtime treatment differs from the E1 manifest."""

    expected = CANONICAL_TREATMENT
    declares_v2 = payload.get("turn_accounting_schema_version") == TURN_ACCOUNTING_SCHEMA
    trusted_config = _canonical_source_config_payload(num_agents)
    checks = {
        "task": (payload.get("task"), "default"),
        "logical_participant_count": (
            _count(payload.get("logical_participant_count")),
            num_agents,
        ),
        "team_topology": (payload.get("team_topology"), expected["topology"]),
        "coordination_strategy": (
            payload.get("coordination_strategy"),
            expected["coordination_strategy"],
        ),
        "team configuration": (payload.get("team"), trusted_config.get("team")),
        "agent configuration": (payload.get("agent"), trusted_config.get("agent")),
    }
    mismatches = [
        f"{label}={actual!r} (expected {wanted!r})"
        for label, (actual, wanted) in checks.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(f"{path}: E1 treatment contamination: {', '.join(mismatches)}")

    clients = payload.get("clients")
    if not isinstance(clients, list) or len(clients) != CANONICAL_CLIENT_SLOTS:
        raise ValueError(f"{path}: client configuration does not match six-slot E1 profile")
    expected_keys = manifest["cache_keys"][str(num_agents)]
    trusted_clients = trusted_config.get("clients")
    if not isinstance(trusted_clients, list) or len(trusted_clients) != CANONICAL_CLIENT_SLOTS:
        raise ValueError("Trusted Source profile has an invalid client declaration")
    derived_client_fields = {
        "enable_thinking_resolved",
        "model_id_resolved",
        "model_ids_resolved",
        "prompt_cache_key_resolved",
        "prompt_cache_traffic_shard_resolved",
    }
    for worker_id, client in enumerate(clients):
        if not isinstance(client, dict):
            raise ValueError(f"{path}: clients[{worker_id}] is not an object")
        unexpected_fields = set(client) - set(trusted_clients[worker_id]) - derived_client_fields
        if unexpected_fields:
            raise ValueError(
                f"{path}: unexpected client treatment fields at index {worker_id}: "
                + ", ".join(sorted(unexpected_fields))
            )
        if {key: client.get(key) for key in trusted_clients[worker_id]} != trusted_clients[
            worker_id
        ]:
            raise ValueError(f"{path}: wrong immutable client treatment at index {worker_id}")
        if client.get("enable_thinking_resolved") is not False:
            raise ValueError(f"{path}: wrong resolved thinking mode at client slot {worker_id}")
        client_resolved_model = client.get("model_id_resolved")
        client_resolved_models = client.get("model_ids_resolved")
        if worker_id < num_agents:
            if declares_v2 and (
                client_resolved_model != resolved_model_id
                or client_resolved_models != [resolved_model_id]
            ):
                raise ValueError(
                    f"{path}: client {worker_id} lacks the trusted resolved model snapshot"
                )
            if client_resolved_model is not None and client_resolved_model != resolved_model_id:
                raise ValueError(f"{path}: client {worker_id} resolved model drifted")
            if client_resolved_models is not None and client_resolved_models != [resolved_model_id]:
                raise ValueError(f"{path}: client {worker_id} resolved model set drifted")
        elif client_resolved_model is not None or client_resolved_models not in (None, []):
            raise ValueError(f"{path}: unused client slot {worker_id} has a resolved model")
        expected_key = f"alem:e1:g54n:n{num_agents}:a{worker_id}"
        if worker_id < num_agents and expected_key != expected_keys[worker_id]:
            raise ValueError(f"{path}: manifest cache-key declaration is inconsistent")
        if _nested(client, "generate_kwargs", "prompt_cache_key") != expected_key:
            raise ValueError(f"{path}: wrong prompt cache key at client slot {worker_id}")
        resolved_key = client.get("prompt_cache_key_resolved")
        resolved_shard = client.get("prompt_cache_traffic_shard_resolved")
        if worker_id < num_agents:
            if resolved_key != f"{expected_key}:traffic-0" or resolved_shard != 0:
                raise ValueError(f"{path}: unresolved physical-worker cache route {worker_id}")
        elif resolved_key is not None or resolved_shard is not None:
            raise ValueError(f"{path}: unused client slot {worker_id} has a resolved cache route")

    usage = payload.get("model_usage_records")
    model_call_count = _count(payload.get("model_call_count"))
    decision_call_count = _count(payload.get("decision_model_call_count"))
    num_steps = _count(payload.get("num_steps"))
    expected_decision_calls = num_steps * num_agents if num_steps is not None else None
    provider_request_count = _count(payload.get("provider_request_count"))
    decision_provider_request_count = _count(payload.get("decision_provider_request_count"))
    transport_error_count = _count(payload.get("transport_error_count"))
    if (
        not isinstance(usage, list)
        or model_call_count is None
        or decision_call_count is None
        or expected_decision_calls is None
        or model_call_count != expected_decision_calls
        or decision_call_count != expected_decision_calls
        or len(usage) != expected_decision_calls
    ):
        raise ValueError(f"{path}: incomplete baseline decision-call coverage")
    max_transport_retries = _count(trusted_clients[0].get("max_retries"))
    if max_transport_retries is None or any(
        _count(client.get("max_retries")) != max_transport_retries
        for client in trusted_clients[:num_agents]
    ):
        raise ValueError(f"{path}: invalid trusted transport retry ceiling")
    transport_error_reasons = _transport_reason_counts(
        payload.get("transport_error_reasons"),
        context=str(path),
    )
    if (
        transport_error_count is None
        or sum(transport_error_reasons.values()) != transport_error_count
        or provider_request_count != expected_decision_calls + transport_error_count
        or decision_provider_request_count != provider_request_count
        or transport_error_count > expected_decision_calls * max_transport_retries
    ):
        raise ValueError(
            f"{path}: provider-request/transport-error accounting does not reconcile "
            "to completed baseline decisions"
        )
    if (
        _count(payload.get("incomplete_response_count")) != 0
        or payload.get("incomplete_response_reasons") != {}
        or payload.get("stop_reason_counts") != {"stop": expected_decision_calls}
    ):
        raise ValueError(f"{path}: baseline decisions contain unresolved or incomplete responses")
    if _count(payload.get("debrief_model_call_count")) != 0:
        raise ValueError(f"{path}: E1 baseline unexpectedly contains debrief calls")
    if _count(payload.get("commander_plan_model_call_count")) != 0:
        raise ValueError(f"{path}: E1 baseline unexpectedly contains commander calls")
    leader_calls = payload.get("leader_model_call_count")
    if leader_calls is not None and _count(leader_calls) != 0:
        raise ValueError(f"{path}: E1 baseline unexpectedly contains leader calls")
    if not usage:
        raise ValueError(f"{path}: missing model_usage_records")
    calls_by_worker = dict.fromkeys(range(num_agents), 0)
    usage_fields = (
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_write_tokens",
    )
    usage_totals = dict.fromkeys(usage_fields, 0)
    usage_by_worker = {worker_id: dict.fromkeys(usage_fields, 0) for worker_id in range(num_agents)}
    usage_attempts_by_worker = dict.fromkeys(range(num_agents), 0)
    usage_errors_by_worker = dict.fromkeys(range(num_agents), 0)
    usage_reasons_by_worker = {worker_id: defaultdict(int) for worker_id in range(num_agents)}
    usage_transport_attempts = 0
    usage_transport_errors = 0
    usage_transport_reasons: dict[str, int] = defaultdict(int)
    usage_has_exact_transport = None
    response_ids = set()
    for index, call in enumerate(usage):
        if not isinstance(call, dict):
            raise ValueError(f"{path}: model_usage_records[{index}] is not an object")
        participant = _count(call.get("participant_id"))
        if (
            participant is None
            or participant >= num_agents
            or call.get("phase") != "decision"
            or call.get("model_id") != resolved_model_id
        ):
            raise ValueError(f"{path}: contaminated model usage record at index {index}")
        response_id = call.get("response_id")
        if not isinstance(response_id, str) or not response_id or response_id in response_ids:
            raise ValueError(f"{path}: invalid or duplicate response_id at usage record {index}")
        response_ids.add(response_id)
        for field in usage_fields:
            value = _count(call.get(field))
            if field == "cache_write_tokens" and value is None and not declares_v2:
                value = 0
            if value is None:
                raise ValueError(f"{path}: invalid or missing {field} at usage record {index}")
            usage_totals[field] += value
            usage_by_worker[participant][field] += value
        if call["cached_tokens"] > call["input_tokens"]:
            raise ValueError(f"{path}: cached tokens exceed input at usage record {index}")
        transport_fields = (
            "transport_attempt_count",
            "transport_error_count",
            "transport_error_types",
        )
        has_transport_fields = [field in call for field in transport_fields]
        if any(has_transport_fields) and not all(has_transport_fields):
            raise ValueError(f"{path}: partial per-call transport evidence at usage record {index}")
        call_has_exact_transport = all(has_transport_fields)
        if usage_has_exact_transport is None:
            usage_has_exact_transport = call_has_exact_transport
        elif usage_has_exact_transport != call_has_exact_transport:
            raise ValueError(f"{path}: mixed per-call transport evidence")
        if declares_v2 and not call_has_exact_transport:
            raise ValueError(f"{path}: finalized v2 usage lacks per-call transport evidence")
        if call_has_exact_transport:
            attempts = _count(call.get("transport_attempt_count"))
            errors = _count(call.get("transport_error_count"))
            reason_counts = _transport_reason_list_counts(
                call.get("transport_error_types"),
                context=f"{path}: usage record {index}",
            )
            if (
                attempts is None
                or errors is None
                or attempts != errors + 1
                or errors > max_transport_retries
                or sum(reason_counts.values()) != errors
            ):
                raise ValueError(
                    f"{path}: per-call provider attempts do not reconcile at usage record {index}"
                )
            if (
                call.get("provider_status") != "completed"
                or call.get("stop_reason") != "stop"
                or call.get("incomplete_reason") is not None
            ):
                raise ValueError(f"{path}: usage record {index} is not a final completed response")
            usage_transport_attempts += attempts
            usage_transport_errors += errors
            usage_attempts_by_worker[participant] += attempts
            usage_errors_by_worker[participant] += errors
            for reason, count in reason_counts.items():
                usage_transport_reasons[reason] += count
                usage_reasons_by_worker[participant][reason] += count
        calls_by_worker[participant] += 1

    if usage_has_exact_transport and (
        usage_transport_attempts != provider_request_count
        or usage_transport_errors != transport_error_count
        or dict(usage_transport_reasons) != transport_error_reasons
    ):
        raise ValueError(f"{path}: per-call transport evidence disagrees with episode aggregates")

    if declares_v2 and (
        payload.get("resolved_model_id") != resolved_model_id
        or payload.get("resolved_model_ids") != [resolved_model_id]
    ):
        raise ValueError(f"{path}: episode lacks the trusted resolved model snapshot")
    if payload.get("resolved_model_id") not in (None, resolved_model_id):
        raise ValueError(f"{path}: episode resolved model drifted")
    if payload.get("resolved_model_ids") not in (None, [resolved_model_id]):
        raise ValueError(f"{path}: episode resolved model set drifted")

    aggregate_fields = {
        "input_tokens": "input_tokens",
        "cached_tokens": "cached_tokens",
        "output_tokens": "output_tokens",
        "reasoning_tokens": "reasoning_tokens",
        "cache_write_tokens": "cache_write_tokens",
    }
    aggregate_mismatches = [
        payload_field
        for usage_field, payload_field in aggregate_fields.items()
        if _count(payload.get(payload_field)) != usage_totals[usage_field]
        or _count(payload.get(f"decision_{payload_field}")) != usage_totals[usage_field]
    ]
    if aggregate_mismatches:
        raise ValueError(
            f"{path}: per-call usage disagrees with episode/decision aggregates for "
            + ", ".join(aggregate_mismatches)
        )
    worker_provider_total = 0
    worker_error_total = 0
    worker_error_counts = {}
    worker_provider_counts = {}
    for worker_id, call_count in calls_by_worker.items():
        if call_count != num_steps:
            raise ValueError(f"{path}: worker {worker_id} decision-call coverage is incomplete")
        if _count(payload.get(f"agent_{worker_id}_model_call_count")) != num_steps:
            raise ValueError(f"{path}: worker {worker_id} model-call counter is inconsistent")
        worker_provider_count = _count(payload.get(f"agent_{worker_id}_provider_request_count"))
        worker_error_count = _count(payload.get(f"agent_{worker_id}_transport_error_count"))
        if (
            worker_error_count is None
            or worker_provider_count != num_steps + worker_error_count
            or worker_error_count > num_steps * max_transport_retries
        ):
            raise ValueError(
                f"{path}: worker {worker_id} provider-request/error counters are inconsistent"
            )
        if payload.get(f"agent_{worker_id}_incomplete_response_count") != 0 or payload.get(
            f"agent_{worker_id}_stop_reason_counts"
        ) != {"stop": num_steps}:
            raise ValueError(f"{path}: worker {worker_id} has unresolved or incomplete responses")
        worker_provider_counts[worker_id] = worker_provider_count
        worker_error_counts[worker_id] = worker_error_count
        worker_provider_total += worker_provider_count
        worker_error_total += worker_error_count
        if usage_has_exact_transport and (
            usage_attempts_by_worker[worker_id] != worker_provider_count
            or usage_errors_by_worker[worker_id] != worker_error_count
        ):
            raise ValueError(
                f"{path}: per-call transport evidence disagrees with worker {worker_id}"
            )
        worker_mismatches = [
            payload_field
            for usage_field, payload_field in aggregate_fields.items()
            if _count(payload.get(f"agent_{worker_id}_{payload_field}"))
            != usage_by_worker[worker_id][usage_field]
        ]
        if worker_mismatches:
            raise ValueError(
                f"{path}: per-call usage disagrees with worker {worker_id} aggregates for "
                + ", ".join(worker_mismatches)
            )

    if (
        worker_provider_total != provider_request_count
        or worker_error_total != transport_error_count
    ):
        raise ValueError(f"{path}: worker provider-request/error totals do not reconcile")

    worker_reason_fields = [
        f"agent_{worker_id}_transport_error_reasons" in payload for worker_id in range(num_agents)
    ]
    if any(worker_reason_fields) and not all(worker_reason_fields):
        raise ValueError(f"{path}: partial worker transport-error type evidence")
    if all(worker_reason_fields):
        worker_reason_total: dict[str, int] = defaultdict(int)
        for worker_id in range(num_agents):
            reason_counts = _transport_reason_counts(
                payload[f"agent_{worker_id}_transport_error_reasons"],
                context=f"{path}: worker {worker_id}",
            )
            if sum(reason_counts.values()) != worker_error_counts[worker_id]:
                raise ValueError(
                    f"{path}: worker {worker_id} transport error types do not match its count"
                )
            if usage_has_exact_transport and reason_counts != dict(
                usage_reasons_by_worker[worker_id]
            ):
                raise ValueError(
                    f"{path}: per-call transport error types disagree with worker {worker_id}"
                )
            for reason, count in reason_counts.items():
                worker_reason_total[reason] += count
        if dict(worker_reason_total) != transport_error_reasons:
            raise ValueError(f"{path}: worker transport error types do not reconcile globally")
        retry_provenance = (
            "per_call_exact" if usage_has_exact_transport else "legacy_worker_aggregate_exact"
        )
    else:
        if declares_v2:
            raise ValueError(f"{path}: finalized v2 episode lacks worker transport-error types")
        workers_with_errors = [
            worker_id for worker_id, count in worker_error_counts.items() if count
        ]
        if len(workers_with_errors) > 1:
            raise ValueError(
                f"{path}: legacy transport error types cannot be attributed exactly "
                "across multiple workers"
            )
        retry_provenance = (
            "legacy_aggregate_exact_single_worker_type_inference"
            if workers_with_errors
            else "legacy_aggregate_exact"
        )

    return {
        "recovered_transport_retry_count": transport_error_count,
        "transport_error_count": transport_error_count,
        "transport_error_reasons": transport_error_reasons,
        "retry_accounting_provenance": retry_provenance,
        "retry_accounting_complete": True,
    }


def _require_episode_companions(path: Path, *, root: Path) -> None:
    stem = path.name.removesuffix(".json")
    companions = (
        path.with_name(f"{stem}.csv"),
        path.with_name(f"{stem}_trajectory.npz"),
        path.with_name(f"{stem}_states.pkl.gz"),
        path.with_name(f"{stem}_debug.jsonl"),
    )
    for companion in companions:
        _require_managed_regular_file(
            companion,
            managed_root=root,
            label=f"canonical episode companion {companion.name}",
            nonempty=True,
        )


def _validate_episode_attempt_binding(
    payload: dict[str, Any],
    path: Path,
    *,
    episode_index: int,
    root: Path,
) -> None:
    attempt_id = payload.get("attempt_id")
    if not isinstance(attempt_id, str) or re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise ValueError(f"{path}: missing canonical attempt_id")
    ledger_path = path.parent / "attempt_ledger.jsonl"
    _require_managed_regular_file(
        ledger_path,
        managed_root=root,
        label="canonical attempt ledger",
        nonempty=True,
    )

    rows = []
    seen_attempt_ids = set()
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read attempt ledger: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ledger_path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or row.get("schema_version") != "alem-dice-attempt-v1":
            raise ValueError(f"{ledger_path}:{line_number}: invalid attempt row")
        row_attempt_id = row.get("attempt_id")
        if not isinstance(row_attempt_id, str) or not row_attempt_id:
            raise ValueError(f"{ledger_path}:{line_number}: missing attempt_id")
        if row_attempt_id in seen_attempt_ids:
            raise ValueError(f"{ledger_path}:{line_number}: duplicate attempt_id")
        seen_attempt_ids.add(row_attempt_id)
        rows.append(row)

    matches = [row for row in rows if row["attempt_id"] == attempt_id]
    if len(matches) != 1:
        raise ValueError(f"{path}: attempt_id does not bind to exactly one ledger row")
    ledger_row = matches[0]
    declares_v2 = payload.get("turn_accounting_schema_version") == TURN_ACCOUNTING_SCHEMA
    exact = {
        "episode_index": episode_index,
        "artifact_status": "complete",
        "error": None,
        "seed": payload.get("seed"),
        "termination_reason": payload.get("termination_reason"),
        "num_steps": payload.get("num_steps"),
        "model_call_count": payload.get("model_call_count"),
        "provider_request_count": payload.get("provider_request_count"),
        "transport_error_count": payload.get("transport_error_count"),
        "transport_error_reasons": payload.get("transport_error_reasons"),
        "incomplete_response_count": payload.get("incomplete_response_count"),
        "incomplete_response_reasons": payload.get("incomplete_response_reasons"),
        "stop_reason_counts": payload.get("stop_reason_counts"),
        "decision_model_call_count": payload.get("decision_model_call_count"),
        "input_tokens": payload.get("input_tokens"),
        "output_tokens": payload.get("output_tokens"),
        "reasoning_tokens": payload.get("reasoning_tokens"),
        "cached_tokens": payload.get("cached_tokens"),
        "cache_write_tokens": payload.get("cache_write_tokens"),
        "model_usage_records": payload.get("model_usage_records"),
    }
    # The original attempt-ledger schema predates this redundant decision-only
    # provider counter. Require it for finalized v2 artifacts (and bind it when
    # a legacy row happens to provide it) without making historical ledgers
    # impossible to audit.
    if declares_v2 or "decision_provider_request_count" in ledger_row:
        exact["decision_provider_request_count"] = payload.get("decision_provider_request_count")
    optional_evidence_fields = (
        "resolved_model_id",
        "resolved_model_ids",
        "decision_input_tokens",
        "decision_output_tokens",
        "decision_reasoning_tokens",
        "decision_cached_tokens",
        "decision_cache_write_tokens",
        "episode_return",
        "performance_metrics",
        "user_info",
        "environment_steps_completed",
        "agent_turns_submitted",
        "alive_agent_turns",
        "actionable_agent_turns",
    )
    for field in optional_evidence_fields:
        if declares_v2 or field in ledger_row:
            exact[field] = payload.get(field)
    worker_evidence_suffixes = (
        "model_call_count",
        "provider_request_count",
        "transport_error_count",
        "transport_error_reasons",
        "stop_reason_counts",
        "incomplete_response_count",
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_write_tokens",
        "return",
        "resolved_model_id",
        "resolved_model_ids",
    )
    worker_count = _count(payload.get("physical_worker_count"))
    if worker_count is None:
        raise ValueError(f"{path}: missing physical worker count for ledger binding")
    for worker_id in range(worker_count):
        for suffix in worker_evidence_suffixes:
            field = f"agent_{worker_id}_{suffix}"
            if declares_v2 or field in ledger_row:
                exact[field] = payload.get(field)
    mismatches = [field for field, expected in exact.items() if ledger_row.get(field) != expected]
    if mismatches:
        raise ValueError(
            f"{path}: stable episode disagrees with its attempt ledger for " + ", ".join(mismatches)
        )

    if declares_v2:
        if (
            ledger_row.get("turn_accounting_coverage") != "complete"
            or ledger_row.get("turn_accounting_unavailable_reason") is not None
        ):
            raise ValueError(f"{path}: finalized v2 attempt ledger coverage is not complete")
        validate_turn_accounting(
            ledger_row,
            context=f"{path}: attempt-ledger v2 turn accounting",
        )
        accounting_fields = (
            "turn_accounting_schema_version",
            "turn_accounting_features",
            "turn_accounting_complete",
            "turn_accounting_semantics",
            "turn_accounting_provenance",
            "physical_worker_count",
            "action_parse_success",
            "action_parse_fail",
            "action_parse_skipped_inactive",
            *TURN_ACCOUNTING_AGENT_SUFFIXES.keys(),
        )
        accounting_fields = tuple(dict.fromkeys(accounting_fields))
        worker_fields = tuple(
            f"agent_{worker_id}_{suffix}"
            for worker_id in range(worker_count)
            for suffix in TURN_ACCOUNTING_AGENT_SUFFIXES.values()
        )
        mismatches = [
            field
            for field in (*accounting_fields, *worker_fields)
            if ledger_row.get(field) != payload.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"{path}: v2 accounting disagrees with its attempt ledger for "
                + ", ".join(mismatches)
            )


def _legacy_parse_failed(agent_debug: dict[str, Any]) -> bool | None:
    """Re-run the committed Source parser on the final raw response."""

    if agent_debug.get("stop_reason") == "content_filter":
        return True
    raw = agent_debug.get("llm_raw_output")
    if not isinstance(raw, str):
        return None
    return extract_action_multistrategy(raw) is None


def _iter_debug_records(debug_path: Path):
    """Yield decoded JSONL records without retaining image-heavy journals."""

    record_index = 0
    try:
        with debug_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{debug_path}:{line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{debug_path}:{line_number}: record is not an object")
                yield record_index, record
                record_index += 1
    except OSError as exc:
        raise ValueError(f"Cannot read debug journal {debug_path}: {exc}") from exc


def _classify_reconstructed_turn(
    *,
    pre_step_inactive: bool,
    selected_action: Any,
    canonical_action: str,
    parse_failed: bool,
) -> dict[str, Any]:
    inactive = bool(pre_step_inactive)
    failed = bool(parse_failed)
    canonical_noop = canonical_action == "Noop"
    effective_noop = inactive or canonical_noop
    intentional = selected_action == "Noop" and canonical_noop and not inactive and not failed
    parse_fallback = canonical_noop and not inactive and failed
    validation_fallback = (
        canonical_noop
        and not inactive
        and not failed
        and isinstance(selected_action, str)
        and selected_action != "Noop"
    )
    inactive_effective = inactive
    active_residual = (
        effective_noop
        and not inactive_effective
        and not intentional
        and not parse_fallback
        and not validation_fallback
    )
    partition = (
        intentional,
        parse_fallback,
        validation_fallback,
        inactive_effective,
        active_residual,
    )
    if sum(map(int, partition)) != int(effective_noop):
        raise AssertionError("reconstructed effective Noop partition is not exhaustive")
    return {
        "parse_classification": (
            "skipped_inactive" if inactive else ("failure" if failed else "success")
        ),
        "intentional_actionable_noops": intentional,
        "parse_fallback_noops": parse_fallback,
        "active_action_validation_fallback_noops": validation_fallback,
        "active_residual_effective_noops": active_residual,
        "inactive_billed_turns": inactive,
        "inactive_effective_noops": inactive_effective,
        "canonical_submitted_noops": canonical_noop,
        "effective_environment_noops": effective_noop,
        "executed_noops": canonical_noop,
    }


def _debug_noop_metrics(
    episode_path: Path,
    *,
    num_agents: int,
    expected_submitted_turns: int | None,
) -> dict[str, Any] | None:
    """Recover the Noop taxonomy from a per-turn debug journal.

    New journals carry exact pre-step classifications.  For legacy journals,
    raw outputs are reparsed and pre-step inactivity at turn ``t`` is recovered
    from the post-step inactive classification recorded at ``t-1``.  Alem
    initializes all workers active, which supplies the turn-zero state.
    """

    debug_path = episode_path.with_name(f"{episode_path.stem}_debug.jsonl")
    if not debug_path.is_file():
        return None

    count_fields = (
        "intentional_actionable_noops",
        "parse_fallback_noops",
        "active_action_validation_fallback_noops",
        "active_residual_effective_noops",
        "inactive_billed_turns",
        "inactive_effective_noops",
        "canonical_submitted_noops",
        "effective_environment_noops",
        "executed_noops",
        "action_parse_success",
        "action_parse_fail",
        "action_parse_skipped_inactive",
    )
    counts = dict.fromkeys(count_fields, 0)
    worker_counts = {agent_idx: dict.fromkeys(count_fields, 0) for agent_idx in range(num_agents)}
    previous_skipped = [0] * num_agents
    pre_step_inactive = [False] * num_agents
    exact_records = 0
    legacy_records = 0
    recovered_turns = 0
    unknown_turns = 0
    seen_steps = set()
    journal_mode = None

    for expected_step, record in _iter_debug_records(debug_path):
        step = _count(record.get("step"))
        if step is None or step in seen_steps:
            raise ValueError(f"{debug_path}: missing or duplicate non-negative step")
        if step != expected_step:
            raise ValueError(
                f"{debug_path}: expected contiguous step {expected_step}, found {step}"
            )
        seen_steps.add(step)
        agents = record.get("agents")
        parse_stats = record.get("action_parse_stats")
        if not isinstance(agents, dict):
            raise ValueError(f"{debug_path}: step {step} has no agents mapping")
        if not isinstance(parse_stats, dict):
            parse_stats = {}

        for agent_idx in range(num_agents):
            agent_debug = agents.get(str(agent_idx))
            if not isinstance(agent_debug, dict):
                unknown_turns += 1
                continue
            classification = agent_debug.get("action_turn_classification")
            exact_required = {
                "pre_step_inactive",
                "parse_classification",
                "intentional_actionable_noop",
                "parse_fallback_noop",
                "active_action_validation_fallback_noop",
                "active_residual_effective_noop",
                "inactive_submitted_turn",
                "inactive_effective_noop",
                "canonical_submitted_noop",
                "effective_environment_noop",
                "executed_noop",
            }
            exact_declared = "turn_accounting_schema_version" in record
            exact_version = record.get("turn_accounting_schema_version")
            if isinstance(classification, dict) and not exact_declared:
                raise ValueError(
                    f"{debug_path}: step {step}, agent {agent_idx} has an "
                    "unversioned exact classification"
                )
            if exact_declared:
                if exact_version != TURN_ACCOUNTING_SCHEMA:
                    raise ValueError(
                        f"{debug_path}: step {step} declares unsupported "
                        f"turn-accounting schema {exact_version!r}"
                    )
                if list(record.get("turn_accounting_features") or ()) != list(
                    TURN_ACCOUNTING_FEATURES
                ):
                    raise ValueError(
                        f"{debug_path}: step {step} lacks exact v2 feature declaration"
                    )
                if record.get("turn_accounting_semantics") != TURN_ACCOUNTING_SEMANTICS:
                    raise ValueError(f"{debug_path}: step {step} lacks exact v2 semantics")
                if record.get("turn_accounting_provenance") != "evaluator_exact_pre_step":
                    raise ValueError(f"{debug_path}: step {step} lacks exact v2 provenance")
                if not isinstance(classification, dict) or not (
                    exact_required <= classification.keys()
                ):
                    raise ValueError(
                        f"{debug_path}: step {step}, agent {agent_idx} declares v2 "
                        "without a complete exact classification"
                    )

            if exact_version == TURN_ACCOUNTING_SCHEMA:
                boolean_keys = exact_required - {"parse_classification"}
                if any(not isinstance(classification[key], bool) for key in boolean_keys):
                    raise ValueError(
                        f"{debug_path}: step {step}, agent {agent_idx} has a "
                        "non-boolean exact classification"
                    )
                parse_classification = classification["parse_classification"]
                if parse_classification not in {
                    "success",
                    "failure",
                    "skipped_inactive",
                }:
                    raise ValueError(
                        f"{debug_path}: step {step}, agent {agent_idx} has an "
                        "invalid parse classification"
                    )
                record_mode = "exact"
                turn = {
                    "intentional_actionable_noops": bool(
                        classification["intentional_actionable_noop"]
                    ),
                    "parse_fallback_noops": bool(classification["parse_fallback_noop"]),
                    "active_action_validation_fallback_noops": bool(
                        classification["active_action_validation_fallback_noop"]
                    ),
                    "active_residual_effective_noops": bool(
                        classification["active_residual_effective_noop"]
                    ),
                    "inactive_billed_turns": bool(classification["inactive_submitted_turn"]),
                    "inactive_effective_noops": bool(classification["inactive_effective_noop"]),
                    "canonical_submitted_noops": bool(classification["canonical_submitted_noop"]),
                    "effective_environment_noops": bool(
                        classification["effective_environment_noop"]
                    ),
                    "executed_noops": bool(classification["executed_noop"]),
                    "parse_classification": parse_classification,
                }
                inactive = bool(classification["pre_step_inactive"])
                active_noop_partition = sum(
                    int(turn[key])
                    for key in (
                        "intentional_actionable_noops",
                        "parse_fallback_noops",
                        "active_action_validation_fallback_noops",
                        "active_residual_effective_noops",
                    )
                )
                if (
                    turn["inactive_billed_turns"] != inactive
                    or turn["inactive_effective_noops"] != inactive
                    or (turn["parse_classification"] == "skipped_inactive") != inactive
                    or active_noop_partition
                    != int(turn["canonical_submitted_noops"] and not inactive)
                    or turn["effective_environment_noops"]
                    != (inactive or turn["canonical_submitted_noops"])
                    or turn["executed_noops"] != turn["canonical_submitted_noops"]
                    or (
                        turn["parse_classification"] == "failure"
                        and bool(turn["parse_fallback_noops"])
                        != bool(turn["canonical_submitted_noops"])
                    )
                    or (
                        turn["parse_classification"] == "success"
                        and bool(turn["parse_fallback_noops"])
                    )
                ):
                    raise ValueError(
                        f"{debug_path}: step {step}, agent {agent_idx} has "
                        "inconsistent exact turn accounting"
                    )
                exact_records += 1
            else:
                record_mode = "legacy"
                parsed_action = agent_debug.get("parsed_action")
                parse_failed = _legacy_parse_failed(agent_debug)
                if not isinstance(parsed_action, str) or parse_failed is None:
                    unknown_turns += 1
                    continue
                selected_action = (
                    None
                    if parse_failed
                    else extract_action_multistrategy(agent_debug["llm_raw_output"])
                )
                turn = _classify_reconstructed_turn(
                    pre_step_inactive=pre_step_inactive[agent_idx],
                    selected_action=selected_action,
                    canonical_action=parsed_action,
                    parse_failed=parse_failed,
                )
                legacy_records += 1
            if journal_mode is None:
                journal_mode = record_mode
            elif journal_mode != record_mode:
                raise ValueError(f"{debug_path}: mixes exact and legacy classification records")

            for key in count_fields:
                if key.startswith("action_parse_"):
                    continue
                counts[key] += int(turn[key])
                worker_counts[agent_idx][key] += int(turn[key])
            parse_counter = {
                "success": "action_parse_success",
                "failure": "action_parse_fail",
                "skipped_inactive": "action_parse_skipped_inactive",
            }[turn["parse_classification"]]
            counts[parse_counter] += 1
            worker_counts[agent_idx][parse_counter] += 1
            recovered_turns += 1

            # Legacy cumulative skipped-inactive counts used the post-step
            # observation.  Its per-step increment is therefore the next
            # turn's pre-step inactivity state.
            if record_mode == "legacy":
                stats = parse_stats.get(str(agent_idx))
                if not isinstance(stats, dict):
                    raise ValueError(
                        f"{debug_path}: step {step}, agent {agent_idx} lacks "
                        "legacy parse statistics"
                    )
                skipped = _count(stats.get("skipped_inactive"))
                if skipped is None:
                    raise ValueError(
                        f"{debug_path}: step {step}, agent {agent_idx} lacks a "
                        "legacy skipped-inactive count"
                    )
                delta = skipped - previous_skipped[agent_idx]
                if delta not in (0, 1):
                    raise ValueError(
                        f"{debug_path}: invalid skipped-inactive delta at "
                        f"step {step}, agent {agent_idx}"
                    )
                pre_step_inactive[agent_idx] = delta == 1
                previous_skipped[agent_idx] = skipped

    complete = unknown_turns == 0 and (
        expected_submitted_turns is None or recovered_turns == expected_submitted_turns
    )
    if exact_records and not legacy_records:
        provenance = "debug_jsonl_exact_pre_step"
        accounting_schema = TURN_ACCOUNTING_SCHEMA
        note = "Exact classifications recorded by the corrected evaluator."
    elif legacy_records and not exact_records:
        provenance = "debug_jsonl_legacy_reconstruction"
        accounting_schema = LEGACY_TURN_RECONSTRUCTION_SCHEMA
        note = (
            "Final raw outputs reparsed with the Source parser; pre-step inactivity "
            "shifted from the prior turn's legacy post-step classification; turn zero "
            "uses Alem's all-active reset contract."
        )
    else:
        provenance = "debug_jsonl_mixed_reconstruction"
        accounting_schema = None
        note = "Mixed exact and legacy debug records."
    if not complete:
        note += (
            f" Incomplete reconstruction: recovered={recovered_turns}, "
            f"unknown={unknown_turns}, expected={expected_submitted_turns}."
        )
        for key in counts:
            counts[key] = None
        for agent_counts in worker_counts.values():
            for key in agent_counts:
                agent_counts[key] = None
    parse_attempts = (
        counts["action_parse_success"] + counts["action_parse_fail"] if complete else None
    )
    counts["action_parse_rate"] = (
        counts["action_parse_success"] / parse_attempts
        if parse_attempts is not None and parse_attempts > 0
        else None
    )
    return {
        **counts,
        **{
            f"agent_{agent_idx}_{field}": value
            for agent_idx, agent_counts in worker_counts.items()
            for field, value in agent_counts.items()
        },
        "noop_metrics_provenance": provenance,
        "noop_metrics_complete": complete,
        "noop_metrics_note": note,
        "turn_accounting_schema_version": accounting_schema,
        "turn_accounting_provenance": provenance,
        "turn_accounting_complete": complete,
        "turn_accounting_note": note,
    }


def _episode_noop_metrics(
    payload: dict[str, Any],
    episode_path: Path,
    *,
    num_agents: int,
    expected_submitted_turns: int | None,
) -> dict[str, Any]:
    explicit_fields = {
        "action_parse_success": "action_parse_success",
        "action_parse_fail": "action_parse_fail",
        "action_parse_skipped_inactive": "action_parse_skipped_inactive",
        "intentional_actionable_noops": "intentional_actionable_noop_count",
        "parse_fallback_noops": "parse_fallback_noop_count",
        "active_action_validation_fallback_noops": ("active_action_validation_fallback_noop_count"),
        "active_residual_effective_noops": "active_residual_effective_noop_count",
        "inactive_billed_turns": "inactive_submitted_turn_count",
        "inactive_effective_noops": "inactive_effective_noop_count",
        "canonical_submitted_noops": "canonical_submitted_noop_count",
        "effective_environment_noops": "effective_environment_noop_count",
    }
    explicit = {target: _count(payload.get(source)) for target, source in explicit_fields.items()}
    if "turn_accounting_schema_version" in payload:
        validate_turn_accounting(
            payload,
            context=f"{episode_path}: episode v2 turn accounting",
        )
        if any(value is None for value in explicit.values()):
            # The shared validator should make this unreachable, but retaining
            # the explicit assertion protects the CSV field mapping itself.
            raise ValueError(f"{episode_path}: missing mapped v2 accounting field")
        reconstructed = _debug_noop_metrics(
            episode_path,
            num_agents=num_agents,
            expected_submitted_turns=expected_submitted_turns,
        )
        if (
            reconstructed is None
            or reconstructed.get("turn_accounting_schema_version") != TURN_ACCOUNTING_SCHEMA
            or reconstructed.get("turn_accounting_provenance") != "debug_jsonl_exact_pre_step"
            or reconstructed.get("turn_accounting_complete") is not True
        ):
            raise ValueError(
                f"{episode_path}: finalized v2 episode lacks a complete exact v2 debug journal"
            )
        compared = {
            **explicit,
            "executed_noops": explicit["canonical_submitted_noops"],
        }
        for agent_idx in range(num_agents):
            for target, source in explicit_fields.items():
                worker_suffix = TURN_ACCOUNTING_AGENT_SUFFIXES[source]
                compared[f"agent_{agent_idx}_{target}"] = _count(
                    payload.get(f"agent_{agent_idx}_{worker_suffix}")
                )
            compared[f"agent_{agent_idx}_executed_noops"] = _count(
                payload.get(f"agent_{agent_idx}_executed_noop_count")
            )
        mismatches = [
            field for field, expected in compared.items() if reconstructed.get(field) != expected
        ]
        if mismatches:
            raise ValueError(
                f"{episode_path}: v2 debug journal disagrees with episode/ledger accounting for "
                + ", ".join(mismatches)
            )
        parse_attempts = explicit["action_parse_success"] + explicit["action_parse_fail"]
        return {
            **explicit,
            "executed_noops": explicit["canonical_submitted_noops"],
            "action_parse_rate": (
                explicit["action_parse_success"] / parse_attempts if parse_attempts else None
            ),
            "noop_metrics_provenance": "episode_debug_reconciled_exact_pre_step",
            "noop_metrics_complete": True,
            "noop_metrics_note": (
                "Exact evaluator counters reconciled across episode, attempt ledger, "
                "and per-turn debug journal."
            ),
            "turn_accounting_schema_version": TURN_ACCOUNTING_SCHEMA,
            "turn_accounting_provenance": "episode_debug_reconciled_exact_pre_step",
            "turn_accounting_complete": True,
            "turn_accounting_note": (
                "Versioned aggregate, ledger, and complete exact debug journal agree."
            ),
        }
    reconstructed = _debug_noop_metrics(
        episode_path,
        num_agents=num_agents,
        expected_submitted_turns=expected_submitted_turns,
    )
    if reconstructed is not None:
        return reconstructed
    action_frequency = payload.get("action_frequency")
    canonical_noops = (
        _count(action_frequency.get("Noop")) if isinstance(action_frequency, dict) else None
    )
    actionable_turns = _count(
        _nested(payload, "performance_metrics", "exposure", "actionable_agent_turns")
    )
    inactive_turns = (
        expected_submitted_turns - actionable_turns
        if expected_submitted_turns is not None
        and actionable_turns is not None
        and actionable_turns <= expected_submitted_turns
        else None
    )
    return {
        "intentional_actionable_noops": None,
        "parse_fallback_noops": None,
        "active_action_validation_fallback_noops": None,
        "active_residual_effective_noops": None,
        "inactive_billed_turns": inactive_turns,
        "inactive_effective_noops": inactive_turns,
        "canonical_submitted_noops": canonical_noops,
        "effective_environment_noops": None,
        "executed_noops": canonical_noops,
        "action_parse_success": None,
        "action_parse_fail": None,
        "action_parse_skipped_inactive": None,
        "action_parse_rate": None,
        "noop_metrics_provenance": "episode_aggregate_partial",
        "noop_metrics_complete": False,
        "noop_metrics_note": (
            "No per-turn debug journal or exact episode taxonomy. Aggregate executed "
            "Noops come from action_frequency and mean canonical submitted Noops only; "
            "inactive billed turns come from submitted minus actionable exposure when "
            "available. Corrected parse counts, effective environment Noops, and their "
            "partition are unavailable."
        ),
        "turn_accounting_schema_version": None,
        "turn_accounting_provenance": "unavailable_without_versioned_episode_or_debug",
        "turn_accounting_complete": False,
        "turn_accounting_note": (
            "Legacy aggregate parse statistics are not trusted because they used "
            "post-step inactivity."
        ),
    }


def _validate_episode(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: artifact is not a JSON object")
    reasons = []
    if payload.get("schema_version") != "alem-dice-episode-v1":
        reasons.append("wrong schema_version")
    if payload.get("artifact_status") != "complete":
        reasons.append("artifact_status is not complete")
    if payload.get("error"):
        reasons.append("episode contains error")
    if not payload.get("termination_reason"):
        reasons.append("missing termination_reason")
    if payload.get("early_stop_reason") == "consecutive_length_incomplete_responses":
        reasons.append("provider length-guard early stop")
    if reasons:
        raise ValueError(f"{path}: " + ", ".join(reasons))
    return payload


def _reconciled_episode_return(
    payload: dict[str, Any],
    path: Path,
    *,
    num_agents: int,
    require_trajectory: bool,
) -> float:
    """Recompute the headline return from worker returns and raw rewards."""

    worker_returns = [
        _number(payload.get(f"agent_{worker_id}_return")) for worker_id in range(num_agents)
    ]
    if any(value is None for value in worker_returns):
        raise ValueError(f"{path}: missing finite per-worker episode returns")
    derived_return = sum(float(value) for value in worker_returns) / num_agents
    recorded_return = _number(payload.get("episode_return"))
    if recorded_return is None or not math.isclose(
        float(recorded_return),
        derived_return,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}: episode_return disagrees with per-worker returns")

    if require_trajectory:
        trajectory_path = path.with_name(f"{path.stem}_trajectory.npz")
        try:
            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                rewards = np.asarray(trajectory["rewards"])
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}: cannot read canonical trajectory rewards: {exc}") from exc
        expected_steps = _count(payload.get("num_steps"))
        if (
            rewards.ndim != 2
            or rewards.shape != (expected_steps, num_agents)
            or not np.issubdtype(rewards.dtype, np.number)
            or not np.isfinite(rewards).all()
        ):
            raise ValueError(f"{path}: canonical trajectory rewards have invalid shape/data")
        trajectory_returns = np.sum(rewards, axis=0, dtype=np.float64)
        for worker_id, (trajectory_return, worker_return) in enumerate(
            zip(trajectory_returns, worker_returns, strict=True)
        ):
            if not math.isclose(
                float(trajectory_return),
                float(worker_return),
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError(
                    f"{path}: worker {worker_id} return disagrees with trajectory rewards"
                )
    return derived_return


def _rebuilt_performance_metrics(
    payload: dict[str, Any],
    path: Path,
    *,
    num_agents: int,
) -> tuple[dict[str, Any], bool]:
    """Rebuild performance headlines from raw environment/source counters."""

    recorded = payload.get("performance_metrics")
    used_legacy_fallback = recorded is None
    if recorded is not None and not isinstance(recorded, dict):
        raise ValueError(f"{path}: performance_metrics is not an object")
    if isinstance(recorded, dict) and recorded.get("schema_version") != PERFORMANCE_METRICS_SCHEMA:
        raise ValueError(
            f"{path}: unsupported performance_metrics schema {recorded.get('schema_version')!r}"
        )

    num_steps = _count(payload.get("num_steps"))
    if num_steps is None or num_steps < 1:
        raise ValueError(f"{path}: missing positive episode step count")
    capacity = num_agents * num_steps
    user_info = payload.get("user_info")
    if not isinstance(user_info, dict):
        raise ValueError(f"{path}: missing raw user_info performance counters")

    declares_v2 = payload.get("turn_accounting_schema_version") == TURN_ACCOUNTING_SCHEMA
    raw_steps = _count(payload.get("environment_steps_completed"))
    raw_submitted = _count(payload.get("agent_turns_submitted"))
    raw_alive = _count(payload.get("alive_agent_turns"))
    raw_actionable = _count(payload.get("actionable_agent_turns"))
    if declares_v2 and (
        raw_steps is None or raw_submitted is None or raw_alive is None or raw_actionable is None
    ):
        raise ValueError(f"{path}: finalized v2 episode lacks raw exposure counters")
    if raw_steps is not None and raw_steps != num_steps:
        raise ValueError(f"{path}: raw environment step count disagrees with num_steps")
    if raw_submitted is not None and raw_submitted != capacity:
        raise ValueError(f"{path}: raw submitted-turn count disagrees with physical capacity")

    if raw_alive is None:
        raw_alive = _count(user_info.get("Cooperation/alive_agent_steps"))
    if raw_actionable is None:
        raw_actionable = _count(user_info.get("Cooperation/actionable_agent_steps"))
    if raw_actionable is None:
        inactive_turns = _count(payload.get("inactive_submitted_turn_count"))
        if inactive_turns is None and not declares_v2:
            inactive_turns = _count(payload.get("action_parse_skipped_inactive"))
        if inactive_turns is not None:
            raw_actionable = capacity - inactive_turns
    if raw_alive is None and raw_actionable == capacity:
        # Legacy single-agent summaries omitted cooperation exposure counters;
        # a fully actionable capacity proves every turn was also alive.
        raw_alive = capacity
    if (
        raw_alive is None
        or raw_actionable is None
        or not 0 <= raw_actionable <= raw_alive <= capacity
    ):
        raise ValueError(f"{path}: raw alive/actionable exposure is incomplete or invalid")

    rebuilt = build_performance_metrics(
        payload,
        num_agents,
        environment_steps_completed=num_steps,
        agent_turns_submitted=capacity,
        alive_agent_turns=raw_alive,
        actionable_agent_turns=raw_actionable,
    )
    if recorded is not None and recorded != rebuilt:
        raise ValueError(f"{path}: recorded performance_metrics disagree with rebuilt raw counters")
    return rebuilt, used_legacy_fallback


def episode_row(
    path: Path,
    root: Path,
    *,
    requested_environment_steps: int | None = None,
    manifest: dict[str, Any] | None = None,
    resolved_model_id: str | None = None,
) -> dict[str, Any]:
    """Extract one validated canonical episode into the stable E1 CSV schema."""

    if manifest is not None:
        _require_managed_regular_file(
            path,
            managed_root=root,
            label="canonical episode artifact",
            nonempty=True,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read episode JSON: {exc}") from exc
    payload = _validate_episode(payload, path)

    match = EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path}: filename does not contain a canonical episode index")
    episode_index = int(match.group(1))
    payload_num_agents = _count(payload.get("physical_worker_count"))
    num_agents = payload_num_agents
    path_agents = _path_population(path, root)
    if num_agents is None:
        num_agents = path_agents
    if num_agents is None or num_agents < 1:
        raise ValueError(f"{path}: missing positive physical_worker_count")
    if path_agents is not None and path_agents != num_agents:
        raise ValueError(
            f"{path}: n{path_agents} path disagrees with physical_worker_count={num_agents}"
        )
    retry_accounting = {
        "recovered_transport_retry_count": _count(payload.get("transport_error_count")),
        "transport_error_count": _count(payload.get("transport_error_count")),
        "transport_error_reasons": (
            payload.get("transport_error_reasons")
            if isinstance(payload.get("transport_error_reasons"), dict)
            else {}
        ),
        "retry_accounting_provenance": "unvalidated_without_manifest",
        "retry_accounting_complete": False,
    }
    if manifest is not None:
        if resolved_model_id is None:
            resolved_model_id = _validate_preflight_contract(root, manifest)
        if payload_num_agents is None:
            raise ValueError(f"{path}: manifest-bound episode lacks physical_worker_count")
        expected_path = (
            root
            / f"n{num_agents}"
            / str(CANONICAL_TREATMENT["difficulty"])
            / "alem"
            / "default"
            / f"default_run_{episode_index:02d}.json"
        )
        if path.resolve() != expected_path.resolve() or path.is_symlink():
            raise ValueError(f"{path}: episode path is not canonical for E1")
        _require_episode_companions(path, root=root)
        _validate_episode_attempt_binding(
            payload,
            path,
            episode_index=episode_index,
            root=root,
        )
        retry_accounting = _validate_episode_treatment(
            payload,
            path,
            manifest,
            num_agents=num_agents,
            resolved_model_id=resolved_model_id,
        )
    seed = _count(payload.get("seed"))
    if seed is None:
        raise ValueError(f"{path}: missing non-negative integer seed")
    payload_num_steps = _count(payload.get("num_steps"))
    preliminary_submitted_turns = (
        payload_num_steps * num_agents if payload_num_steps is not None else None
    )
    noop_metrics = _episode_noop_metrics(
        payload,
        path,
        num_agents=num_agents,
        expected_submitted_turns=preliminary_submitted_turns,
    )

    reconciled_episode_return = _reconciled_episode_return(
        payload,
        path,
        num_agents=num_agents,
        require_trajectory=manifest is not None,
    )
    performance, used_legacy_fallback = _rebuilt_performance_metrics(
        payload,
        path,
        num_agents=num_agents,
    )

    exposure = _nested(performance, "exposure") or {}
    paper = _nested(performance, "paper_score_percent") or {}
    coverage = _nested(performance, "achievement_coverage_percent") or {}
    team_unique = _nested(performance, "achievement_first_unlock_count", "team_unique") or {}
    summed_agents = (
        _nested(performance, "achievement_first_unlock_count", "summed_across_agents") or {}
    )
    events = _nested(performance, "event_counters") or {}

    input_tokens = _number(payload.get("input_tokens"))
    cached_input_tokens = _number(payload.get("cached_tokens"))
    if (
        input_tokens is not None
        and cached_input_tokens is not None
        and cached_input_tokens > input_tokens
    ):
        raise ValueError(f"{path}: cached input tokens exceed total input tokens")
    uncached_input_tokens = (
        input_tokens - cached_input_tokens
        if input_tokens is not None and cached_input_tokens is not None
        else None
    )
    output_tokens = _number(payload.get("output_tokens"))
    reasoning_tokens = _number(payload.get("reasoning_tokens"))
    # Provider output-token totals are inclusive of reasoning tokens in the
    # normalized evaluator contract (including OpenAI Responses). Keep the
    # reasoning field as a diagnostic subset; do not bill it twice.
    total_tokens = (
        (input_tokens or 0) + (output_tokens or 0)
        if input_tokens is not None or output_tokens is not None
        else None
    )
    turn_denominator = _number(exposure.get("agent_turns_submitted"))
    if turn_denominator is None:
        turn_denominator = _number(exposure.get("completed_agent_turn_capacity"))
    actionable_turns = _count(exposure.get("actionable_agent_turns"))
    actionable_denominator = actionable_turns
    actual_steps = _count(exposure.get("environment_steps_completed"))
    submitted_turns = _count(exposure.get("agent_turns_submitted"))
    completed_turn_capacity = _count(exposure.get("completed_agent_turn_capacity"))
    classified_turns = _count(exposure.get("classified_action_turns"))
    alive_turns = _count(exposure.get("alive_agent_turns"))
    if payload_num_steps is None or actual_steps != payload_num_steps:
        raise ValueError(f"{path}: exposure steps disagree with episode num_steps")
    expected_capacity = actual_steps * num_agents if actual_steps is not None else None
    if (
        actual_steps is None
        or actual_steps < 1
        or expected_capacity is None
        or completed_turn_capacity != expected_capacity
        or submitted_turns != expected_capacity
        or classified_turns != expected_capacity
    ):
        raise ValueError(f"{path}: incomplete or inconsistent physical-worker exposure")
    if (
        alive_turns is None
        or actionable_turns is None
        or not 0 <= actionable_turns <= alive_turns <= expected_capacity
    ):
        raise ValueError(f"{path}: invalid alive/actionable turn exposure")
    expected_survival = _divide(alive_turns, expected_capacity)
    expected_actionable = _divide(actionable_turns, expected_capacity)
    recorded_survival = _number(exposure.get("survival_fraction"))
    recorded_actionable = _number(exposure.get("actionable_fraction"))
    if (
        recorded_survival is None
        or recorded_actionable is None
        or not math.isclose(float(recorded_survival), expected_survival, abs_tol=1e-12)
        or not math.isclose(float(recorded_actionable), expected_actionable, abs_tol=1e-12)
    ):
        raise ValueError(f"{path}: exposure fractions disagree with canonical turn counts")
    if noop_metrics["noop_metrics_complete"]:
        corrected_classified = sum(
            noop_metrics[field]
            for field in (
                "action_parse_success",
                "action_parse_fail",
                "action_parse_skipped_inactive",
            )
        )
        if submitted_turns is not None and corrected_classified != submitted_turns:
            raise ValueError(
                f"{path}: corrected parse classifications do not cover submitted turns"
            )
        action_frequency = payload.get("action_frequency")
        aggregate_noops = (
            _count(action_frequency.get("Noop")) if isinstance(action_frequency, dict) else None
        )
        if (
            aggregate_noops is not None
            and aggregate_noops != noop_metrics["canonical_submitted_noops"]
        ):
            raise ValueError(f"{path}: canonical submitted Noops disagree with action_frequency")
        if (
            submitted_turns is not None
            and actionable_turns is not None
            and submitted_turns - actionable_turns != noop_metrics["inactive_billed_turns"]
        ):
            raise ValueError(
                f"{path}: reconstructed inactive turns disagree with actionable exposure"
            )
    delivery_bytes = _delivered_bytes(payload)
    user_info = payload.get("user_info")
    if not isinstance(user_info, dict):
        user_info = {}

    return {
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "csv_compatibility_note": (
            "Deprecated survival/actionable/input-token/byte aliases are preserved; "
            "executed_noops is a deprecated alias for canonical_submitted_noops."
        ),
        "artifact_path": str(path.relative_to(root)),
        "attempt_id": payload.get("attempt_id"),
        "num_agents": num_agents,
        "seed": seed,
        "episode_index": episode_index,
        "termination_reason": payload.get("termination_reason"),
        "used_legacy_metric_fallback": used_legacy_fallback,
        "noop_metrics_provenance": noop_metrics["noop_metrics_provenance"],
        "noop_metrics_complete": noop_metrics["noop_metrics_complete"],
        "noop_metrics_note": noop_metrics["noop_metrics_note"],
        "turn_accounting_schema_version": noop_metrics["turn_accounting_schema_version"],
        "turn_accounting_provenance": noop_metrics["turn_accounting_provenance"],
        "turn_accounting_complete": noop_metrics["turn_accounting_complete"],
        "turn_accounting_note": noop_metrics["turn_accounting_note"],
        "retry_accounting_provenance": retry_accounting["retry_accounting_provenance"],
        "retry_accounting_complete": retry_accounting["retry_accounting_complete"],
        "transport_error_reasons_json": json.dumps(
            retry_accounting["transport_error_reasons"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "legacy_recorded_action_parse_rate": _number(payload.get("action_parse_rate")),
        "paper_base_percent": _number(paper.get("base")),
        "paper_coord_percent": _number(paper.get("coord")),
        "paper_total_percent": _number(paper.get("total")),
        "episode_return": reconciled_episode_return,
        "team_unique_base_achievements": _count(team_unique.get("base")),
        "team_unique_coord_achievements": _count(team_unique.get("coord")),
        "team_unique_total_achievements": _count(team_unique.get("total")),
        "summed_agent_base_achievements": _count(summed_agents.get("base")),
        "summed_agent_coord_achievements": _count(summed_agents.get("coord")),
        "summed_agent_total_achievements": _count(summed_agents.get("total")),
        "achievement_base_percent": _number(coverage.get("base")),
        "achievement_coord_percent": _number(coverage.get("coord")),
        "achievement_total_percent": _number(coverage.get("total")),
        "coordination_attempts": _count(events.get("coordination_attempts")),
        "coordination_resolved_attempts": _count(events.get("coordination_resolved_attempts")),
        "coordination_successes": _count(events.get("coordination_successes")),
        "give_attempts": _count(events.get("give_attempts")),
        "successful_transfers": _count(events.get("successful_transfers")),
        "requests": _count(events.get("requests")),
        "revives": _count(events.get("revives")),
        "deaths": _count(user_info.get("Deaths/total_deaths")),
        "action_parse_rate": noop_metrics["action_parse_rate"],
        "action_parse_success": noop_metrics["action_parse_success"],
        "action_parse_fail": noop_metrics["action_parse_fail"],
        "action_parse_skipped_inactive": noop_metrics["action_parse_skipped_inactive"],
        "environment_steps_completed": actual_steps,
        "completed_agent_turn_capacity": completed_turn_capacity,
        "agent_turns_submitted": submitted_turns,
        "classified_action_turns": classified_turns,
        "alive_agent_turns": alive_turns,
        "actionable_agent_turns": actionable_turns,
        "alive_turn_fraction": _number(exposure.get("survival_fraction")),
        "actionable_turn_fraction": _number(exposure.get("actionable_fraction")),
        "requested_environment_steps": requested_environment_steps,
        "step_completion_fraction": _divide(actual_steps, requested_environment_steps),
        "intentional_actionable_noops": noop_metrics["intentional_actionable_noops"],
        "parse_fallback_noops": noop_metrics["parse_fallback_noops"],
        "active_action_validation_fallback_noops": noop_metrics[
            "active_action_validation_fallback_noops"
        ],
        "active_residual_effective_noops": noop_metrics["active_residual_effective_noops"],
        "inactive_billed_turns": noop_metrics["inactive_billed_turns"],
        "inactive_effective_noops": noop_metrics["inactive_effective_noops"],
        "canonical_submitted_noops": noop_metrics["canonical_submitted_noops"],
        "effective_environment_noops": noop_metrics["effective_environment_noops"],
        "executed_noops": noop_metrics["executed_noops"],
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "input_cache_fraction": _divide(cached_input_tokens, input_tokens),
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "model_call_count": _count(payload.get("model_call_count")),
        "provider_request_count": _count(payload.get("provider_request_count")),
        "recovered_transport_retry_count": retry_accounting["recovered_transport_retry_count"],
        "transport_error_count": retry_accounting["transport_error_count"],
        "summed_model_latency_seconds": _number(payload.get("model_latency_seconds")),
        "input_tokens_per_submitted_turn": _divide(input_tokens, turn_denominator),
        "input_tokens_per_actionable_turn": _divide(input_tokens, actionable_denominator),
        "total_tokens_per_submitted_turn": _divide(total_tokens, turn_denominator),
        "total_tokens_per_actionable_turn": _divide(total_tokens, actionable_denominator),
        "delivered_bytes": delivery_bytes,
        "delivered_bytes_per_submitted_turn": _divide(delivery_bytes, turn_denominator),
        "delivered_bytes_per_actionable_turn": _divide(delivery_bytes, actionable_denominator),
        "episode_wall_seconds": _number(payload.get("episode_wall_seconds")),
        "mean_tick_wall_seconds": _number(payload.get("mean_tick_wall_seconds")),
        # Deprecated aliases retained for downstream notebooks using v2 CSVs.
        "survival_fraction": _number(exposure.get("survival_fraction")),
        "actionable_fraction": _number(exposure.get("actionable_fraction")),
        "input_tokens_per_agent_turn": _divide(input_tokens, turn_denominator),
        "delivered_bytes_per_agent_turn": _divide(delivery_bytes, turn_denominator),
    }


def discover_rows(
    root: Path,
    *,
    requested_environment_steps: int | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved_model_id = (
        _validate_preflight_contract(root, manifest) if manifest is not None else None
    )
    canonical_paths = {
        (
            root
            / f"n{population}"
            / CANONICAL_TREATMENT["difficulty"]
            / "alem"
            / "default"
            / f"default_run_{episode_index:02d}.json"
        ).resolve()
        for population in CANONICAL_POPULATIONS
        for episode_index in range(len(CANONICAL_SEEDS))
    }
    discovered = sorted(
        path
        for path in root.rglob("*_run_*.json")
        if "attempt_archive" not in path.relative_to(root).parts
    )
    unexpected = [
        path for path in discovered if path.resolve() not in canonical_paths or path.is_symlink()
    ]
    if unexpected:
        raise ValueError(
            "Non-canonical episode artifact path(s): "
            + ", ".join(str(path.relative_to(root)) for path in unexpected)
        )
    paths = sorted(path for path in discovered if path.resolve() in canonical_paths)
    if not paths:
        raise ValueError(f"No canonical episode JSON files found below {root}")
    if manifest is not None:
        for population in sorted(
            {
                int(POPULATION_PATTERN.fullmatch(path.relative_to(root).parts[0]).group(1))
                for path in paths
            }
        ):
            validate_population_run_binding(root, manifest, population)
    rows = [
        episode_row(
            path,
            root,
            requested_environment_steps=requested_environment_steps,
            manifest=manifest,
            resolved_model_id=resolved_model_id,
        )
        for path in paths
    ]
    identities = [(row["num_agents"], row["seed"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate (population, seed) episode identity")
    sorted_rows = sorted(
        rows,
        key=lambda row: (row["num_agents"], row["seed"], row["episode_index"]),
    )
    if manifest is not None:
        campaign_retry_accounting(root, manifest, sorted_rows)
    return sorted_rows


def campaign_retry_accounting(
    root: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconcile every accepted attempt and enforce the frozen campaign ceilings."""

    expected_attempts_by_population: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("Canonical E1 row lacks its bound attempt_id")
        expected_attempts_by_population[int(row["num_agents"])].add(attempt_id)

    ledger_logical_responses = 0
    ledger_provider_attempts = 0
    ledger_transport_errors = 0
    seen_campaign_attempt_ids = set()
    observed_ledger_populations = set()
    for ledger_path in sorted(root.glob("n*/easy/alem/default/attempt_ledger.jsonl")):
        population_match = POPULATION_PATTERN.fullmatch(ledger_path.relative_to(root).parts[0])
        if population_match is None:
            raise ValueError(f"Non-canonical E1 attempt ledger path: {ledger_path}")
        population = int(population_match.group(1))
        if population not in CANONICAL_POPULATIONS:
            raise ValueError(f"Attempt ledger has an undeclared population: {ledger_path}")
        observed_ledger_populations.add(population)
        _require_managed_regular_file(
            ledger_path,
            managed_root=root,
            label=f"N={population} campaign attempt ledger",
            nonempty=True,
        )
        observed_ids = set()
        try:
            lines = ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"Cannot read campaign attempt ledger {ledger_path}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                ledger_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{ledger_path}:{line_number}: invalid JSON") from exc
            if (
                not isinstance(ledger_row, dict)
                or ledger_row.get("schema_version") != "alem-dice-attempt-v1"
            ):
                raise ValueError(f"{ledger_path}:{line_number}: invalid attempt row")
            attempt_id = ledger_row.get("attempt_id")
            if (
                not isinstance(attempt_id, str)
                or not attempt_id
                or attempt_id in seen_campaign_attempt_ids
            ):
                raise ValueError(f"{ledger_path}:{line_number}: invalid or duplicate attempt_id")
            seen_campaign_attempt_ids.add(attempt_id)
            observed_ids.add(attempt_id)
            if (
                ledger_row.get("artifact_status") != "complete"
                or ledger_row.get("error") is not None
            ):
                raise ValueError(
                    f"{ledger_path}:{line_number}: unrecovered provider/episode failure "
                    "is not canonical E1 evidence"
                )
            logical_responses = _count(ledger_row.get("model_call_count"))
            provider_attempts = _count(ledger_row.get("provider_request_count"))
            transport_errors = _count(ledger_row.get("transport_error_count"))
            reason_counts = _transport_reason_counts(
                ledger_row.get("transport_error_reasons"),
                context=f"{ledger_path}:{line_number}",
            )
            if (
                logical_responses is None
                or provider_attempts is None
                or transport_errors is None
                or provider_attempts != logical_responses + transport_errors
                or sum(reason_counts.values()) != transport_errors
            ):
                raise ValueError(
                    f"{ledger_path}:{line_number}: campaign provider attempts do not reconcile"
                )
            ledger_logical_responses += logical_responses
            ledger_provider_attempts += provider_attempts
            ledger_transport_errors += transport_errors
        if observed_ids != expected_attempts_by_population.get(population, set()):
            raise ValueError(
                f"{ledger_path}: attempt ledger contains missing, failed, or unbound attempts"
            )

    if observed_ledger_populations != set(expected_attempts_by_population):
        raise ValueError("Campaign attempt-ledger populations do not match canonical episodes")

    preflight_attempt_dir = root / "preflight_attempts"
    preflight_attempt_paths = (
        sorted(preflight_attempt_dir.glob("preflight_*.json"))
        if preflight_attempt_dir.is_dir()
        else []
    )
    if [path.name for path in preflight_attempt_paths] != ["preflight_01.json"]:
        raise ValueError("Canonical E1 requires exactly the bound first preflight attempt")
    preflight_attempt = preflight_attempt_paths[0]
    _require_managed_regular_file(
        preflight_attempt,
        managed_root=root,
        label="canonical E1 preflight attempt",
        nonempty=True,
    )
    if _sha256_file(preflight_attempt) != manifest.get("preflight_sha256"):
        raise ValueError("Canonical E1 preflight attempt does not match preflight.json")
    try:
        preflight_payload = json.loads(preflight_attempt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read canonical preflight attempt {preflight_attempt}") from exc
    preflight_logical = _count(preflight_payload.get("logical_response_count"))
    preflight_provider = _count(preflight_payload.get("transport_attempt_count"))
    preflight_errors = _count(preflight_payload.get("transport_error_count"))
    if (
        preflight_logical != 1
        or preflight_provider != 1
        or preflight_errors != 0
        or preflight_payload.get("transport_error_types") != []
    ):
        raise ValueError("Canonical E1 preflight attempt accounting is inconsistent")

    row_logical = sum(int(row["model_call_count"]) for row in rows)
    row_provider = sum(int(row["provider_request_count"]) for row in rows)
    row_errors = sum(int(row["transport_error_count"]) for row in rows)
    if (
        ledger_logical_responses != row_logical
        or ledger_provider_attempts != row_provider
        or ledger_transport_errors != row_errors
    ):
        raise ValueError("Campaign attempt-ledger totals disagree with canonical episode rows")

    total_logical = preflight_logical + ledger_logical_responses
    total_provider = preflight_provider + ledger_provider_attempts
    total_errors = preflight_errors + ledger_transport_errors
    logical_cap = _count(manifest.get("campaign_logical_call_cap"))
    provider_cap = _count(manifest.get("campaign_provider_attempt_cap"))
    if (
        logical_cap is None
        or provider_cap is None
        or total_provider != total_logical + total_errors
        or total_logical > logical_cap
        or total_provider > provider_cap
    ):
        raise ValueError("Observed E1 campaign usage exceeds or violates its frozen call caps")

    reason_totals: dict[str, int] = defaultdict(int)
    provenance_counts: dict[str, int] = defaultdict(int)
    episodes_with_retries = 0
    for row in rows:
        try:
            reasons = json.loads(str(row["transport_error_reasons_json"]))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Canonical E1 row has invalid transport error reason evidence"
            ) from exc
        reasons = _transport_reason_counts(reasons, context="canonical E1 row")
        for reason, count in reasons.items():
            reason_totals[reason] += count
        recovered = _count(row.get("recovered_transport_retry_count"))
        if recovered is None:
            raise ValueError("Canonical E1 row lacks recovered transport retry accounting")
        episodes_with_retries += int(recovered > 0)
        provenance_counts[str(row["retry_accounting_provenance"])] += 1

    return {
        "status": "validated",
        "invariant": "provider_attempts = completed_logical_responses + transport_errors",
        "unrecovered_provider_failure_count": 0,
        "episode_logical_responses": ledger_logical_responses,
        "episode_provider_attempts": ledger_provider_attempts,
        "episode_recovered_transport_retries": ledger_transport_errors,
        "episodes_with_recovered_transport_retries": episodes_with_retries,
        "transport_error_reason_counts": dict(sorted(reason_totals.items())),
        "retry_accounting_provenance_counts": dict(sorted(provenance_counts.items())),
        "preflight_logical_responses": preflight_logical,
        "preflight_provider_attempts": preflight_provider,
        "campaign_logical_responses_observed": total_logical,
        "campaign_provider_attempts_observed": total_provider,
        "campaign_recovered_transport_retries": total_errors,
        "campaign_logical_call_cap": logical_cap,
        "campaign_provider_attempt_cap": provider_cap,
        "campaign_logical_calls_remaining": logical_cap - total_logical,
        "campaign_provider_attempts_remaining": provider_cap - total_provider,
        "within_campaign_caps": True,
    }


def validate_manifest_grid(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    """Enforce the manifest's paired population-by-seed grid."""

    if manifest is None:
        raise ValueError("Study manifest is absent; --allow-incomplete relaxes only missing cells")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Study manifest has unsupported schema_version")
    if manifest.get("profile") != CANONICAL_TREATMENT["profile"]:
        raise ValueError("Study manifest has wrong E1 profile")
    populations = manifest.get("planned_counts")
    seeds = manifest.get("seeds")
    if populations != CANONICAL_POPULATIONS:
        raise ValueError("Study manifest has invalid planned_counts")
    if seeds != CANONICAL_SEEDS:
        raise ValueError("Study manifest has invalid seeds")
    populations = [int(value) for value in populations]
    seeds = [int(value) for value in seeds]
    if len(populations) != len(set(populations)) or len(seeds) != len(set(seeds)):
        raise ValueError("Study manifest population and seed declarations must be unique")
    episodes_per_count = _count(manifest.get("episodes_per_count"))
    if episodes_per_count != len(seeds):
        raise ValueError("Study manifest episodes_per_count disagrees with its declared seed count")

    expected = {(population, seed) for population in populations for seed in seeds}
    observed = {(int(row["num_agents"]), int(row["seed"])) for row in rows}
    seed_episode_index = {seed: index for index, seed in enumerate(seeds)}
    mismatched_indices = [
        {
            "num_agents": int(row["num_agents"]),
            "seed": int(row["seed"]),
            "episode_index": row.get("episode_index"),
            "expected_episode_index": seed_episode_index.get(int(row["seed"])),
        }
        for row in rows
        if int(row["seed"]) in seed_episode_index
        and row.get("episode_index") != seed_episode_index[int(row["seed"])]
    ]
    if mismatched_indices:
        raise ValueError(f"Episode indices disagree with manifest seed order: {mismatched_indices}")
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(f"Artifacts outside manifest-declared grid: {unexpected}")
    missing = sorted(expected - observed)
    complete = not missing
    if missing and not allow_incomplete:
        raise ValueError(
            "Manifest-declared paired grid is incomplete; missing "
            f"{missing}. Re-run with --allow-incomplete only for a watermarked interim analysis."
        )
    watermark = None
    if missing:
        watermark = (
            "INCOMPLETE INTERIM ANALYSIS — "
            f"{len(observed)}/{len(expected)} manifest-declared population/seed pairs present"
        )
    return {
        "manifest_present": True,
        "complete": complete,
        "expected_pair_count": len(expected),
        "observed_pair_count": len(observed),
        "planned_populations": populations,
        "planned_seeds": seeds,
        "missing_pairs": [{"num_agents": population, "seed": seed} for population, seed in missing],
        "watermark": watermark,
    }


def _validate_bootstrap_reps(reps: int) -> None:
    if reps < MIN_BOOTSTRAP_REPS:
        raise ValueError(
            f"bootstrap repetitions must be at least {MIN_BOOTSTRAP_REPS} "
            "for stable percentile endpoints"
        )


def bootstrap_mean_ci(
    values: list[int | float],
    *,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, int | float | None]:
    _validate_bootstrap_reps(reps)
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = float(array.mean())
    if array.size == 1:
        return {"n": 1, "mean": mean, "ci_low": mean, "ci_high": mean}
    indices = rng.integers(0, array.size, size=(reps, array.size))
    bootstrapped = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrapped, [0.025, 0.975])
    return {
        "n": int(array.size),
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    reps: int,
    bootstrap_seed: int,
) -> dict[int, dict[str, dict[str, int | float | None]]]:
    by_population: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_population[row["num_agents"]].append(row)
    rng = np.random.default_rng(bootstrap_seed)
    return {
        num_agents: {
            metric: bootstrap_mean_ci(
                [
                    value
                    for row in population_rows
                    if (value := _number(row.get(metric))) is not None
                ],
                reps=reps,
                rng=rng,
            )
            for metric in SUMMARY_METRICS
        }
        for num_agents, population_rows in sorted(by_population.items())
    }


def paired_population_contrasts(
    rows: list[dict[str, Any]],
    *,
    reps: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    """Compute higher-minus-lower contrasts by resampling common seed IDs."""

    indexed = {(int(row["num_agents"]), int(row["seed"])): row for row in rows}
    populations = sorted({int(row["num_agents"]) for row in rows})
    rng = np.random.default_rng(bootstrap_seed)
    contrasts = {}
    for lower, higher in combinations(populations, 2):
        common_seeds = sorted(
            {
                seed
                for population, seed in indexed
                if population == lower and (higher, seed) in indexed
            }
        )
        metrics = {}
        for metric in SUMMARY_METRICS:
            differences = []
            metric_seeds = []
            for seed in common_seeds:
                lower_value = _number(indexed[(lower, seed)].get(metric))
                higher_value = _number(indexed[(higher, seed)].get(metric))
                if lower_value is None or higher_value is None:
                    continue
                differences.append(float(higher_value) - float(lower_value))
                metric_seeds.append(seed)
            estimate = bootstrap_mean_ci(differences, reps=reps, rng=rng)
            metrics[metric] = {
                "common_seed_count": estimate.pop("n"),
                "common_seeds": metric_seeds,
                "mean_difference": estimate.pop("mean"),
                **estimate,
            }
        contrasts[f"n{higher}_minus_n{lower}"] = {
            "lower_population": lower,
            "higher_population": higher,
            "common_seeds": common_seeds,
            "metrics": metrics,
        }
    return contrasts


def termination_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[str, int] = defaultdict(int)
    by_population: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        reason = str(row["termination_reason"])
        overall[reason] += 1
        by_population[int(row["num_agents"])][reason] += 1
    return {
        "overall": dict(sorted(overall.items())),
        "by_population": {
            str(population): dict(sorted(counts.items()))
            for population, counts in sorted(by_population.items())
        },
    }


def _format_value(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{float(number):.{digits}f}"


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[int, dict[str, dict[str, Any]]],
    root: Path,
    *,
    grid: dict[str, Any],
    contrasts: dict[str, dict[str, Any]],
    terminations: dict[str, Any],
    retry_accounting: dict[str, Any],
) -> None:
    lines = [
        "# E1 Source Scaling Summary",
        "",
        f"Source root: `{root}`",
        "",
    ]
    if grid.get("watermark"):
        lines.extend((f"> **{grid['watermark']}**", ""))
    lines.extend(
        [
            "| Agents | Seeds | Total % | Base % | Coord % | Per-agent return | "
            "Unique team first-unlocks | Summed agent first-unlocks | Alive-turn fraction |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    counts = defaultdict(int)
    for row in rows:
        counts[row["num_agents"]] += 1
    for num_agents, metrics in summary.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(num_agents),
                    str(counts[num_agents]),
                    _format_value(metrics["paper_total_percent"]["mean"]),
                    _format_value(metrics["paper_base_percent"]["mean"]),
                    _format_value(metrics["paper_coord_percent"]["mean"]),
                    _format_value(metrics["episode_return"]["mean"]),
                    _format_value(metrics["team_unique_total_achievements"]["mean"]),
                    _format_value(metrics["summed_agent_total_achievements"]["mean"]),
                    _format_value(metrics["alive_turn_fraction"]["mean"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Intervals in `summary.json` and the figure resample seed IDs within each "
            "population. With three canonical seeds they are coarse descriptive uncertainty "
            "intervals, not hypothesis tests. Raw seed values are retained in `episodes.csv`.",
            "",
            "Base/Coord/Total are reward-weighted paper scores on a 0–100 scale. "
            "Achievement counts are cumulative binary first-unlock types, not repeated events. "
            "Team-unique counts each achievement type once across the team; the summed "
            "agent count can count the same type once for every attaining agent.",
            "",
            "## Exposure, cache, and time",
            "",
            "| Agents | Actual/requested ticks | Alive-turn fraction | "
            "Actionable-turn fraction | "
            "Cached input | Uncached input | Cache fraction | Episode wall (s) | "
            "Summed model latency (s) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for num_agents, metrics in summary.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(num_agents),
                    f"{_format_value(metrics['environment_steps_completed']['mean'], 1)}/"
                    f"{_format_value(metrics['requested_environment_steps']['mean'], 1)}",
                    _format_value(metrics["alive_turn_fraction"]["mean"], 3),
                    _format_value(metrics["actionable_turn_fraction"]["mean"], 3),
                    _format_value(metrics["cached_input_tokens"]["mean"], 1),
                    _format_value(metrics["uncached_input_tokens"]["mean"], 1),
                    _format_value(metrics["input_cache_fraction"]["mean"], 3),
                    _format_value(metrics["episode_wall_seconds"]["mean"], 2),
                    _format_value(metrics["summed_model_latency_seconds"]["mean"], 2),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Summed model latency adds per-call latency and can exceed episode wall time "
            "because physical-agent requests overlap within a tick.",
            "",
            "| Agents | Input tokens/submitted | Input tokens/actionable | "
            "Total tokens/submitted | Total tokens/actionable | "
            "Broadcast bytes/submitted | Broadcast bytes/actionable |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for num_agents, metrics in summary.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(num_agents),
                    _format_value(metrics["input_tokens_per_submitted_turn"]["mean"], 2),
                    _format_value(metrics["input_tokens_per_actionable_turn"]["mean"], 2),
                    _format_value(metrics["total_tokens_per_submitted_turn"]["mean"], 2),
                    _format_value(metrics["total_tokens_per_actionable_turn"]["mean"], 2),
                    _format_value(metrics["delivered_bytes_per_submitted_turn"]["mean"], 2),
                    _format_value(metrics["delivered_bytes_per_actionable_turn"]["mean"], 2),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Byte rates are Source ordinary peer-broadcast delivered fan-out bytes, not "
            "serialized prompt bytes or provider network traffic.",
            "",
            "## Provider retry audit",
            "",
            "| Agents | Logical responses | Provider attempts | Recovered retries | "
            "Episodes with retries |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for num_agents in sorted(summary):
        population_rows = [row for row in rows if row["num_agents"] == num_agents]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(num_agents),
                    str(sum(int(row["model_call_count"]) for row in population_rows)),
                    str(sum(int(row["provider_request_count"]) for row in population_rows)),
                    str(
                        sum(int(row["recovered_transport_retry_count"]) for row in population_rows)
                    ),
                    str(
                        sum(
                            int(row["recovered_transport_retry_count"] > 0)
                            for row in population_rows
                        )
                    ),
                )
            )
            + " |"
        )
    reason_counts = retry_accounting["transport_error_reason_counts"]
    rendered_reasons = (
        ", ".join(f"`{reason}`={count}" for reason, count in reason_counts.items())
        if reason_counts
        else "none"
    )
    lines.extend(
        [
            "",
            f"Recovered transport error types: {rendered_reasons}. "
            f"Campaign usage including preflight is "
            f"{retry_accounting['campaign_logical_responses_observed']}/"
            f"{retry_accounting['campaign_logical_call_cap']} logical responses and "
            f"{retry_accounting['campaign_provider_attempts_observed']}/"
            f"{retry_accounting['campaign_provider_attempt_cap']} provider attempts.",
            "",
            "A recovered retry is accepted only when every logical decision has one final "
            "completed response and `provider_attempts = logical_responses + "
            "transport_errors` reconciles globally, by worker, and against the attempt "
            "ledger. Unrecovered failures, incomplete responses, untyped errors, and "
            "unaccounted attempts remain disqualifying.",
            "",
            "## Noop audit",
            "",
            "| Agents | Intentional actionable | Parse fallback | Validation fallback | "
            "Inactive effective | Residual | Canonical submitted | Effective environment |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for num_agents, metrics in summary.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(num_agents),
                    _format_value(metrics["intentional_actionable_noops"]["mean"], 2),
                    _format_value(metrics["parse_fallback_noops"]["mean"], 2),
                    _format_value(metrics["active_action_validation_fallback_noops"]["mean"], 2),
                    _format_value(metrics["inactive_effective_noops"]["mean"], 2),
                    _format_value(metrics["active_residual_effective_noops"]["mean"], 2),
                    _format_value(metrics["canonical_submitted_noops"]["mean"], 2),
                    _format_value(metrics["effective_environment_noops"]["mean"], 2),
                )
            )
            + " |"
        )
    provenance_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        provenance_counts[str(row["noop_metrics_provenance"])] += 1
    lines.extend(
        [
            "",
            "Noop provenance: "
            + ", ".join(
                f"`{provenance}`={count}" for provenance, count in sorted(provenance_counts.items())
            )
            + ". Per-episode reconstruction notes are retained in `episodes.csv`.",
            "",
            "Canonical submitted Noops are the post-validation actions passed to `env.step` "
            "and are the only values compared with legacy `action_frequency.Noop`. Effective "
            "environment Noops additionally include every inactive worker turn, because Alem "
            "masks those actions to Noop. The five cause columns form an exact partition of "
            "effective environment Noops when accounting is complete.",
            "",
            "## Terminations",
            "",
        ]
    )
    for population, counts_by_reason in terminations["by_population"].items():
        rendered = ", ".join(f"{reason}={count}" for reason, count in counts_by_reason.items())
        lines.append(f"- N={population}: {rendered}")
    lines.extend(
        [
            "",
            "Actual tick counts are reported separately from the manifest-requested tick "
            "cap; termination reasons are never silently treated as full-horizon runs.",
            "",
            "## Paired population contrasts",
            "",
            "| Contrast | Common seeds | Reward-weighted Total difference (points) | "
            "95% descriptive interval |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, contrast in contrasts.items():
        metric = contrast["metrics"]["paper_total_percent"]
        lines.append(
            f"| {name} | {metric['common_seed_count']} | "
            f"{_format_value(metric['mean_difference'], 3)} | "
            f"[{_format_value(metric['ci_low'], 3)}, "
            f"{_format_value(metric['ci_high'], 3)}] |"
        )
    lines.extend(
        [
            "",
            "Contrasts are higher-population minus lower-population values on common seed "
            "IDs and use a paired seed bootstrap. Three-seed intervals remain descriptive "
            "and coarse.",
            "",
            "Coordination event totals are canonical environment counters, but their "
            "subdomains use heterogeneous counting units (for example per-timestep sync "
            "attempts versus per-setup handovers). Do not add the event fields together. "
            "Coordination is undefined for the N=1 wrapper and remains missing rather than zero.",
            "",
            "The curve is descriptive: changing population also changes spawn geometry, "
            "mob pressure, specialization balance, prompt size, and coordination requirements.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_table(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[int, dict[str, dict[str, Any]]],
    *,
    grid: dict[str, Any],
) -> None:
    """Write a dependency-light table fragment using the exact JSON estimates."""

    counts = defaultdict(int)
    for row in rows:
        counts[row["num_agents"]] += 1

    def _tex_value(value: Any, digits: int) -> str:
        number = _number(value)
        return "--" if number is None else f"{float(number):.{digits}f}"

    caption_prefix = "INCOMPLETE interim analysis. " if grid.get("watermark") else ""
    lines = [
        f"% {grid['watermark']}" if grid.get("watermark") else "% Complete paired grid.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{r r r r r r}",
        r"\hline",
        r"$N$ & Seeds & Base reward \% & Coord. reward \% & Total reward \% & "
        r"Return/agent \\",
        r"\hline",
    ]
    for num_agents, metrics in summary.items():
        lines.append(
            " & ".join(
                (
                    str(num_agents),
                    str(counts[num_agents]),
                    _tex_value(metrics["paper_base_percent"]["mean"], 2),
                    _tex_value(metrics["paper_coord_percent"]["mean"], 2),
                    _tex_value(metrics["paper_total_percent"]["mean"], 2),
                    _tex_value(metrics["episode_return"]["mean"], 3),
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\hline",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{tabular}{r r r}",
            r"\hline",
            r"$N$ & Unique team first-unlocks & $\sum$ agent first-unlocks \\",
            r"\hline",
        )
    )
    for num_agents, metrics in summary.items():
        lines.append(
            " & ".join(
                (
                    str(num_agents),
                    _tex_value(metrics["team_unique_total_achievements"]["mean"], 2),
                    _tex_value(metrics["summed_agent_total_achievements"]["mean"], 2),
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\hline",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{tabular}{r r r r r}",
            r"\hline",
            r"$N$ & Ticks (actual/cap) & Tick wall (s) & "
            r"Alive frac. & Actionable frac. \\",
            r"\hline",
        )
    )
    for num_agents, metrics in summary.items():
        lines.append(
            " & ".join(
                (
                    str(num_agents),
                    (
                        _tex_value(metrics["environment_steps_completed"]["mean"], 1)
                        + "/"
                        + _tex_value(metrics["requested_environment_steps"]["mean"], 1)
                    ),
                    _tex_value(metrics["mean_tick_wall_seconds"]["mean"], 3),
                    _tex_value(metrics["alive_turn_fraction"]["mean"], 3),
                    _tex_value(metrics["actionable_turn_fraction"]["mean"], 3),
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption_prefix}Source-baseline population screen. Entries are "
            r"seed means; score columns are reward-weighted paper scores. Coordination "
            r"is undefined for the single-agent wrapper. Three-seed intervals in the "
            r"companion artifacts are coarse and descriptive.}",
            r"\label{tab:e1-source-scaling}",
            r"\end{table}",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[int, dict[str, dict[str, Any]]],
    *,
    grid: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    populations = sorted(summary)
    rng = np.random.default_rng(4317)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for axis, (metric, label) in zip(axes, HEADLINE_METRICS, strict=True):
        for num_agents in populations:
            raw_values = [
                float(value)
                for row in rows
                if row["num_agents"] == num_agents
                and (value := _number(row.get(metric))) is not None
            ]
            if raw_values:
                jitter = rng.uniform(-0.055, 0.055, len(raw_values))
                axis.scatter(
                    np.asarray([num_agents] * len(raw_values)) + jitter,
                    raw_values,
                    color="#4472C4",
                    alpha=0.6,
                    s=25,
                    zorder=2,
                )
        means = [summary[n][metric]["mean"] for n in populations]
        lows = [summary[n][metric]["ci_low"] for n in populations]
        highs = [summary[n][metric]["ci_high"] for n in populations]
        valid = [
            index
            for index, (mean, low, high) in enumerate(zip(means, lows, highs, strict=True))
            if mean is not None and low is not None and high is not None
        ]
        if valid:
            x_values = np.asarray([populations[index] for index in valid], dtype=float)
            y_values = np.asarray([means[index] for index in valid], dtype=float)
            lower = y_values - np.asarray([lows[index] for index in valid], dtype=float)
            upper = np.asarray([highs[index] for index in valid], dtype=float) - y_values
            axis.errorbar(
                x_values,
                y_values,
                yerr=np.vstack((lower, upper)),
                color="#17365D",
                marker="o",
                linewidth=1.7,
                capsize=4,
                zorder=3,
            )
        axis.set_xlabel("Physical agents")
        axis.set_ylabel(label)
        axis.set_xticks(
            populations,
            [
                f"{n}\nsolo" if n == 1 else (f"{n}\nextension" if n == 6 else str(n))
                for n in populations
            ],
        )
        axis.grid(axis="y", alpha=0.25)
    title = "Alem Source baseline: fixed-world population curve"
    if grid.get("watermark"):
        title += "\nINCOMPLETE INTERIM ANALYSIS"
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="E1 output root containing n<N> arms")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Results/e1_source_scaling"),
        help="output directory (default: Results/e1_source_scaling)",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=8675309)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Permit a partial manifest-declared population/seed grid. Outputs are "
            "explicitly watermarked as an incomplete interim analysis."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_bootstrap_reps(args.bootstrap_reps)
    root = args.run_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Run root does not exist: {root}")
    manifest = _read_manifest(root)
    treatment_binding = validate_manifest_contract(root, manifest)
    requested_steps = (
        _count(manifest.get("max_steps_per_episode")) if manifest is not None else None
    )
    if manifest is not None and (requested_steps is None or requested_steps < 1):
        raise ValueError("Study manifest has invalid max_steps_per_episode")
    rows = discover_rows(
        root,
        requested_environment_steps=requested_steps,
        manifest=manifest,
    )
    retry_accounting = campaign_retry_accounting(root, manifest, rows)
    grid = validate_manifest_grid(rows, manifest, allow_incomplete=args.allow_incomplete)
    summary = summarize_rows(
        rows,
        reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    contrasts = paired_population_contrasts(
        rows,
        reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed + 1,
    )
    terminations = termination_counts(rows)
    noop_provenance_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        noop_provenance_counts[str(row["noop_metrics_provenance"])] += 1

    analysis_status = "complete" if grid.get("complete") is True else "incomplete_interim"
    analysis_watermark = grid.get("watermark")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "analysis_status": analysis_status,
                "analysis_watermark": analysis_watermark,
            }
            for row in rows
        )
    with (args.out / "paired_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "contrast",
                "lower_population",
                "higher_population",
                "metric",
                "common_seed_count",
                "common_seeds",
                "mean_difference",
                "ci_low",
                "ci_high",
                "analysis_status",
                "analysis_watermark",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for name, contrast in contrasts.items():
            for metric, estimate in contrast["metrics"].items():
                writer.writerow(
                    {
                        "contrast": name,
                        "lower_population": contrast["lower_population"],
                        "higher_population": contrast["higher_population"],
                        "metric": metric,
                        "common_seed_count": estimate["common_seed_count"],
                        "common_seeds": ",".join(str(seed) for seed in estimate["common_seeds"]),
                        "mean_difference": estimate["mean_difference"],
                        "ci_low": estimate["ci_low"],
                        "ci_high": estimate["ci_high"],
                        "analysis_status": analysis_status,
                        "analysis_watermark": analysis_watermark,
                    }
                )
    summary_payload = {
        "schema_version": "alem-dice-e1-scaling-summary-v4",
        "episodes_csv_schema_version": CSV_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_root": str(root),
        "analysis_status": analysis_status,
        "watermark": analysis_watermark,
        "grid_validation": grid,
        "treatment_binding": {
            "status": "validated",
            "manifest_schema": MANIFEST_SCHEMA,
            "profile": CANONICAL_TREATMENT["profile"],
            "source_commit": treatment_binding["source_commit"],
            "uv_lock_sha256": treatment_binding["uv_lock_sha256"],
            "observed_population_run_bindings": sorted({int(row["num_agents"]) for row in rows}),
        },
        "episode_count": len(rows),
        "termination_counts": terminations,
        "provider_retry_accounting": retry_accounting,
        "noop_reconstruction": {
            "all_episode_taxonomies_complete": all(
                bool(row["noop_metrics_complete"]) for row in rows
            ),
            "provenance_counts": dict(sorted(noop_provenance_counts.items())),
            "unrecoverable_without_debug": (
                "For legacy episodes without a complete per-turn debug journal, corrected "
                "parse counts and effective-environment Noop causes are unavailable. "
                "action_frequency.Noop recovers canonical submitted Noops only; it must "
                "not be interpreted as effective environment execution."
            ),
        },
        "turn_accounting": {
            "canonical_schema_version": TURN_ACCOUNTING_SCHEMA,
            "legacy_reconstruction_schema_version": LEGACY_TURN_RECONSTRUCTION_SCHEMA,
            "all_episodes_complete": all(bool(row["turn_accounting_complete"]) for row in rows),
            "semantics": TURN_ACCOUNTING_SEMANTICS,
            "legacy_parse_override": (
                "When debug reconstruction is complete, action_parse_rate and its "
                "success/fail/skipped counts use corrected pre-step inactivity. The "
                "biased episode value is retained only as legacy_recorded_action_parse_rate."
            ),
        },
        "populations": {
            str(num_agents): {
                "seed_count": sum(row["num_agents"] == num_agents for row in rows),
                "termination_counts": terminations["by_population"].get(str(num_agents), {}),
                "metrics": metrics,
            }
            for num_agents, metrics in summary.items()
        },
        "paired_population_contrasts": contrasts,
        "bootstrap": {
            "unit": "seed",
            "population_method": "seed-ID resampling within population",
            "contrast_method": (
                "paired seed-ID resampling of higher-population minus "
                "lower-population differences on common seeds"
            ),
            "confidence": 0.95,
            "repetitions": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "contrast_seed": args.bootstrap_seed + 1,
            "interpretation": (
                "With three canonical seeds, intervals are coarse descriptive "
                "uncertainty summaries and are not hypothesis tests."
            ),
        },
        "metric_semantics": {
            "paper_score_percent": "Reward-weighted Base/Coord/Total paper score, 0–100.",
            "achievement_counts": (
                "Cumulative binary first-unlock types, not repeated completion events."
            ),
            "alive_turn_fraction": "alive_agent_turns / completed_agent_turn_capacity.",
            "actionable_turn_fraction": ("actionable_agent_turns / completed_agent_turn_capacity."),
            "cached_input_tokens": ("Provider-reported cached input-token subset of input_tokens."),
            "uncached_input_tokens": "input_tokens - cached_input_tokens.",
            "total_tokens": (
                "input_tokens + provider output_tokens; reasoning_tokens are a subset "
                "of output_tokens and are not added twice."
            ),
            "summed_model_latency_seconds": (
                "Sum of per-call latency; may exceed episode wall time when calls overlap."
            ),
            "recovered_transport_retry_count": (
                "A retryable provider transport error followed by one final completed "
                "logical response; equal to provider attempts minus logical responses."
            ),
            "delivered_bytes": (
                "Source worker_peer delivered fan-out bytes, not provider network bytes."
            ),
            "inactive_billed_turns": (
                "Submitted model turns whose worker was inactive in the pre-step observation."
            ),
            "canonical_submitted_noops": TURN_ACCOUNTING_SEMANTICS["canonical_submitted_noop"],
            "effective_environment_noops": TURN_ACCOUNTING_SEMANTICS["effective_environment_noop"],
            "executed_noops": ("Deprecated CSV compatibility alias for canonical_submitted_noops."),
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.out / "summary.md",
        rows,
        summary,
        root,
        grid=grid,
        contrasts=contrasts,
        terminations=terminations,
        retry_accounting=retry_accounting,
    )
    write_tex_table(args.out / "summary_table.tex", rows, summary, grid=grid)
    write_plot(args.out / "performance_vs_agents.png", rows, summary, grid=grid)
    status = "complete" if grid.get("complete") is True else "WATERMARKED INCOMPLETE"
    print(
        f"Wrote {len(rows)} seeds across {len(summary)} populations "
        f"({status}) to {args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
