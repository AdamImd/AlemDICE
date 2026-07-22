#!/usr/bin/env python3
"""Summarize the paired three-seed, 200-step coordination signal study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ARMS = ("free_thinking", "free_concise", "cohesion_concise")
EXPECTED_SEEDS = (9999, 10000, 10001)
MAX_STEPS = 200
CONTRASTS = {
    "response_mode": ("free_concise", "free_thinking"),
    "cohesion": ("cohesion_concise", "free_concise"),
}
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "model_call_count",
    "provider_request_count",
    "transport_error_count",
    "model_latency_seconds",
    "episode_wall_seconds",
)
OUTCOME_CONTRAST_FIELDS = (
    "num_steps",
    "episode_return",
    "total_achievements",
    "coordination_attempts",
    "coordination_successes",
    "action_parse_rate",
    "delivery_bytes",
    "input_tokens",
    "output_tokens",
    "model_call_count",
    "total_tokens",
    "episode_wall_seconds",
    "input_tokens_per_model_call",
    "output_tokens_per_model_call",
    "wall_seconds_per_step",
    "delivery_bytes_per_step",
)
MATCHED_PREFIX_FIELDS = (
    "classification",
    "eligible_for_registered_efficacy",
    "seed",
    "steps",
    "matched_steps_complete",
    "concise_reward_sum",
    "thinking_reward_sum",
    "concise_parse_rate",
    "thinking_parse_rate",
    "concise_output_tokens_per_call",
    "thinking_output_tokens_per_call",
    "concise_mean_latency_seconds",
    "thinking_mean_latency_seconds",
    "concise_length_stops",
    "thinking_length_stops",
    "concise_empty_outputs",
    "thinking_empty_outputs",
    "concise_source_path",
    "thinking_source_path",
)
DATA_INTEGRITY_REASONS = {
    "missing_artifact",
    "malformed_json",
    "artifact_not_object",
    "wrong_schema_version",
    "missing_seed",
    "seed_mismatch",
    "duplicate_seed",
}


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _usage(payload: dict[str, Any] | None) -> dict[str, int | float | None]:
    if payload is None:
        return {field: None for field in USAGE_FIELDS}
    return {field: _number(payload.get(field)) for field in USAGE_FIELDS}


def _sum_usage(
    records: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    if not records:
        return {field: None for field in USAGE_FIELDS}
    return {
        field: sum((_number(record.get(field)) or 0) for record in records)
        for field in USAGE_FIELDS
    }


def _valid_attempt_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    episode_index = record.get("episode_index")
    if (
        record.get("schema_version") != "alem-dice-attempt-v1"
        or record.get("artifact_status") not in {"complete", "failed"}
        or not record.get("termination_reason")
        or isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        return False
    attempt_id = record.get("attempt_id")
    return "attempt_id" not in record or (isinstance(attempt_id, str) and bool(attempt_id.strip()))


def _load_attempts(
    task_dir: Path,
    arm: str,
) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    ledger = task_dir / "attempt_ledger.jsonl"
    if not ledger.is_file():
        return attempts, warnings
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return attempts, [f"{arm}: could not read attempt ledger: {exc}"]
    seen_attempt_ids: set[tuple[int, str]] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"{arm}: malformed ledger row {line_number}: {exc}")
            continue
        if not _valid_attempt_record(record):
            warnings.append(f"{arm}: invalid ledger row {line_number}")
            continue
        episode_index = record["episode_index"]
        if episode_index >= len(EXPECTED_SEEDS):
            warnings.append(
                f"{arm}: unexpected ledger episode index {episode_index} at row {line_number}"
            )
            continue
        attempt_id = record.get("attempt_id")
        if attempt_id is not None:
            key = (episode_index, attempt_id)
            if key in seen_attempt_ids:
                warnings.append(
                    f"{arm}: duplicate attempt id {attempt_id!r} for episode {episode_index}"
                )
                continue
            seen_attempt_ids.add(key)
        attempts[episode_index].append(record)
    return attempts, warnings


def _invalid_reasons(payload: dict[str, Any], expected_seed: int) -> list[str]:
    reasons = []
    if payload.get("schema_version") != "alem-dice-episode-v1":
        reasons.append("wrong_schema_version")
    if payload.get("artifact_status") != "complete":
        reasons.append("artifact_not_complete")
    if bool(payload.get("error")):
        reasons.append("episode_error")
    if not payload.get("termination_reason"):
        reasons.append("missing_termination_reason")
    if payload.get("early_stop_reason") == "consecutive_length_incomplete_responses":
        reasons.append("length_incomplete_early_stop")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        reasons.append("missing_seed")
    elif seed != expected_seed:
        reasons.append("seed_mismatch")
    return reasons


def _delivered_bytes(communication: Any) -> int | float | None:
    if not isinstance(communication, dict):
        return None
    total: int | float = 0
    for channel in communication.values():
        if isinstance(channel, dict):
            total += _number(channel.get("delivery_bytes")) or 0
    return total


def _divide(
    numerator: int | float | None,
    denominator: int | float | None,
) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _empty_partial_debug(relative_path: Path) -> dict[str, Any]:
    """Return the fixed schema used for interrupted debug-only episodes.

    Debug JSONL is written after every environment step, while the canonical
    episode JSON is only finalized when an episode exits normally.  Therefore
    this telemetry can document work completed before an interruption, but it
    must never make an episode valid or enter an efficacy contrast.
    """

    return {
        "available": False,
        "classification": "partial_non_efficacy",
        "eligible_for_efficacy": False,
        "source_path": str(relative_path),
        "debug_record_count": 0,
        "observed_steps": 0,
        "observed_step_values": [],
        "first_step": None,
        "last_step": None,
        "reward_sum": None,
        "model_call_count": 0,
        "provider_request_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model_latency_seconds": 0.0,
        "mean_model_latency_seconds": None,
        "length_stop_count": 0,
        "empty_output_count": 0,
        "action_parse_success": None,
        "action_parse_fail": None,
        "action_parse_rate": None,
        "transport_error_count": 0,
        "communications": {
            "emitted_messages": 0,
            "delivered_messages": 0,
            "payload_bytes": 0,
            "delivery_bytes": 0,
        },
    }


def _is_length_stop(agent: dict[str, Any]) -> bool:
    values = (
        agent.get("stop_reason"),
        agent.get("incomplete_reason"),
        agent.get("provider_incomplete_reason"),
    )
    markers = ("length", "max_completion", "max_output", "max_token")
    return any(
        any(marker in str(value).strip().lower() for marker in markers)
        for value in values
        if value is not None
    )


def _partial_debug_metrics(
    debug_path: Path,
    relative_path: Path,
    *,
    allowed_steps: set[int] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Extract conservative, interruption-safe telemetry from a debug JSONL."""

    metrics = _empty_partial_debug(relative_path)
    warnings: list[str] = []
    if not debug_path.is_file():
        return metrics, warnings

    metrics["available"] = True
    steps: set[int] = set()
    latency_observations = 0
    latest_parse_step: int | None = None
    latest_parse_stats: dict[str, Any] | None = None
    reward_sum = 0.0

    try:
        handle = debug_path.open(encoding="utf-8")
    except OSError as exc:
        metrics["available"] = False
        warnings.append(f"could not read partial debug JSONL: {exc}")
        return metrics, warnings

    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"malformed partial debug row {line_number}: {exc}")
                continue
            if not isinstance(record, dict):
                warnings.append(f"partial debug row {line_number} is not an object")
                continue
            step = record.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                warnings.append(f"partial debug row {line_number} has an invalid step")
                continue
            if allowed_steps is not None and step not in allowed_steps:
                continue

            metrics["debug_record_count"] += 1
            steps.add(step)
            rewards = record.get("rewards")
            if isinstance(rewards, list):
                reward_sum += sum(_number(value) or 0 for value in rewards)

            parse_stats = record.get("action_parse_stats")
            if isinstance(parse_stats, dict) and (
                latest_parse_step is None or step >= latest_parse_step
            ):
                latest_parse_step = step
                latest_parse_stats = parse_stats

            agents = record.get("agents")
            if isinstance(agents, dict):
                for agent in agents.values():
                    if not isinstance(agent, dict):
                        continue
                    metrics["model_call_count"] += 1
                    metrics["provider_request_count"] += (
                        _number(agent.get("transport_attempt_count")) or 0
                    )
                    metrics["input_tokens"] += _number(agent.get("input_tokens")) or 0
                    metrics["output_tokens"] += _number(agent.get("output_tokens")) or 0
                    latency = _number(agent.get("latency_seconds"))
                    if latency is not None:
                        metrics["model_latency_seconds"] += latency
                        latency_observations += 1
                    metrics["transport_error_count"] += (
                        _number(agent.get("transport_error_count")) or 0
                    )
                    if _is_length_stop(agent):
                        metrics["length_stop_count"] += 1
                    raw_output = agent.get("llm_raw_output")
                    if raw_output is None or (
                        isinstance(raw_output, str) and not raw_output.strip()
                    ):
                        metrics["empty_output_count"] += 1

            routes = record.get("communication_routes")
            if isinstance(routes, list):
                communication = metrics["communications"]
                for route in routes:
                    if not isinstance(route, dict):
                        continue
                    content = route.get("content")
                    if not isinstance(content, str):
                        content = ""
                    recipients = route.get("recipients")
                    recipient_count = len(recipients) if isinstance(recipients, list) else 0
                    payload_bytes = len(content.encode("utf-8"))
                    communication["emitted_messages"] += 1
                    communication["delivered_messages"] += recipient_count
                    communication["payload_bytes"] += payload_bytes
                    communication["delivery_bytes"] += payload_bytes * recipient_count

    metrics["observed_steps"] = len(steps)
    metrics["observed_step_values"] = sorted(steps)
    if steps:
        metrics["first_step"] = min(steps)
        metrics["last_step"] = max(steps)
        metrics["reward_sum"] = reward_sum
    metrics["mean_model_latency_seconds"] = _divide(
        metrics["model_latency_seconds"], latency_observations
    )

    if latest_parse_stats is not None:
        parse_success = 0
        parse_fail = 0
        found_parse_count = False
        for stats in latest_parse_stats.values():
            if not isinstance(stats, dict):
                continue
            success = _number(stats.get("success"))
            fail = _number(stats.get("fail"))
            if success is not None:
                parse_success += success
                found_parse_count = True
            if fail is not None:
                parse_fail += fail
                found_parse_count = True
        if found_parse_count:
            metrics["action_parse_success"] = parse_success
            metrics["action_parse_fail"] = parse_fail
            metrics["action_parse_rate"] = _divide(
                parse_success, parse_success + parse_fail
            )
    return metrics, warnings


def _matched_prefix_rows(
    run_root: Path,
    episode_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build descriptive, non-efficacy response comparisons on observed common steps."""

    by_arm_seed = {(row["arm"], row["expected_seed"]): row for row in episode_rows}
    comparison_rows = []
    warnings = []
    for episode_index, seed in enumerate(EXPECTED_SEEDS):
        thinking_row = by_arm_seed[("free_thinking", seed)]
        thinking = thinking_row["partial_debug"]
        steps = set(thinking["observed_step_values"])
        if not thinking["available"] or not steps:
            continue

        concise_relative_path = (
            Path("free_concise")
            / "alem"
            / "default"
            / f"default_run_{episode_index:02d}_debug.jsonl"
        )
        concise, concise_warnings = _partial_debug_metrics(
            run_root / concise_relative_path,
            concise_relative_path,
            allowed_steps=steps,
        )
        warnings.extend(
            f"matched prefix concise seed {seed}: {warning}"
            for warning in concise_warnings
        )
        matched_steps_complete = (
            concise["available"] and set(concise["observed_step_values"]) == steps
        )
        if not matched_steps_complete:
            warnings.append(
                f"matched prefix concise seed {seed}: expected steps {sorted(steps)}, "
                f"found {concise['observed_step_values']}"
            )

        comparison_rows.append(
            {
                "classification": "matched_prefix_descriptive_non_efficacy",
                "eligible_for_registered_efficacy": False,
                "seed": seed,
                "steps": len(steps),
                "matched_steps_complete": matched_steps_complete,
                "concise_reward_sum": concise["reward_sum"],
                "thinking_reward_sum": thinking["reward_sum"],
                "concise_parse_rate": concise["action_parse_rate"],
                "thinking_parse_rate": thinking["action_parse_rate"],
                "concise_output_tokens_per_call": _divide(
                    concise["output_tokens"], concise["model_call_count"]
                ),
                "thinking_output_tokens_per_call": _divide(
                    thinking["output_tokens"], thinking["model_call_count"]
                ),
                "concise_mean_latency_seconds": concise[
                    "mean_model_latency_seconds"
                ],
                "thinking_mean_latency_seconds": thinking[
                    "mean_model_latency_seconds"
                ],
                "concise_length_stops": concise["length_stop_count"],
                "thinking_length_stops": thinking["length_stop_count"],
                "concise_empty_outputs": concise["empty_output_count"],
                "thinking_empty_outputs": thinking["empty_output_count"],
                "concise_source_path": str(concise_relative_path),
                "thinking_source_path": thinking["source_path"],
                "source_steps": sorted(steps),
            }
        )
    return comparison_rows, warnings


def _episode_row(
    run_root: Path,
    arm: str,
    episode_index: int,
    attempts: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    expected_seed = EXPECTED_SEEDS[episode_index]
    relative_path = Path(arm) / "alem" / "default" / f"default_run_{episode_index:02d}.json"
    relative_debug_path = (
        Path(arm) / "alem" / "default" / f"default_run_{episode_index:02d}_debug.jsonl"
    )
    artifact_path = run_root / relative_path
    row: dict[str, Any] = {
        "arm": arm,
        "episode_index": episode_index,
        "expected_seed": expected_seed,
        "seed": None,
        "artifact_path": str(relative_path),
        "present": artifact_path.is_file(),
        "json_readable": False,
        "valid": False,
        "invalid_reasons": [],
        "audit_warnings": [],
        "artifact": {
            "schema_version": None,
            "artifact_status": None,
            "attempt_id": None,
            "error": None,
            "early_stop_reason": None,
            "termination_reason": None,
        },
        "execution": {"num_steps": None, "reached_step_cap": False},
        "outcomes": {
            "episode_return": None,
            "total_achievements": None,
            "coordination_attempts": None,
            "coordination_successes": None,
            "coordination_success_rate": None,
            "action_parse_rate": None,
            "action_parse_success": None,
            "action_parse_fail": None,
            "action_parse_skipped_inactive": None,
        },
        "mechanism": {
            "strategy": None,
            "protocol_parse_rate": None,
            "status_window_coverage": None,
            "status_messages": None,
            "protocol_messages": None,
        },
        "communication": {
            "delivery_bytes": None,
            "worker_peer_delivery_bytes": None,
        },
        "normalized": {
            "input_tokens_per_model_call": None,
            "output_tokens_per_model_call": None,
            "wall_seconds_per_step": None,
            "delivery_bytes_per_step": None,
        },
        "final_attempt_usage": _usage(None),
        "cumulative_attempt_usage": _usage(None),
        "cumulative_usage_source": None,
        "attempt_count": 0,
        "failed_attempt_count": 0,
        "partial_debug": _empty_partial_debug(relative_debug_path),
    }
    payload: dict[str, Any] | None = None
    if not artifact_path.is_file():
        row["invalid_reasons"].append("missing_artifact")
    else:
        try:
            decoded = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            row["invalid_reasons"].append("malformed_json")
            row["audit_warnings"].append(str(exc))
        else:
            if not isinstance(decoded, dict):
                row["invalid_reasons"].append("artifact_not_object")
            else:
                payload = decoded
                row["json_readable"] = True
                row["seed"] = payload.get("seed")
                row["invalid_reasons"].extend(_invalid_reasons(payload, expected_seed))
                row["artifact"] = {
                    "schema_version": payload.get("schema_version"),
                    "artifact_status": payload.get("artifact_status"),
                    "attempt_id": payload.get("attempt_id"),
                    "error": payload.get("error"),
                    "early_stop_reason": payload.get("early_stop_reason"),
                    "termination_reason": payload.get("termination_reason"),
                }
                num_steps = _number(payload.get("num_steps"))
                row["execution"] = {
                    "num_steps": num_steps,
                    "reached_step_cap": num_steps == MAX_STEPS,
                }
                user_info = payload.get("user_info")
                if not isinstance(user_info, dict):
                    user_info = {}
                row["outcomes"] = {
                    "episode_return": _number(payload.get("episode_return")),
                    "total_achievements": _number(user_info.get("Team/total_achievements")),
                    "coordination_attempts": _number(user_info.get("Coordination/total_attempts")),
                    "coordination_successes": _number(
                        user_info.get("Coordination/total_successes")
                    ),
                    "coordination_success_rate": _number(
                        user_info.get("Coordination/coordination_success_rate")
                    ),
                    "action_parse_rate": _number(payload.get("action_parse_rate")),
                    "action_parse_success": _number(payload.get("action_parse_success")),
                    "action_parse_fail": _number(payload.get("action_parse_fail")),
                    "action_parse_skipped_inactive": _number(
                        payload.get("action_parse_skipped_inactive")
                    ),
                }
                protocol = payload.get("coordination_protocol")
                if not isinstance(protocol, dict):
                    protocol = {}
                row["mechanism"] = {
                    "strategy": protocol.get("strategy"),
                    "protocol_parse_rate": _number(protocol.get("protocol_parse_rate")),
                    "status_window_coverage": _number(protocol.get("status_window_coverage")),
                    "status_messages": _number(protocol.get("status")),
                    "protocol_messages": _number(protocol.get("protocol_messages")),
                }
                communication = payload.get("communication_metrics")
                worker_peer = (
                    communication.get("worker_peer") if isinstance(communication, dict) else None
                )
                row["communication"] = {
                    "delivery_bytes": _delivered_bytes(communication),
                    "worker_peer_delivery_bytes": (
                        _number(worker_peer.get("delivery_bytes"))
                        if isinstance(worker_peer, dict)
                        else None
                    ),
                }
                row["final_attempt_usage"] = _usage(payload)
                model_calls = row["final_attempt_usage"]["model_call_count"]
                episode_wall = row["final_attempt_usage"]["episode_wall_seconds"]
                delivery_bytes = row["communication"]["delivery_bytes"]
                row["normalized"] = {
                    "input_tokens_per_model_call": _divide(
                        row["final_attempt_usage"]["input_tokens"], model_calls
                    ),
                    "output_tokens_per_model_call": _divide(
                        row["final_attempt_usage"]["output_tokens"], model_calls
                    ),
                    "wall_seconds_per_step": _divide(episode_wall, num_steps),
                    "delivery_bytes_per_step": _divide(delivery_bytes, num_steps),
                }

    # A debug stream without a usable canonical artifact proves only that a
    # prefix ran. Keep those measurements in a separate, explicitly
    # non-efficacy namespace and leave `valid` false.
    if payload is None:
        partial_debug, partial_warnings = _partial_debug_metrics(
            run_root / relative_debug_path,
            relative_debug_path,
        )
        row["partial_debug"] = partial_debug
        row["audit_warnings"].extend(partial_warnings)

    episode_attempts = attempts.get(episode_index, [])
    if episode_attempts:
        row["attempt_count"] = len(episode_attempts)
        row["failed_attempt_count"] = sum(
            record.get("artifact_status") != "complete" for record in episode_attempts
        )
        row["cumulative_attempt_usage"] = _sum_usage(episode_attempts)
        row["cumulative_usage_source"] = "attempt_ledger"
        final_attempt_id = payload.get("attempt_id") if isinstance(payload, dict) else None
        if (
            isinstance(final_attempt_id, str)
            and final_attempt_id
            and not any(record.get("attempt_id") == final_attempt_id for record in episode_attempts)
        ):
            row["audit_warnings"].append("stable artifact attempt_id is absent from attempt ledger")
    elif payload is not None:
        row["attempt_count"] = 1
        row["failed_attempt_count"] = int(payload.get("artifact_status") != "complete")
        row["cumulative_attempt_usage"] = _usage(payload)
        row["cumulative_usage_source"] = "stable_artifact_fallback"
    row["valid"] = not row["invalid_reasons"]
    return row


def _mark_duplicate_seeds(rows: list[dict[str, Any]]) -> None:
    by_arm_and_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = row["seed"]
        if isinstance(seed, int) and not isinstance(seed, bool):
            by_arm_and_seed[(row["arm"], seed)].append(row)
    for duplicates in by_arm_and_seed.values():
        if len(duplicates) < 2:
            continue
        for row in duplicates:
            if "duplicate_seed" not in row["invalid_reasons"]:
                row["invalid_reasons"].append("duplicate_seed")
                row["valid"] = False


def _metric_values(
    rows: list[dict[str, Any]],
    getter,
) -> dict[str, int | float | None]:
    values = [getter(row) for row in rows]
    values = [value for value in values if _number(value) is not None]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _arm_elapsed_seconds(run_root: Path) -> dict[str, float | None]:
    elapsed: dict[str, float | None] = {arm: None for arm in ARMS}
    events_path = run_root / "events.jsonl"
    if not events_path.is_file():
        return elapsed
    starts: dict[str, datetime] = {}
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return elapsed
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict) or event.get("arm") not in ARMS:
            continue
        try:
            timestamp = datetime.fromisoformat(str(event.get("at")))
        except ValueError:
            continue
        arm = event["arm"]
        if event.get("event") == "arm_started":
            starts[arm] = timestamp
        elif event.get("event") == "arm_finished" and arm in starts:
            elapsed[arm] = (timestamp - starts[arm]).total_seconds()
    return elapsed


def _arm_aggregate(
    arm: str,
    rows: list[dict[str, Any]],
    arm_elapsed_seconds: float | None,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["valid"]]
    partial_rows = [row for row in rows if row["partial_debug"]["available"]]
    metrics = {
        "num_steps": _metric_values(valid_rows, lambda row: row["execution"]["num_steps"]),
        "episode_return": _metric_values(valid_rows, lambda row: row["outcomes"]["episode_return"]),
        "total_achievements": _metric_values(
            valid_rows, lambda row: row["outcomes"]["total_achievements"]
        ),
        "coordination_attempts": _metric_values(
            valid_rows, lambda row: row["outcomes"]["coordination_attempts"]
        ),
        "coordination_successes": _metric_values(
            valid_rows, lambda row: row["outcomes"]["coordination_successes"]
        ),
        "action_parse_rate": _metric_values(
            valid_rows, lambda row: row["outcomes"]["action_parse_rate"]
        ),
        "delivery_bytes": _metric_values(
            valid_rows, lambda row: row["communication"]["delivery_bytes"]
        ),
        "episode_wall_seconds": _metric_values(
            valid_rows, lambda row: row["final_attempt_usage"]["episode_wall_seconds"]
        ),
        "protocol_parse_rate": _metric_values(
            valid_rows, lambda row: row["mechanism"]["protocol_parse_rate"]
        ),
        "status_window_coverage": _metric_values(
            valid_rows, lambda row: row["mechanism"]["status_window_coverage"]
        ),
        "input_tokens_per_model_call": _metric_values(
            valid_rows,
            lambda row: row["normalized"]["input_tokens_per_model_call"],
        ),
        "output_tokens_per_model_call": _metric_values(
            valid_rows,
            lambda row: row["normalized"]["output_tokens_per_model_call"],
        ),
        "wall_seconds_per_step": _metric_values(
            valid_rows, lambda row: row["normalized"]["wall_seconds_per_step"]
        ),
        "delivery_bytes_per_step": _metric_values(
            valid_rows, lambda row: row["normalized"]["delivery_bytes_per_step"]
        ),
    }
    cumulative_cost = {
        field: sum(row["cumulative_attempt_usage"].get(field) or 0 for row in rows)
        for field in USAGE_FIELDS
    }
    cumulative_cost["total_tokens"] = (
        cumulative_cost["input_tokens"] + cumulative_cost["output_tokens"]
    )
    partial_metric_getters = {
        "observed_steps": lambda row: row["partial_debug"]["observed_steps"],
        "reward_sum": lambda row: row["partial_debug"]["reward_sum"],
        "model_call_count": lambda row: row["partial_debug"]["model_call_count"],
        "provider_request_count": lambda row: row["partial_debug"][
            "provider_request_count"
        ],
        "input_tokens": lambda row: row["partial_debug"]["input_tokens"],
        "output_tokens": lambda row: row["partial_debug"]["output_tokens"],
        "mean_model_latency_seconds": lambda row: row["partial_debug"][
            "mean_model_latency_seconds"
        ],
        "length_stop_count": lambda row: row["partial_debug"]["length_stop_count"],
        "empty_output_count": lambda row: row["partial_debug"]["empty_output_count"],
        "action_parse_success": lambda row: row["partial_debug"]["action_parse_success"],
        "action_parse_fail": lambda row: row["partial_debug"]["action_parse_fail"],
        "action_parse_rate": lambda row: row["partial_debug"]["action_parse_rate"],
        "transport_error_count": lambda row: row["partial_debug"][
            "transport_error_count"
        ],
        "communication_emitted_messages": lambda row: row["partial_debug"][
            "communications"
        ]["emitted_messages"],
        "communication_delivered_messages": lambda row: row["partial_debug"][
            "communications"
        ]["delivered_messages"],
        "communication_delivery_bytes": lambda row: row["partial_debug"][
            "communications"
        ]["delivery_bytes"],
    }
    return {
        "arm": arm,
        "expected_episode_count": len(EXPECTED_SEEDS),
        "artifact_count": sum(row["present"] for row in rows),
        "valid_episode_count": len(valid_rows),
        "valid_episode_rate": len(valid_rows) / len(EXPECTED_SEEDS),
        "reached_step_cap_count": sum(row["execution"]["reached_step_cap"] for row in valid_rows),
        "attempt_count": sum(row["attempt_count"] for row in rows),
        "failed_attempt_count": sum(row["failed_attempt_count"] for row in rows),
        "termination_reason_counts": dict(
            sorted(
                Counter(
                    row["artifact"]["termination_reason"]
                    for row in valid_rows
                    if row["artifact"]["termination_reason"]
                ).items()
            )
        ),
        "metrics": metrics,
        "cumulative_attempt_usage": cumulative_cost,
        "cumulative_usage_episode_coverage": sum(
            row["cumulative_usage_source"] is not None for row in rows
        ),
        "arm_elapsed_seconds": arm_elapsed_seconds,
        "partial_debug_non_efficacy": {
            "classification": "partial_non_efficacy",
            "eligible_for_efficacy": False,
            "episode_count": len(partial_rows),
            "metrics": {
                field: _metric_values(partial_rows, getter)
                for field, getter in partial_metric_getters.items()
            },
        },
    }


def _contrast_value(row: dict[str, Any], field: str) -> int | float | None:
    if field == "num_steps":
        return row["execution"]["num_steps"]
    if field in {
        "episode_return",
        "total_achievements",
        "coordination_attempts",
        "coordination_successes",
        "action_parse_rate",
    }:
        return row["outcomes"][field]
    if field == "delivery_bytes":
        return row["communication"]["delivery_bytes"]
    if field in {"input_tokens", "output_tokens", "model_call_count"}:
        return row["final_attempt_usage"][field]
    if field == "total_tokens":
        input_tokens = row["final_attempt_usage"]["input_tokens"]
        output_tokens = row["final_attempt_usage"]["output_tokens"]
        if input_tokens is None or output_tokens is None:
            return None
        return input_tokens + output_tokens
    if field == "episode_wall_seconds":
        return row["final_attempt_usage"]["episode_wall_seconds"]
    if field in {
        "input_tokens_per_model_call",
        "output_tokens_per_model_call",
        "wall_seconds_per_step",
        "delivery_bytes_per_step",
    }:
        return row["normalized"][field]
    raise KeyError(field)


def _sign_summary(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "positive": 0,
            "tie": 0,
            "negative": 0,
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "positive": sum(value > 0 for value in values),
        "tie": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
    }


def _data_integrity_problem(rows: list[dict[str, Any]]) -> bool:
    return any(
        any(reason in DATA_INTEGRITY_REASONS for reason in row["invalid_reasons"]) for row in rows
    )


def _decision(
    name: str,
    seed_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def count_where(field: str, predicate) -> int:
        return sum(
            row["paired_valid"]
            and row["deltas"][field] is not None
            and predicate(row["deltas"][field])
            for row in seed_rows
        )

    valid_pair_count = sum(row["paired_valid"] for row in seed_rows)
    if name == "response_mode":
        criteria = {
            "treatment_completes_at_least_as_many": (
                sum(row["valid"] for row in treatment_rows)
                >= sum(row["valid"] for row in control_rows)
            ),
            "parse_not_lower_on_at_least_two_seeds": (
                count_where("action_parse_rate", lambda value: value >= 0) >= 2
            ),
            "output_tokens_per_call_lower_on_at_least_two_seeds": (
                count_where("output_tokens_per_model_call", lambda value: value < 0)
                >= 2
            ),
            "achievements_reduced_on_at_most_one_seed": (
                count_where("total_achievements", lambda value: value < 0) <= 1
            ),
        }
    else:
        criteria = {
            "achievements_improve_on_at_least_two_seeds": (
                count_where("total_achievements", lambda value: value > 0) >= 2
            ),
            "parse_lower_on_at_most_one_seed": (
                count_where("action_parse_rate", lambda value: value < 0) <= 1
            ),
        }
    integrity_problem = _data_integrity_problem(treatment_rows + control_rows)
    criteria_evaluated = valid_pair_count == len(EXPECTED_SEEDS)
    if not criteria_evaluated:
        status = "indeterminate_incomplete_pairs"
    elif all(criteria.values()):
        status = "positive_signal"
    else:
        status = "not_positive"
    return {
        "status": status,
        "criteria_evaluated": criteria_evaluated,
        "criteria": (
            criteria
            if criteria_evaluated
            else {criterion: None for criterion in criteria}
        ),
        "valid_pair_count": valid_pair_count,
        "required_valid_pair_count": len(EXPECTED_SEEDS),
        "data_integrity_problem": integrity_problem,
    }


def _contrast(
    name: str,
    treatment: str,
    control: str,
    rows_by_arm_seed: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    seed_rows = []
    treatment_rows = []
    control_rows = []
    for seed in EXPECTED_SEEDS:
        treatment_row = rows_by_arm_seed[(treatment, seed)]
        control_row = rows_by_arm_seed[(control, seed)]
        treatment_rows.append(treatment_row)
        control_rows.append(control_row)
        paired_valid = treatment_row["valid"] and control_row["valid"]
        if treatment_row["valid"] and control_row["valid"]:
            validity = "both_valid"
        elif treatment_row["valid"]:
            validity = "treatment_only"
        elif control_row["valid"]:
            validity = "control_only"
        else:
            validity = "neither_valid"
        deltas = {}
        for field in OUTCOME_CONTRAST_FIELDS:
            treatment_value = _contrast_value(treatment_row, field)
            control_value = _contrast_value(control_row, field)
            deltas[field] = (
                treatment_value - control_value
                if paired_valid and treatment_value is not None and control_value is not None
                else None
            )
        seed_rows.append(
            {
                "contrast": name,
                "seed": seed,
                "treatment": treatment,
                "control": control,
                "treatment_valid": treatment_row["valid"],
                "control_valid": control_row["valid"],
                "paired_valid": paired_valid,
                "validity": validity,
                "deltas": deltas,
            }
        )
    aggregate = {
        "valid_pair_count": sum(row["paired_valid"] for row in seed_rows),
        "metrics": {
            field: _sign_summary(
                [row["deltas"][field] for row in seed_rows if row["deltas"][field] is not None]
            )
            for field in OUTCOME_CONTRAST_FIELDS
        },
    }
    return {
        "name": name,
        "treatment": treatment,
        "control": control,
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "decision": _decision(name, seed_rows, treatment_rows, control_rows),
    }


def _flatten_episode(row: dict[str, Any]) -> dict[str, Any]:
    partial_debug = row["partial_debug"]
    return {
        "arm": row["arm"],
        "episode_index": row["episode_index"],
        "expected_seed": row["expected_seed"],
        "seed": row["seed"],
        "valid": row["valid"],
        "invalid_reasons": ";".join(row["invalid_reasons"]),
        "artifact_status": row["artifact"]["artifact_status"],
        "error": row["artifact"]["error"],
        "termination_reason": row["artifact"]["termination_reason"],
        "num_steps": row["execution"]["num_steps"],
        "reached_step_cap": row["execution"]["reached_step_cap"],
        **row["outcomes"],
        **row["mechanism"],
        **row["communication"],
        **row["normalized"],
        **{f"final_{field}": value for field, value in row["final_attempt_usage"].items()},
        "attempt_count": row["attempt_count"],
        "failed_attempt_count": row["failed_attempt_count"],
        **{
            f"cumulative_{field}": value for field, value in row["cumulative_attempt_usage"].items()
        },
        "cumulative_usage_source": row["cumulative_usage_source"],
        **{
            f"partial_{field}": value
            for field, value in partial_debug.items()
            if field != "communications"
        },
        **{
            f"partial_communication_{field}": value
            for field, value in partial_debug["communications"].items()
        },
        "artifact_path": row["artifact_path"],
    }


def _flatten_contrast(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "contrast": row["contrast"],
        "seed": row["seed"],
        "treatment": row["treatment"],
        "control": row["control"],
        "treatment_valid": row["treatment_valid"],
        "control_valid": row["control_valid"],
        "paired_valid": row["paired_valid"],
        "validity": row["validity"],
        **{f"delta_{field}": value for field, value in row["deltas"].items()},
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer CSV fields for empty rows: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames or tuple(rows[0])),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_selected_debug_steps(
    source: Path,
    destination: Path,
    selected_steps: set[int],
) -> int:
    """Archive exact JSONL records for selected steps and return their count."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_lines = []
    with source.open(encoding="utf-8") as source_handle:
        for line in source_handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("step") in selected_steps:
                selected_lines.append(line if line.endswith("\n") else line + "\n")
    destination.write_text("".join(selected_lines), encoding="utf-8")
    return len(selected_lines)


def _archive_audit_artifacts(
    run_root: Path,
    artifact_dir: Path,
    matched_prefix_rows: list[dict[str, Any]],
    derived_paths: tuple[Path, ...],
) -> Path:
    """Copy compact raw/provenance evidence out of the ignored output tree."""

    candidates = [
        (run_root / "manifest.json", Path("study") / "manifest.json"),
        (run_root / "events.jsonl", Path("study") / "events.jsonl"),
    ]
    for arm in ARMS:
        arm_root = run_root / arm
        for name in ("resolved_config.yaml", "run_manifest.json", "summary_stats.json"):
            candidates.append((arm_root / name, Path("arms") / arm / name))
        candidates.append((arm_root / "eval.log", Path("arms") / arm / "eval.log"))
        candidates.append(
            (
                arm_root / "alem" / "default" / "attempt_ledger.jsonl",
                Path("arms") / arm / "attempt_ledger.jsonl",
            )
        )
        for episode_index in range(len(EXPECTED_SEEDS)):
            name = f"default_run_{episode_index:02d}.json"
            canonical_path = arm_root / "alem" / "default" / name
            candidates.append(
                (
                    canonical_path,
                    Path("raw_episodes") / arm / name,
                )
            )
            if not canonical_path.is_file():
                for partial_name in (
                    f"default_run_{episode_index:02d}_debug.jsonl",
                    f"default_run_{episode_index:02d}.csv",
                ):
                    candidates.append(
                        (
                            arm_root / "alem" / "default" / partial_name,
                            Path("partial_runs") / arm / partial_name,
                        )
                    )
    for arm in ARMS:
        for console_log in sorted(run_root.glob(f"{arm}.attempt_*.console.log")):
            candidates.append(
                (
                    console_log,
                    Path("study") / "console_logs" / console_log.name,
                )
            )

    audit_root = artifact_dir / "audit"
    inventory = []
    for source, relative_destination in candidates:
        record = {
            "source_relative_path": str(source.relative_to(run_root)),
            "archived_relative_path": str(Path("audit") / relative_destination),
            "present": source.is_file(),
            "size_bytes": None,
            "sha256": None,
        }
        if source.is_file():
            destination = audit_root / relative_destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record["size_bytes"] = destination.stat().st_size
            record["sha256"] = _sha256(destination)
        inventory.append(record)

    for comparison in matched_prefix_rows:
        selected_steps = set(comparison["source_steps"])
        for label in ("concise", "thinking"):
            source_relative = Path(comparison[f"{label}_source_path"])
            source = run_root / source_relative
            destination_relative = (
                Path("matched_prefix_sources")
                / label
                / f"seed_{comparison['seed']}_debug.jsonl"
            )
            destination = audit_root / destination_relative
            record = {
                "source_relative_path": str(source_relative),
                "archived_relative_path": str(Path("audit") / destination_relative),
                "present": source.is_file(),
                "size_bytes": None,
                "sha256": None,
                "source_sha256": None,
                "selection": {"steps": sorted(selected_steps)},
                "selected_record_count": 0,
            }
            if source.is_file():
                record["source_sha256"] = _sha256(source)
                record["selected_record_count"] = _archive_selected_debug_steps(
                    source, destination, selected_steps
                )
                record["size_bytes"] = destination.stat().st_size
                record["sha256"] = _sha256(destination)
            inventory.append(record)

    derived_files = []
    for path in derived_paths:
        record = {
            "relative_path": str(path.relative_to(artifact_dir)),
            "present": path.is_file(),
            "size_bytes": None,
            "sha256": None,
        }
        if path.is_file():
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
        derived_files.append(record)

    inventory_path = artifact_dir / "artifact_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "alem-signal-scale-artifact-inventory-v1",
                "run_root": str(run_root),
                "files": inventory,
                "derived_files": derived_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return inventory_path


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Three-seed, 200-step signal study",
        "",
        "This report evaluates three paired seeds (9999, 10000, and 10001). "
        "The three episodes within each arm ran in parallel. Arms ran sequentially.",
        "",
        "A valid episode must have the canonical episode schema, a complete artifact "
        "status, no error, a terminal reason, no repeated-length early stop, and the "
        "preregistered seed. Natural environment termination before 200 steps remains valid.",
        "",
        "## Arm-level results",
        "",
        "| Arm | Valid | Reached 200 | Return mean | Achievements mean | Coord. successes mean | Parse rate mean | Input/call | Output/call | Wall/step | Delivery/step | Recorded tokens | Arm elapsed (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        aggregate = summary["arm_aggregates"][arm]
        metrics = aggregate["metrics"]
        if aggregate["cumulative_usage_episode_coverage"]:
            recorded_tokens = _fmt(
                aggregate["cumulative_attempt_usage"]["total_tokens"]
            )
        elif aggregate["partial_debug_non_efficacy"]["episode_count"]:
            partial_tokens = sum(
                row["partial_debug"]["input_tokens"]
                + row["partial_debug"]["output_tokens"]
                for row in summary["episodes"]
                if row["arm"] == arm and row["partial_debug"]["available"]
            )
            recorded_tokens = f"{_fmt(partial_tokens)} partial"
        else:
            recorded_tokens = "—"
        lines.append(
            f"| {arm} | {aggregate['valid_episode_count']}/3 | "
            f"{aggregate['reached_step_cap_count']} | "
            f"{_fmt(metrics['episode_return']['mean'])} | "
            f"{_fmt(metrics['total_achievements']['mean'])} | "
            f"{_fmt(metrics['coordination_successes']['mean'])} | "
            f"{_fmt(metrics['action_parse_rate']['mean'])} | "
            f"{_fmt(metrics['input_tokens_per_model_call']['mean'])} | "
            f"{_fmt(metrics['output_tokens_per_model_call']['mean'])} | "
            f"{_fmt(metrics['wall_seconds_per_step']['mean'])} | "
            f"{_fmt(metrics['delivery_bytes_per_step']['mean'])} | "
            f"{recorded_tokens} | "
            f"{_fmt(aggregate['arm_elapsed_seconds'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## Seed-level results",
            "",
            "| Arm | Seed | Valid | Steps | Return | Achievements | Coord. successes | Parse rate | Input tokens | Output tokens | Calls | Input/call | Output/call | Wall/step | Delivery/step |",
            "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["episodes"]:
        invalid = "yes" if row["valid"] else "no (" + ", ".join(row["invalid_reasons"]) + ")"
        lines.append(
            f"| {row['arm']} | {row['expected_seed']} | {invalid} | "
            f"{_fmt(row['execution']['num_steps'])} | "
            f"{_fmt(row['outcomes']['episode_return'])} | "
            f"{_fmt(row['outcomes']['total_achievements'])} | "
            f"{_fmt(row['outcomes']['coordination_successes'])} | "
            f"{_fmt(row['outcomes']['action_parse_rate'])} | "
            f"{_fmt(row['final_attempt_usage']['input_tokens'])} | "
            f"{_fmt(row['final_attempt_usage']['output_tokens'])} | "
            f"{_fmt(row['final_attempt_usage']['model_call_count'])} | "
            f"{_fmt(row['normalized']['input_tokens_per_model_call'])} | "
            f"{_fmt(row['normalized']['output_tokens_per_model_call'])} | "
            f"{_fmt(row['normalized']['wall_seconds_per_step'])} | "
            f"{_fmt(row['normalized']['delivery_bytes_per_step'])} |"
        )
    partial_rows = [
        row for row in summary["episodes"] if row["partial_debug"]["available"]
    ]
    lines.extend(
        [
            "",
            "## Interrupted partial telemetry (non-efficacy)",
            "",
            "These per-step debug streams have no canonical episode artifact. They "
            "describe only an observed prefix and are excluded from efficacy aggregates, "
            "paired contrasts, and preregistered decisions.",
            "",
        ]
    )
    if partial_rows:
        lines.extend(
            [
                "| Arm | Seed | Steps | Reward sum | Calls | Input | Output | Mean latency (s) | Length stops | Empty | Parse ok | Parse fail | Parse rate | Transport errors | Comms | Delivery bytes |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in partial_rows:
            partial = row["partial_debug"]
            lines.append(
                f"| {row['arm']} | {row['expected_seed']} | "
                f"{_fmt(partial['observed_steps'])} | {_fmt(partial['reward_sum'])} | "
                f"{_fmt(partial['model_call_count'])} | {_fmt(partial['input_tokens'])} | "
                f"{_fmt(partial['output_tokens'])} | "
                f"{_fmt(partial['mean_model_latency_seconds'])} | "
                f"{_fmt(partial['length_stop_count'])} | "
                f"{_fmt(partial['empty_output_count'])} | "
                f"{_fmt(partial['action_parse_success'])} | "
                f"{_fmt(partial['action_parse_fail'])} | "
                f"{_fmt(partial['action_parse_rate'])} | "
                f"{_fmt(partial['transport_error_count'])} | "
                f"{_fmt(partial['communications']['emitted_messages'])} | "
                f"{_fmt(partial['communications']['delivery_bytes'])} |"
            )
    else:
        lines.append("No interrupted debug-only episode prefixes were found.")
    matched_prefix_rows = summary["matched_prefix_response"]["rows"]
    if matched_prefix_rows:
        lines.extend(
            [
                "",
                "## Matched response prefix (descriptive only)",
                "",
                "The interrupted thinking arm is compared with the same observed steps "
                "from the completed concise arm. These rows are excluded from the "
                "registered efficacy decision. Exact source slices and hashes are in the "
                "artifact inventory.",
                "",
                "| Seed | Steps | Complete | Concise reward | Thinking reward | Concise parse | Thinking parse | Concise output/call | Thinking output/call | Concise latency | Thinking latency | Concise length/empty | Thinking length/empty |",
                "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in matched_prefix_rows:
            lines.append(
                f"| {row['seed']} | {row['steps']} | "
                f"{_fmt(row['matched_steps_complete'])} | "
                f"{_fmt(row['concise_reward_sum'])} | "
                f"{_fmt(row['thinking_reward_sum'])} | "
                f"{_fmt(row['concise_parse_rate'])} | "
                f"{_fmt(row['thinking_parse_rate'])} | "
                f"{_fmt(row['concise_output_tokens_per_call'])} | "
                f"{_fmt(row['thinking_output_tokens_per_call'])} | "
                f"{_fmt(row['concise_mean_latency_seconds'])} | "
                f"{_fmt(row['thinking_mean_latency_seconds'])} | "
                f"{_fmt(row['concise_length_stops'])}/{_fmt(row['concise_empty_outputs'])} | "
                f"{_fmt(row['thinking_length_stops'])}/{_fmt(row['thinking_empty_outputs'])} |"
            )
    lines.extend(["", "## Paired contrasts", ""])
    for name in CONTRASTS:
        contrast = summary["contrasts"][name]
        lines.extend(
            [
                f"### {name}",
                "",
                f"Treatment minus control: {contrast['treatment']} − {contrast['control']}.",
                "",
                "| Seed | Pair valid | Δ parse | Δ achievements | Δ return | Δ coord. successes | Δ output/call | Δ wall/step |",
                "| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in contrast["seed_rows"]:
            lines.append(
                f"| {row['seed']} | {_fmt(row['paired_valid'])} | "
                f"{_fmt(row['deltas']['action_parse_rate'])} | "
                f"{_fmt(row['deltas']['total_achievements'])} | "
                f"{_fmt(row['deltas']['episode_return'])} | "
                f"{_fmt(row['deltas']['coordination_successes'])} | "
                f"{_fmt(row['deltas']['output_tokens_per_model_call'])} | "
                f"{_fmt(row['deltas']['wall_seconds_per_step'])} |"
            )
        decision = contrast["decision"]
        lines.extend(
            [
                "",
                f"Decision: **{decision['status']}**.",
                "",
            ]
        )
        if decision["status"] == "indeterminate_incomplete_pairs":
            lines.append(
                "- Decision criteria were not evaluated because the required "
                "three valid seed pairs were unavailable."
            )
        else:
            for criterion, passed in decision["criteria"].items():
                lines.append(f"- {criterion}: {_fmt(passed)}")
        lines.append("")
    if summary["audit_warnings"]:
        lines.extend(["## Audit warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["audit_warnings"])
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Three paired seeds are descriptive and underpowered; no p-values or confidence intervals are claimed.",
            "- All three valid pairs are required for either preregistered decision; otherwise the result is indeterminate.",
            "- Failed or missing seed pairs receive no numeric delta and cannot count as an improvement.",
            "- Partial debug telemetry is non-efficacy evidence and cannot make a missing episode or pair valid.",
            "- Paired behavioral and response-cost contrasts use the final stable episode artifact. Cumulative attempt usage separately includes failed retries from the durable attempt ledger.",
            "- Coordination successes are a secondary cohesion diagnostic; the cohesion decision uses total achievements and action parsing.",
            "- Episode wall times overlap because seeds ran concurrently. Arm elapsed time is derived from runner events; model latency is not treated as wall time.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(
    run_root: Path,
    artifact_dir: Path,
    results_md: Path,
) -> tuple[Path, ...]:
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {run_root}")
    artifact_dir = artifact_dir.expanduser().resolve()
    results_md = results_md.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_md.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    warnings = []
    for arm in ARMS:
        task_dir = run_root / arm / "alem" / "default"
        attempts, ledger_warnings = _load_attempts(task_dir, arm)
        warnings.extend(ledger_warnings)
        rows.extend(
            _episode_row(run_root, arm, episode_index, attempts)
            for episode_index in range(len(EXPECTED_SEEDS))
        )
    _mark_duplicate_seeds(rows)
    for row in rows:
        warnings.extend(
            f"{row['arm']} seed {row['expected_seed']}: {warning}"
            for warning in row["audit_warnings"]
        )
    matched_prefix_rows, matched_prefix_warnings = _matched_prefix_rows(run_root, rows)
    warnings.extend(matched_prefix_warnings)

    elapsed = _arm_elapsed_seconds(run_root)
    arm_aggregates = {
        arm: _arm_aggregate(
            arm,
            [row for row in rows if row["arm"] == arm],
            elapsed[arm],
        )
        for arm in ARMS
    }
    rows_by_arm_seed = {(row["arm"], row["expected_seed"]): row for row in rows}
    contrasts = {
        name: _contrast(name, treatment, control, rows_by_arm_seed)
        for name, (treatment, control) in CONTRASTS.items()
    }
    summary = {
        "schema_version": "alem-signal-scale-summary-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_root": str(run_root),
        "study": {
            "arms": list(ARMS),
            "expected_seeds": list(EXPECTED_SEEDS),
            "max_steps_per_episode": MAX_STEPS,
            "parallel_episode_workers": 3,
        },
        "audit_warnings": warnings,
        "episodes": rows,
        "arm_aggregates": arm_aggregates,
        "contrasts": contrasts,
        "decisions": {name: contrast["decision"] for name, contrast in contrasts.items()},
        "partial_debug_policy": {
            "classification": "partial_non_efficacy",
            "eligible_for_efficacy": False,
            "description": (
                "Debug-only prefixes are reported for interruption auditing and excluded "
                "from valid episodes, contrasts, and decisions."
            ),
        },
        "matched_prefix_response": {
            "classification": "matched_prefix_descriptive_non_efficacy",
            "eligible_for_registered_efficacy": False,
            "description": (
                "Thinking debug-only prefixes are compared with the exact same steps "
                "from concise debug telemetry and remain excluded from registered decisions."
            ),
            "rows": matched_prefix_rows,
        },
        "interpretation_limits": [
            "Three paired seeds are descriptive and underpowered.",
            "All three valid pairs are required for a preregistered decision.",
            "Invalid pairs are not imputed and do not receive numeric deltas.",
            "Paired contrasts use final stable episode metrics; cumulative usage includes retries.",
            "Episode elapsed times overlap under parallel execution.",
            "Debug-only prefixes are interruption telemetry, not efficacy outcomes.",
        ],
    }

    summary_path = artifact_dir / "summary.json"
    episodes_path = artifact_dir / "episodes.csv"
    contrasts_path = artifact_dir / "paired_contrasts.csv"
    matched_prefix_path = artifact_dir / "matched_prefix_response.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(episodes_path, [_flatten_episode(row) for row in rows])
    flat_contrasts = [
        _flatten_contrast(seed_row)
        for contrast in contrasts.values()
        for seed_row in contrast["seed_rows"]
    ]
    _write_csv(contrasts_path, flat_contrasts)
    _write_csv(
        matched_prefix_path,
        matched_prefix_rows,
        fieldnames=MATCHED_PREFIX_FIELDS,
    )
    results_md.write_text(_markdown(summary), encoding="utf-8")
    inventory_path = _archive_audit_artifacts(
        run_root,
        artifact_dir,
        matched_prefix_rows,
        (summary_path, episodes_path, contrasts_path, matched_prefix_path),
    )
    return (
        summary_path,
        episodes_path,
        contrasts_path,
        matched_prefix_path,
        inventory_path,
        results_md,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--results-md", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    outputs = summarize(args.run_root, args.artifact_dir, args.results_md)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
