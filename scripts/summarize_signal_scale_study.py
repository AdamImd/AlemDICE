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


def _episode_row(
    run_root: Path,
    arm: str,
    episode_index: int,
    attempts: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    expected_seed = EXPECTED_SEEDS[episode_index]
    relative_path = Path(arm) / "alem" / "default" / f"default_run_{episode_index:02d}.json"
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
    if valid_pair_count != len(EXPECTED_SEEDS):
        status = "indeterminate_incomplete_pairs"
    elif all(criteria.values()):
        status = "positive_signal"
    else:
        status = "not_positive"
    return {
        "status": status,
        "criteria": criteria,
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_audit_artifacts(run_root: Path, artifact_dir: Path) -> Path:
    """Copy compact raw/provenance evidence out of the ignored output tree."""

    candidates = [
        (run_root / "manifest.json", Path("study") / "manifest.json"),
        (run_root / "events.jsonl", Path("study") / "events.jsonl"),
    ]
    for arm in ARMS:
        arm_root = run_root / arm
        for name in ("resolved_config.yaml", "run_manifest.json", "summary_stats.json"):
            candidates.append((arm_root / name, Path("arms") / arm / name))
        candidates.append(
            (
                arm_root / "alem" / "default" / "attempt_ledger.jsonl",
                Path("arms") / arm / "attempt_ledger.jsonl",
            )
        )
        for episode_index in range(len(EXPECTED_SEEDS)):
            name = f"default_run_{episode_index:02d}.json"
            candidates.append(
                (
                    arm_root / "alem" / "default" / name,
                    Path("raw_episodes") / arm / name,
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

    inventory_path = artifact_dir / "artifact_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "alem-signal-scale-artifact-inventory-v1",
                "run_root": str(run_root),
                "files": inventory,
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
        "| Arm | Valid | Reached 200 | Return mean | Achievements mean | Coord. successes mean | Parse rate mean | Input/call | Output/call | Wall/step | Delivery/step | Total attempt tokens | Arm elapsed (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        aggregate = summary["arm_aggregates"][arm]
        metrics = aggregate["metrics"]
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
            f"{_fmt(aggregate['cumulative_attempt_usage']['total_tokens'])} | "
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
) -> tuple[Path, Path, Path, Path, Path]:
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
        "interpretation_limits": [
            "Three paired seeds are descriptive and underpowered.",
            "All three valid pairs are required for a preregistered decision.",
            "Invalid pairs are not imputed and do not receive numeric deltas.",
            "Paired contrasts use final stable episode metrics; cumulative usage includes retries.",
            "Episode elapsed times overlap under parallel execution.",
        ],
    }

    summary_path = artifact_dir / "summary.json"
    episodes_path = artifact_dir / "episodes.csv"
    contrasts_path = artifact_dir / "paired_contrasts.csv"
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
    results_md.write_text(_markdown(summary), encoding="utf-8")
    inventory_path = _archive_audit_artifacts(run_root, artifact_dir)
    return summary_path, episodes_path, contrasts_path, inventory_path, results_md


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
