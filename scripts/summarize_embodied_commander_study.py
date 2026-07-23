#!/usr/bin/env python3
"""Summarize and gate a staged embodied-commander study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

PRIMARY_ARMS = ("baseline", "embodied_commander_broadcast")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def _load_complete_episode(root: Path, arm: str, index: int) -> tuple[Path, dict]:
    path = root / arm / "easy" / "alem" / "default" / f"default_run_{index:02d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "alem-dice-episode-v1"
        or payload.get("artifact_status") != "complete"
        or payload.get("error")
    ):
        raise ValueError(f"Episode is not a complete canonical artifact: {path}")
    return path, payload


def _attempt_usage(path: Path, index: int, episode: dict) -> dict[str, float]:
    ledger = path.parent / "attempt_ledger.jsonl"
    records = []
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("episode_index") == index:
                records.append(record)
    if not records:
        records = [episode]
    keys = (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "model_call_count",
        "provider_request_count",
        "transport_error_count",
        "model_latency_seconds",
    )
    return {key: sum(float(record.get(key, 0) or 0) for record in records) for key in keys}


def _metric(mapping: dict, key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _availability(value: Any, reason: str) -> dict[str, Any]:
    return (
        {"available": True, "value": value}
        if value is not None
        else {"available": False, "reason": reason}
    )


def _row(root: Path, arm: str, index: int, expected_seed: int) -> dict[str, Any]:
    path, episode = _load_complete_episode(root, arm, index)
    if int(episode.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"Seed mismatch for {arm} episode {index}: expected {expected_seed}, "
            f"got {episode.get('seed')}"
        )
    usage = _attempt_usage(path, index, episode)
    user_info = episode.get("user_info", {}) or {}
    commander = episode.get("squad_commander") or {}
    total_tokens = usage["input_tokens"] + usage["output_tokens"]
    steps = int(episode.get("num_steps", 0) or 0)
    return {
        "arm": arm,
        "episode_index": index,
        "seed": expected_seed,
        "steps": steps,
        "termination_reason": episode.get("termination_reason"),
        "team_return": float(episode.get("episode_return", 0.0) or 0.0),
        "achievement_pct": _metric(user_info, "Team/achievement_pct"),
        "normal_achievement_pct": _metric(user_info, "Team/normal_achievement_pct"),
        "coordination_achievement_pct": _metric(user_info, "Team/coordination_achievement_pct"),
        "reward_pct_of_max": _metric(user_info, "Team/reward_pct_of_max"),
        "action_parse_rate": float(episode.get("action_parse_rate", 0.0) or 0.0),
        "input_tokens": usage["input_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "total_tokens": total_tokens,
        "model_calls": usage["model_call_count"],
        "provider_requests": usage["provider_request_count"],
        "transport_errors": usage["transport_error_count"],
        "episode_wall_seconds": float(episode.get("episode_wall_seconds", 0.0) or 0.0),
        "wall_seconds_per_step": float(episode.get("episode_wall_seconds", 0.0) or 0.0)
        / max(steps, 1),
        "commander_plan_wall_seconds": float(
            episode.get("commander_plan_phase_wall_seconds", 0.0) or 0.0
        ),
        "worker_round_wall_seconds": float(episode.get("worker_round_wall_seconds", 0.0) or 0.0),
        "plan_calls": int(commander.get("plan_calls", 0) or 0),
        "valid_plans": int(commander.get("valid_plans", 0) or 0),
        "plan_parse_rate": _metric(commander, "plan_parse_rate"),
        "active_plan_coverage": _metric(commander, "active_plan_coverage"),
        "status_attempts": int(commander.get("status_attempts", 0) or 0),
        "valid_status": int(commander.get("valid_status", 0) or 0),
        "status_parse_rate": _metric(commander, "status_parse_rate"),
        "status_attempt_rate": _metric(commander, "status_attempt_rate"),
        "valid_status_coverage": _metric(commander, "valid_status_coverage"),
        "assignment_ack_rate": _metric(commander, "assignment_ack_rate"),
        "assignment_switches": int(commander.get("assignment_switches", 0) or 0),
        "stale_status_attempts": int(commander.get("stale_status", 0) or 0),
        "wrong_assignment_status_attempts": int(commander.get("wrong_assignment_status", 0) or 0),
        "unauthorized_plan_attempts": int(commander.get("unauthorized_plan_attempts", 0) or 0),
        "accepted_unauthorized_plans": int(commander.get("accepted_unauthorized_plans", 0) or 0),
        "accepted_stale_statuses": int(commander.get("accepted_stale_statuses", 0) or 0),
        "hidden_state_leak_guard_violations": int(
            commander.get("hidden_state_leak_guard_violations", 0) or 0
        ),
        "commander_metrics_available": bool(commander),
    }


def _ratio_regression(treatment: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return treatment / baseline - 1.0


def _weighted_rate(rows: list[dict], numerator: str, denominator: str) -> float:
    return sum(row[numerator] for row in rows) / max(sum(row[denominator] for row in rows), 1)


def _stage_30_gate(rows: list[dict]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows if row["arm"] in PRIMARY_ARMS}
    missing = [arm for arm in PRIMARY_ARMS if arm not in by_arm]
    if missing:
        return {
            "passed": False,
            "available": False,
            "reason": "missing primary arm(s): " + ", ".join(missing),
        }
    source = by_arm["baseline"]
    treatment = by_arm["embodied_commander_broadcast"]
    checks = {
        "complete_no_transport_errors": all(
            row["transport_errors"] == 0 for row in (source, treatment)
        ),
        "at_least_one_valid_plan": treatment["valid_plans"] >= 1,
        "active_plan_coverage_at_least_80pct": (
            treatment["active_plan_coverage"] is not None
            and treatment["active_plan_coverage"] >= 0.80
        ),
        "action_parse_at_least_95pct": treatment["action_parse_rate"] >= 0.95,
        "status_parse_at_least_90pct": (
            treatment["status_parse_rate"] is not None and treatment["status_parse_rate"] >= 0.90
        ),
        "zero_accepted_unauthorized_plans": (treatment["accepted_unauthorized_plans"] == 0),
        "zero_accepted_stale_statuses": treatment["accepted_stale_statuses"] == 0,
        "zero_hidden_state_guard_violations": (
            treatment["hidden_state_leak_guard_violations"] == 0
        ),
    }
    return {
        "available": True,
        "passed": all(checks.values()),
        "checks": checks,
        "manual_semantic_trace_review": _availability(
            None,
            "requires human review of debug prompts and actions before efficacy claims",
        ),
    }


def _stage_100_gate(rows: list[dict]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows if row["arm"] in PRIMARY_ARMS}
    missing = [arm for arm in PRIMARY_ARMS if arm not in by_arm]
    if missing:
        return {
            "passed": False,
            "available": False,
            "reason": "missing primary arm(s): " + ", ".join(missing),
        }
    source = by_arm["baseline"]
    treatment = by_arm["embodied_commander_broadcast"]
    checks = {
        "complete_no_transport_errors": all(
            row["transport_errors"] == 0 for row in (source, treatment)
        ),
        "at_least_one_valid_plan": treatment["valid_plans"] >= 1,
        "valid_plan_calls_at_least_90pct": (
            treatment["plan_parse_rate"] is not None and treatment["plan_parse_rate"] >= 0.90
        ),
        "active_plan_coverage_at_least_90pct": (
            treatment["active_plan_coverage"] is not None
            and treatment["active_plan_coverage"] >= 0.90
        ),
        "both_action_parse_rates_at_least_95pct": all(
            row["action_parse_rate"] >= 0.95 for row in (source, treatment)
        ),
        "valid_status_reports_at_least_90pct": (
            treatment["status_parse_rate"] is not None and treatment["status_parse_rate"] >= 0.90
        ),
        "valid_status_coverage_at_least_80pct": (
            treatment["valid_status_coverage"] is not None
            and treatment["valid_status_coverage"] >= 0.80
        ),
        "zero_accepted_unauthorized_plans": treatment["accepted_unauthorized_plans"] == 0,
        "zero_accepted_stale_statuses": treatment["accepted_stale_statuses"] == 0,
        "zero_hidden_state_guard_violations": (
            treatment["hidden_state_leak_guard_violations"] == 0
        ),
    }
    return {
        "available": True,
        "passed": all(checks.values()),
        "checks": checks,
        "manual_semantic_trace_review": _availability(
            None,
            "requires human review of all planning calls and sampled action traces",
        ),
    }


def _stage_200_gate(rows: list[dict], seeds: list[int]) -> dict[str, Any]:
    indexed = {(row["arm"], row["seed"]): row for row in rows}
    missing = [(arm, seed) for arm in PRIMARY_ARMS for seed in seeds if (arm, seed) not in indexed]
    if missing:
        return {
            "passed": False,
            "available": False,
            "reason": f"missing {len(missing)} primary paired episode(s)",
        }
    source = [indexed[("baseline", seed)] for seed in seeds]
    treatment = [indexed[("embodied_commander_broadcast", seed)] for seed in seeds]
    achievement_deltas = []
    for base, treated in zip(source, treatment, strict=True):
        left = treated["achievement_pct"]
        right = base["achievement_pct"]
        achievement_deltas.append(None if left is None or right is None else left - right)
    parse_drops = [
        base["action_parse_rate"] - treated["action_parse_rate"]
        for base, treated in zip(source, treatment, strict=True)
    ]
    token_regressions = [
        _ratio_regression(treated["total_tokens"], base["total_tokens"])
        for base, treated in zip(source, treatment, strict=True)
    ]
    wall_regressions = [
        _ratio_regression(treated["wall_seconds_per_step"], base["wall_seconds_per_step"])
        for base, treated in zip(source, treatment, strict=True)
    ]
    achievement_available = all(delta is not None for delta in achievement_deltas)
    token_available = all(value is not None for value in token_regressions)
    wall_available = all(value is not None for value in wall_regressions)
    plan_rate = _weighted_rate(treatment, "valid_plans", "plan_calls")
    status_rate = _weighted_rate(treatment, "valid_status", "status_attempts")
    coverage = sum(row["active_plan_coverage"] or 0.0 for row in treatment) / len(treatment)
    mean_token_regression = (
        sum(token_regressions) / len(token_regressions) if token_available else None
    )
    mean_wall_regression = sum(wall_regressions) / len(wall_regressions) if wall_available else None
    checks = {
        "all_three_pairs_complete": len(seeds) == 3,
        "zero_transport_errors": all(row["transport_errors"] == 0 for row in rows),
        "achievements_improve_on_at_least_two_seeds": (
            achievement_available and sum(delta > 0 for delta in achievement_deltas) >= 2
        ),
        "action_parse_drop_over_2pp_on_at_most_one_seed": (
            sum(drop > 0.02 for drop in parse_drops) <= 1
        ),
        "active_plan_coverage_at_least_90pct": coverage >= 0.90,
        "valid_plan_calls_at_least_90pct": plan_rate >= 0.90,
        "valid_status_reports_at_least_90pct": status_rate >= 0.90,
        "zero_accepted_authority_or_stale_violations": all(
            row["accepted_unauthorized_plans"] == 0 and row["accepted_stale_statuses"] == 0
            for row in treatment
        ),
        "mean_token_regression_at_most_35pct": (
            mean_token_regression is not None and mean_token_regression <= 0.35
        ),
        "mean_wall_per_step_regression_at_most_35pct": (
            mean_wall_regression is not None and mean_wall_regression <= 0.35
        ),
    }
    return {
        "available": True,
        "passed": all(checks.values()),
        "checks": checks,
        "achievement_deltas": achievement_deltas,
        "action_parse_drops": parse_drops,
        "mean_active_plan_coverage": coverage,
        "weighted_plan_parse_rate": plan_rate,
        "weighted_status_parse_rate": status_rate,
        "mean_token_regression": _availability(
            mean_token_regression, "baseline token usage was zero"
        ),
        "mean_wall_per_step_regression": _availability(
            mean_wall_regression, "baseline wall time was zero"
        ),
        "manual_semantic_trace_review": _availability(
            None,
            "requires human review; automated gates cover structural leakage only",
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.{digits}f}"


def summarize(root: Path, results_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    manifest_path = root / "commander_study_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing commander study manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = str(manifest["stage"])
    seeds = [int(seed) for seed in manifest["seeds"]]
    arms = tuple(manifest["arms"])
    rows = [_row(root, arm, index, seed) for arm in arms for index, seed in enumerate(seeds)]
    if stage == "30":
        gate = _stage_30_gate(rows)
    elif stage == "100":
        gate = _stage_100_gate(rows)
    else:
        gate = _stage_200_gate(rows, seeds)
    paired = []
    indexed = {(row["arm"], row["seed"]): row for row in rows}
    for seed in seeds:
        if all((arm, seed) in indexed for arm in PRIMARY_ARMS):
            source = indexed[("baseline", seed)]
            treatment = indexed[("embodied_commander_broadcast", seed)]
            paired.append(
                {
                    "seed": seed,
                    "achievement_delta": (
                        None
                        if source["achievement_pct"] is None or treatment["achievement_pct"] is None
                        else treatment["achievement_pct"] - source["achievement_pct"]
                    ),
                    "return_delta": treatment["team_return"] - source["team_return"],
                    "action_parse_delta": treatment["action_parse_rate"]
                    - source["action_parse_rate"],
                    "token_regression": _ratio_regression(
                        treatment["total_tokens"], source["total_tokens"]
                    ),
                    "wall_per_step_regression": _ratio_regression(
                        treatment["wall_seconds_per_step"],
                        source["wall_seconds_per_step"],
                    ),
                }
            )
    summary = {
        "schema_version": "alem-dice-embodied-commander-summary-v1",
        "manifest": manifest,
        "episodes": rows,
        "paired_contrasts": paired,
        "preregistered_gate": gate,
        "interpretation": {
            "30": "wiring/qualitative gate only",
            "100": "one-seed intermediate evaluation; descriptive, not confirmatory",
            "200": "three-seed exploratory estimate; not confirmatory evidence",
        }.get(stage, "exploratory evaluation"),
    }
    summary_path = root / "commander_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = root / "commander_episodes.csv"
    columns = tuple(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Embodied Commander {stage}-Step Study",
        "",
        {
            "30": "This stage is a wiring and qualitative gate, not an efficacy result.",
            "100": (
                "This is a one-seed, 100-tick Easy comparison; findings are descriptive, "
                "not confirmatory evidence."
            ),
            "200": ("This is a three-seed Easy exploratory comparison, not confirmatory evidence."),
        }.get(stage, "This is an exploratory comparison."),
        "",
        f"Preregistered automated gate: **{'PASS' if gate.get('passed') else 'FAIL'}**.",
        "",
        "## Episode results",
        "",
        "| Arm | Seed | Steps/end | Ach. % | Return | Parse | Plan coverage | Plan valid | Status valid/coverage | Calls/requests/errors | Tokens | Wall/step |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['steps']}/{row['termination_reason']} | "
            f"{_fmt(row['achievement_pct'])} | "
            f"{_fmt(row['team_return'])} | {_fmt(100 * row['action_parse_rate'], 1)}% | "
            f"{_fmt(row['active_plan_coverage'])} | {_fmt(row['plan_parse_rate'])} | "
            f"{_fmt(row['status_parse_rate'])}/{_fmt(row['valid_status_coverage'])} | "
            f"{int(row['model_calls'])}/"
            f"{int(row['provider_requests'])}/{int(row['transport_errors'])} | "
            f"{int(row['total_tokens']):,} | "
            f"{_fmt(row['wall_seconds_per_step'])} |"
        )
    lines.extend(["", "## Paired Source contrasts", ""])
    for contrast in paired:
        lines.append(
            f"- Seed {contrast['seed']}: achievement Δ {_fmt(contrast['achievement_delta'])}; "
            f"return Δ {_fmt(contrast['return_delta'])}; parse Δ "
            f"{_fmt(contrast['action_parse_delta'])}; token regression "
            f"{_fmt(contrast['token_regression'])}; wall/step regression "
            f"{_fmt(contrast['wall_per_step_regression'])}."
        )
    lines.extend(["", "## Gate details", ""])
    for name, passed in (gate.get("checks") or {}).items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Audit limits",
            "",
            "- Automated leakage checks establish construction-level data boundaries and zero accepted unauthorized/stale records; semantic trace review remains manual.",
            "- Failed attempts remain in token and transport accounting through the append-only attempt ledger, while outcome metrics use only complete canonical episodes.",
            "- The baseline is the unchanged Source action path; the treatment adds a serial Agent 0 planning call, leased assignments, executor authority prompts, and SCP1 status validation.",
        ]
    )
    report_path = root / "commander_study_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results_dir.mkdir(parents=True, exist_ok=True)
    published_report = results_dir / f"{root.name}.md"
    published_json = results_dir / f"{root.name}.json"
    shutil.copy2(report_path, published_report)
    shutil.copy2(summary_path, published_json)
    return published_report, published_json


if __name__ == "__main__":
    args = _parser().parse_args()
    report, payload = summarize(args.run_dir, args.results_dir)
    print(f"Report: {report}")
    print(f"JSON: {payload}")
