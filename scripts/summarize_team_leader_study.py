#!/usr/bin/env python3
"""Summarize the matched bodyless-team-leader pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

ARMS = ("baseline", "leader_peer", "leader_no_peer")
GPT54 = {"uncached_input": 2.50, "cached_input": 0.25, "output": 15.00}
LUNA = {"uncached_input": 1.00, "cached_input": 0.10, "output": 6.00}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def _episode(root: Path, arm: str) -> tuple[Path, dict]:
    path = root / arm / "easy" / "alem" / "default" / "default_run_00.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed episode for {arm}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_status") != "complete":
        raise ValueError(f"Episode is not complete for {arm}: {path}")
    return path, payload


def _attempt_usage(episode_path: Path, episode: dict) -> dict[str, float]:
    ledger = episode_path.parent / "attempt_ledger.jsonl"
    records = []
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("episode_index") == 0:
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
    result = {key: sum(float(record.get(key, 0) or 0) for record in records) for key in keys}
    for participant_id in range(4):
        for suffix in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "model_call_count",
        ):
            key = f"agent_{participant_id}_{suffix}"
            result[key] = sum(float(record.get(key, 0) or 0) for record in records)
    result["model_usage_records"] = [
        usage_record
        for record in records
        for usage_record in (record.get("model_usage_records") or [])
        if isinstance(usage_record, dict)
    ]
    return result


def _cost(usage: dict[str, float], rates: dict[str, float]) -> float:
    uncached = max(usage["input_tokens"] - usage["cached_tokens"], 0)
    return (
        uncached * rates["uncached_input"]
        + usage["cached_tokens"] * rates["cached_input"]
        + usage["output_tokens"] * rates["output"]
    ) / 1_000_000


def _gpt54_cost_with_context_pricing(usage: dict) -> float:
    records = usage.get("model_usage_records") or []
    if not records:
        return _cost(usage, GPT54)
    total = 0.0
    for record in records:
        input_tokens = float(record.get("input_tokens", 0) or 0)
        cached_tokens = float(record.get("cached_tokens", 0) or 0)
        output_tokens = float(record.get("output_tokens", 0) or 0)
        input_multiplier = 2.0 if input_tokens > 272_000 else 1.0
        output_multiplier = 1.5 if input_tokens > 272_000 else 1.0
        total += (
            max(input_tokens - cached_tokens, 0)
            * GPT54["uncached_input"]
            * input_multiplier
            + cached_tokens * GPT54["cached_input"] * input_multiplier
            + output_tokens * GPT54["output"] * output_multiplier
        ) / 1_000_000
    return total


def _metric(ui: dict, key: str) -> float:
    return float(ui.get(key, 0.0) or 0.0)


def _arm_row(root: Path, arm: str) -> dict:
    episode_path, episode = _episode(root, arm)
    usage = _attempt_usage(episode_path, episode)
    ui = episode.get("user_info", {}) or {}
    communication = episode.get("communication_metrics", {}) or {}
    delivery_bytes = sum(
        float(values.get("delivery_bytes", 0) or 0)
        for channel, values in communication.items()
        if channel != "leader_assignment_prompt"
    )
    emitted_messages = sum(
        float(values.get("emitted_messages", 0) or 0)
        for values in communication.values()
    )
    total_tokens = usage["input_tokens"] + usage["output_tokens"]
    coord_pp = 100 * _metric(ui, "Team/coord_reward_pct_of_max")
    worker_usage = {
        "input_tokens": sum(usage[f"agent_{idx}_input_tokens"] for idx in range(3)),
        "cached_tokens": sum(usage[f"agent_{idx}_cached_tokens"] for idx in range(3)),
        "output_tokens": sum(usage[f"agent_{idx}_output_tokens"] for idx in range(3)),
    }
    leader_usage = {
        "input_tokens": usage["agent_3_input_tokens"],
        "cached_tokens": usage["agent_3_cached_tokens"],
        "output_tokens": usage["agent_3_output_tokens"],
    }
    return {
        "arm": arm,
        "seed": episode.get("seed"),
        "steps": episode.get("num_steps"),
        "termination_reason": episode.get("termination_reason"),
        "team_return": float(episode.get("episode_return", 0.0) or 0.0),
        "base_pct": 100 * _metric(ui, "Team/normal_reward_pct_of_max"),
        "coord_pct": coord_pp,
        "total_pct": 100 * _metric(ui, "Team/reward_pct_of_max"),
        "normal_achievement_pct": 100 * _metric(ui, "Team/normal_achievement_pct"),
        "coord_achievement_pct": 100 * _metric(ui, "Team/coordination_achievement_pct"),
        "achievement_pct": 100 * _metric(ui, "Team/achievement_pct"),
        "action_parse_rate": float(episode.get("action_parse_rate", 0.0) or 0.0),
        "logical_model_calls": usage["model_call_count"],
        "provider_requests": usage["provider_request_count"],
        "transport_errors": usage["transport_error_count"],
        "input_tokens": usage["input_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "total_tokens": total_tokens,
        "model_latency_seconds": usage["model_latency_seconds"],
        "episode_wall_seconds": float(episode.get("episode_wall_seconds", 0.0) or 0.0),
        "leader_phase_wall_seconds": float(
            episode.get("leader_phase_wall_seconds", 0.0) or 0.0
        ),
        "worker_round_wall_seconds": float(
            episode.get("worker_round_wall_seconds", 0.0) or 0.0
        ),
        "worker_input_tokens": sum(
            usage[f"agent_{idx}_input_tokens"] for idx in range(3)
        ),
        "worker_output_tokens": sum(
            usage[f"agent_{idx}_output_tokens"] for idx in range(3)
        ),
        "leader_input_tokens": usage["agent_3_input_tokens"],
        "leader_output_tokens": usage["agent_3_output_tokens"],
        "emitted_messages": emitted_messages,
        "delivered_bytes": delivery_bytes,
        "assignment_prompt_bytes": float(
            communication.get("leader_assignment_prompt", {}).get("payload_bytes", 0) or 0
        ),
        "gpt54_cost_usd": _gpt54_cost_with_context_pricing(usage),
        "worker_gpt54_cost_usd": _cost(worker_usage, GPT54),
        "leader_gpt54_cost_usd": _cost(leader_usage, GPT54),
        "luna_same_token_cost_usd": _cost(usage, LUNA),
        "large_context_call_count": sum(
            int(record.get("input_tokens", 0) or 0) > 272_000
            for record in usage.get("model_usage_records", [])
        ),
        "coord_pp_per_million_tokens": (
            coord_pp / (total_tokens / 1_000_000) if total_tokens and coord_pp else None
        ),
        "coord_pp_per_delivered_mb": (
            coord_pp / (delivery_bytes / 1_000_000) if delivery_bytes and coord_pp else None
        ),
        "leader": episode.get("leader"),
        "communication_metrics": communication,
    }


def _delta(left: dict, right: dict) -> dict:
    numeric = (
        "base_pct",
        "coord_pct",
        "total_pct",
        "team_return",
        "total_tokens",
        "delivered_bytes",
        "gpt54_cost_usd",
        "model_latency_seconds",
        "episode_wall_seconds",
    )
    return {key: left[key] - right[key] for key in numeric}


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "undefined"
    return f"{value:.{digits}f}"


def summarize(root: Path, results_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    manifest = json.loads((root / "study_manifest.json").read_text(encoding="utf-8"))
    rows = [_arm_row(root, arm) for arm in ARMS]
    by_arm = {row["arm"]: row for row in rows}
    contrasts = {
        "leader_peer_minus_baseline": _delta(by_arm["leader_peer"], by_arm["baseline"]),
        "leader_no_peer_minus_baseline": _delta(
            by_arm["leader_no_peer"], by_arm["baseline"]
        ),
        "leader_peer_minus_leader_no_peer": _delta(
            by_arm["leader_peer"], by_arm["leader_no_peer"]
        ),
    }
    summary = {
        "schema_version": "alem-dice-team-leader-summary-v1",
        "manifest": manifest,
        "arms": rows,
        "contrasts": contrasts,
        "interpretation_limits": [
            "One Easy seed; raw descriptive differences only.",
            "The evaluator truncates at 200 steps while workers see the 10,000-step paper horizon.",
            "The published GPT-5.4 High reference uses a longer, multi-seed protocol.",
            "The Luna estimate holds token usage fixed and is not a matched Luna treatment arm.",
        ],
    }
    summary_path = root / "study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = root / "episodes.csv"
    public_columns = [key for key in rows[0] if key not in {"leader", "communication_metrics"}]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in public_columns})

    lines = [
        "# Bodyless Team-Leader 200-Step Pilot",
        "",
        "This is a one-seed Easy, paper-interface-matched pilot. It is not a reproduction of Alem's full multi-seed, 10,000-step evaluation and supports no significance claims.",
        "",
        "## Results",
        "",
        "| Arm | Base % | Coord. % | Total % | Return | Tokens | Delivered bytes | GPT-5.4 cost | Parse rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {_fmt(row['base_pct'])} | {_fmt(row['coord_pct'])} | "
            f"{_fmt(row['total_pct'])} | {_fmt(row['team_return'])} | "
            f"{int(row['total_tokens']):,} | {int(row['delivered_bytes']):,} | "
            f"${_fmt(row['gpt54_cost_usd'], 4)} | {_fmt(100 * row['action_parse_rate'], 1)}% |"
        )
    lines.extend(["", "## Raw contrasts", ""])
    for name, values in contrasts.items():
        lines.append(
            f"- `{name}`: Coord. {_fmt(values['coord_pct'])} pp; Total {_fmt(values['total_pct'])} pp; "
            f"tokens {values['total_tokens']:+,.0f}; delivered bytes {values['delivered_bytes']:+,.0f}; "
            f"cost ${values['gpt54_cost_usd']:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Communication and cost interpretation",
            "",
            "Communication accounting includes direct worker traffic, legal-view forwarding to the leader, leader assignments, and separately reported repeated assignment prompt bytes. Reasoning tokens are a subset of output tokens and are not added twice.",
            "",
            "At current standard rates, GPT-5.4's uncached-input, cached-input, and output rates are each 2.5× the corresponding GPT-5.6 Luna rates. The JSON report includes a same-token Luna counterfactual; it cannot estimate how Luna's `none` reasoning mode would change token use or behavior.",
            "",
            "Published Easy GPT-5.4 High context is 14.1% Base, 7.0% Coordination, and 11.1% Total, but those values are not directly comparable to this 200-step, one-seed pilot.",
            "",
            "## Limits",
            "",
            "- Fixed arm order is baseline, leader-peer, then leader-no-peer.",
            "- Leader treatments add serial model computation as well as a different communication topology.",
            "- Ratios with zero coordination gain are reported as undefined in the JSON artifact.",
        ]
    )
    report_path = root / "study_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results_dir.mkdir(parents=True, exist_ok=True)
    stem = root.name
    published_report = results_dir / f"{stem}.md"
    published_json = results_dir / f"{stem}.json"
    shutil.copy2(report_path, published_report)
    shutil.copy2(summary_path, published_json)
    return published_report, published_json


if __name__ == "__main__":
    args = _parser().parse_args()
    report, payload = summarize(args.run_dir, args.results_dir)
    print(f"Report: {report}")
    print(f"JSON: {payload}")
