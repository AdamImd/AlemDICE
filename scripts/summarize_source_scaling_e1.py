#!/usr/bin/env python3
"""Extract and plot E1 Source-baseline performance versus physical agent count.

Expected layout (additional directories such as ``easy`` are allowed):

    RUN_ROOT/n1/.../alem/default/default_run_00.json
    RUN_ROOT/n2/.../alem/default/default_run_00.json

The canonical ``physical_worker_count`` field is authoritative; the ``n<N>``
path component is checked when present. Only complete canonical episode
artifacts are accepted. Older artifacts without ``performance_metrics`` are
supported from their existing ``user_info`` counters, although exact N=1
survival exposure is unavailable in that legacy format.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_UTILS = ROOT / "baselines" / "llm" / "eval_utils"
if str(EVAL_UTILS) not in sys.path:
    sys.path.insert(0, str(EVAL_UTILS))

from performance_metrics import (  # noqa: E402
    PERFORMANCE_METRICS_SCHEMA,
    build_performance_metrics,
)

EPISODE_PATTERN = re.compile(r".+_run_(\d+)\.json$")
POPULATION_PATTERN = re.compile(r"n(\d+)$", re.IGNORECASE)
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
    "environment_steps_completed",
    "agent_turns_submitted",
    "alive_agent_turns",
    "actionable_agent_turns",
    "survival_fraction",
    "actionable_fraction",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "model_call_count",
    "provider_request_count",
    "input_tokens_per_agent_turn",
    "delivered_bytes",
    "delivered_bytes_per_agent_turn",
    "episode_wall_seconds",
    "mean_tick_wall_seconds",
)
CSV_FIELDS = (
    "artifact_path",
    "num_agents",
    "seed",
    "episode_index",
    "termination_reason",
    "used_legacy_metric_fallback",
    *SUMMARY_METRICS,
    "achievement_base_percent",
    "achievement_coord_percent",
    "achievement_total_percent",
    "completed_agent_turn_capacity",
    "classified_action_turns",
)
HEADLINE_METRICS = (
    ("paper_total_percent", "Paper Total (%)"),
    ("episode_return", "Mean per-agent return"),
    ("team_unique_total_achievements", "Unique team achievements"),
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


def episode_row(path: Path, root: Path) -> dict[str, Any]:
    """Extract one validated canonical episode into the stable E1 CSV schema."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read episode JSON: {exc}") from exc
    payload = _validate_episode(payload, path)

    match = EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path}: filename does not contain a canonical episode index")
    episode_index = int(match.group(1))
    num_agents = _count(payload.get("physical_worker_count"))
    path_agents = _path_population(path, root)
    if num_agents is None:
        num_agents = path_agents
    if num_agents is None or num_agents < 1:
        raise ValueError(f"{path}: missing positive physical_worker_count")
    if path_agents is not None and path_agents != num_agents:
        raise ValueError(
            f"{path}: n{path_agents} path disagrees with physical_worker_count={num_agents}"
        )
    seed = _count(payload.get("seed"))
    if seed is None:
        raise ValueError(f"{path}: missing non-negative integer seed")

    performance = payload.get("performance_metrics")
    used_legacy_fallback = performance is None
    if performance is None:
        performance = build_performance_metrics(payload, num_agents)
    elif not isinstance(performance, dict):
        raise ValueError(f"{path}: performance_metrics is not an object")
    elif performance.get("schema_version") != PERFORMANCE_METRICS_SCHEMA:
        raise ValueError(
            f"{path}: unsupported performance_metrics schema {performance.get('schema_version')!r}"
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
    delivery_bytes = _delivered_bytes(payload)
    user_info = payload.get("user_info")
    if not isinstance(user_info, dict):
        user_info = {}

    return {
        "artifact_path": str(path.relative_to(root)),
        "num_agents": num_agents,
        "seed": seed,
        "episode_index": episode_index,
        "termination_reason": payload.get("termination_reason"),
        "used_legacy_metric_fallback": used_legacy_fallback,
        "paper_base_percent": _number(paper.get("base")),
        "paper_coord_percent": _number(paper.get("coord")),
        "paper_total_percent": _number(paper.get("total")),
        "episode_return": _number(payload.get("episode_return")),
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
        "action_parse_rate": _number(payload.get("action_parse_rate")),
        "environment_steps_completed": _count(exposure.get("environment_steps_completed")),
        "completed_agent_turn_capacity": _count(exposure.get("completed_agent_turn_capacity")),
        "agent_turns_submitted": _count(exposure.get("agent_turns_submitted")),
        "classified_action_turns": _count(exposure.get("classified_action_turns")),
        "alive_agent_turns": _count(exposure.get("alive_agent_turns")),
        "actionable_agent_turns": _count(exposure.get("actionable_agent_turns")),
        "survival_fraction": _number(exposure.get("survival_fraction")),
        "actionable_fraction": _number(exposure.get("actionable_fraction")),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "model_call_count": _count(payload.get("model_call_count")),
        "provider_request_count": _count(payload.get("provider_request_count")),
        "input_tokens_per_agent_turn": _divide(input_tokens, turn_denominator),
        "delivered_bytes": delivery_bytes,
        "delivered_bytes_per_agent_turn": _divide(delivery_bytes, turn_denominator),
        "episode_wall_seconds": _number(payload.get("episode_wall_seconds")),
        "mean_tick_wall_seconds": _number(payload.get("mean_tick_wall_seconds")),
    }


def discover_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        path
        for path in root.rglob("*_run_*.json")
        if EPISODE_PATTERN.fullmatch(path.name) and "attempt_archive" not in path.parts
    )
    if not paths:
        raise ValueError(f"No canonical episode JSON files found below {root}")
    rows = [episode_row(path, root) for path in paths]
    identities = [(row["num_agents"], row["seed"], row["episode_index"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate (num_agents, seed, episode_index) episode identity")
    return sorted(rows, key=lambda row: (row["num_agents"], row["seed"], row["episode_index"]))


def bootstrap_mean_ci(
    values: list[int | float],
    *,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, int | float | None]:
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


def _format_value(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{float(number):.{digits}f}"


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[int, dict[str, dict[str, Any]]],
    root: Path,
) -> None:
    lines = [
        "# E1 Source Scaling Summary",
        "",
        f"Source root: `{root}`",
        "",
        "| Agents | Episodes | Total % | Base % | Coord % | Per-agent return | "
        "Unique team achievements | Summed agent achievements | Survival |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
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
                    _format_value(metrics["survival_fraction"]["mean"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Intervals in `summary.json` and the figure are nonparametric 95% "
            "episode-bootstrap intervals; raw seed values are retained in `episodes.csv`.",
            "",
            "Base/Coord/Total are reward-weighted paper scores on a 0–100 scale. "
            "Achievement counts are cumulative binary first-unlocks, not repeated events. "
            "Team-unique counts each achievement type once across the team; the summed "
            "agent count can count the same type once for every attaining agent.",
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
) -> None:
    """Write a dependency-light table fragment using the exact JSON estimates."""

    counts = defaultdict(int)
    for row in rows:
        counts[row["num_agents"]] += 1

    def _tex_value(value: Any, digits: int) -> str:
        number = _number(value)
        return "--" if number is None else f"{float(number):.{digits}f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{r r r r r r r r r r}",
        r"\hline",
        r"$N$ & Episodes & Base \% & Coord. \% & Total \% & "
        r"Return/agent & Unique total & $\sum$ agent total & Steps & Tick (s) \\",
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
                    _tex_value(metrics["team_unique_total_achievements"]["mean"], 2),
                    _tex_value(metrics["summed_agent_total_achievements"]["mean"], 2),
                    _tex_value(metrics["environment_steps_completed"]["mean"], 1),
                    _tex_value(metrics["mean_tick_wall_seconds"]["mean"], 3),
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\hline",
            r"\end{tabular}",
            r"\caption{Source-baseline population screen. Entries are episode means; "
            r"Coordination is undefined for the single-agent wrapper.}",
            r"\label{tab:e1-source-scaling}",
            r"\end{table}",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[int, dict[str, dict[str, Any]]],
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
    figure.suptitle("Alem Source baseline: fixed-world population curve")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")
    root = args.run_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Run root does not exist: {root}")
    rows = discover_rows(root)
    summary = summarize_rows(
        rows,
        reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary_payload = {
        "schema_version": "alem-dice-e1-scaling-summary-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_root": str(root),
        "episode_count": len(rows),
        "populations": {
            str(num_agents): {
                "episode_count": sum(row["num_agents"] == num_agents for row in rows),
                "metrics": metrics,
            }
            for num_agents, metrics in summary.items()
        },
        "bootstrap": {
            "method": "episode resampling within population",
            "confidence": 0.95,
            "repetitions": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out / "summary.md", rows, summary, root)
    write_tex_table(args.out / "summary_table.tex", rows, summary)
    write_plot(args.out / "performance_vs_agents.png", rows, summary)
    print(f"Wrote {len(rows)} episodes across {len(summary)} populations to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
