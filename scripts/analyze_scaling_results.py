#!/usr/bin/env python3
"""Generate derived scaling metrics, publication figures, and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINE = "#4C78A8"
PROTOCOL = "#E45756"
BASE = "#72B7B2"
COORD = "#F2CF5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-episodes",
        type=Path,
        default=Path("Results/e1_source_scaling/episodes.csv"),
    )
    parser.add_argument(
        "--matched",
        type=Path,
        default=Path("Results/e3b1_scaling_matched/absolute_team_performance.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/scaling_analysis"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct_change(treatment: float, baseline: float) -> float:
    return 100.0 * (treatment / baseline - 1.0)


def elasticity(y0: float, y1: float, n0: int, n1: int) -> float:
    return math.log(y1 / y0) / math.log(n1 / n0)


def validate(
    low_rows: list[dict[str, str]], matched_rows: list[dict[str, str]]
) -> None:
    low_cells = {(int(row["num_agents"]), int(row["seed"])) for row in low_rows}
    expected_low = {
        (agents, seed)
        for agents in (1, 2, 3, 4, 6)
        for seed in (13100, 13101, 13102)
    }
    if low_cells != expected_low:
        raise ValueError("low-scale input is not the complete 5x3 screen")
    if any(row["analysis_status"] != "complete" for row in low_rows):
        raise ValueError("low-scale input contains a non-complete episode")

    matched_cells = {
        (row["condition"], int(row["num_agents"])) for row in matched_rows
    }
    expected_matched = {
        (condition, agents)
        for condition in ("baseline", "team_formation")
        for agents in (8, 16, 32)
    }
    if matched_cells != expected_matched:
        raise ValueError("matched input is not the complete 2x3 comparison")
    if any(int(row["steps"]) != 200 or int(row["seed"]) != 14100 for row in matched_rows):
        raise ValueError("matched cells must all be seed 14100 and 200 steps")


def summarize_low(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    for agents in (1, 2, 3, 4, 6):
        cell = [row for row in rows if int(row["num_agents"]) == agents]
        total_return = [float(row["episode_return"]) * agents for row in cell]
        unlocks = [float(row["summed_agent_total_achievements"]) for row in cell]
        unique = [float(row["team_unique_total_achievements"]) for row in cell]
        summary.append(
            {
                "num_agents": agents,
                "replicates": len(cell),
                "total_return_mean": statistics.mean(total_return),
                "total_return_min": min(total_return),
                "total_return_max": max(total_return),
                "total_return_cv": statistics.stdev(total_return)
                / statistics.mean(total_return),
                "return_per_agent": statistics.mean(total_return) / agents,
                "unlocks_mean": statistics.mean(unlocks),
                "unlocks_min": min(unlocks),
                "unlocks_max": max(unlocks),
                "unlocks_cv": statistics.stdev(unlocks) / statistics.mean(unlocks),
                "unlocks_per_agent": statistics.mean(unlocks) / agents,
                "unique_mean": statistics.mean(unique),
            }
        )
    return summary


def matched_index(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {(row["condition"], int(row["num_agents"])): row for row in rows}


def write_derived_csv(
    path: Path,
    low: list[dict[str, float | int]],
    matched: dict[tuple[str, int], dict[str, str]],
) -> None:
    fields = [
        "campaign",
        "condition",
        "num_agents",
        "replicates",
        "total_team_return",
        "return_per_agent",
        "summed_unlocks",
        "unlocks_per_agent",
        "team_unique_total",
        "summed_base",
        "summed_coord",
        "coord_share_percent",
    ]
    output: list[dict[str, str | int | float]] = []
    for row in low:
        output.append(
            {
                "campaign": "three_seed_screen",
                "condition": "baseline",
                "num_agents": row["num_agents"],
                "replicates": row["replicates"],
                "total_team_return": row["total_return_mean"],
                "return_per_agent": row["return_per_agent"],
                "summed_unlocks": row["unlocks_mean"],
                "unlocks_per_agent": row["unlocks_per_agent"],
                "team_unique_total": row["unique_mean"],
                "summed_base": "",
                "summed_coord": "",
                "coord_share_percent": "",
            }
        )
    for agents in (8, 16, 32):
        for condition in ("baseline", "team_formation"):
            row = matched[(condition, agents)]
            total_return = float(row["total_team_return"])
            summed = float(row["summed_total"])
            coord = float(row["summed_coord"])
            output.append(
                {
                    "campaign": "matched_comparison",
                    "condition": condition,
                    "num_agents": agents,
                    "replicates": 1,
                    "total_team_return": total_return,
                    "return_per_agent": total_return / agents,
                    "summed_unlocks": summed,
                    "unlocks_per_agent": summed / agents,
                    "team_unique_total": float(row["team_unique_total"]),
                    "summed_base": float(row["summed_base"]),
                    "summed_coord": coord,
                    "coord_share_percent": 100.0 * coord / summed,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )


def finish_figure(fig: plt.Figure, path: Path, title: str, subject: str) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Subject": subject,
            "Creator": "AlemDICE",
        },
    )
    plt.close(fig)


def plot_uplift(
    matched: dict[tuple[str, int], dict[str, str]], path: Path
) -> None:
    populations = (8, 16, 32)
    metrics = (
        ("total_team_return", "Total return"),
        ("summed_total", "Agent task completions"),
        ("team_unique_total", "Unique task types"),
    )
    values_by_metric = []
    for field, _ in metrics:
        values_by_metric.append(
            [
                pct_change(
                    float(matched[("team_formation", n)][field]),
                    float(matched[("baseline", n)][field]),
                )
                for n in populations
            ]
        )
    x = np.arange(len(populations))
    width = 0.24
    colors = (PROTOCOL, "#F58518", "#54A24B")
    fig, ax = plt.subplots(figsize=(6.9, 3.25))
    for index, ((_, label), values, color) in enumerate(
        zip(metrics, values_by_metric, colors, strict=True)
    ):
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%+.1f%%", padding=3, fontsize=8)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [str(n) for n in populations])
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Protocol change from baseline")
    ax.set_ylim(-18, 80)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    finish_figure(
        fig,
        path,
        "Protocol Uplift Across Scale",
        "Relative changes in absolute return, agent task completions, and unique task types",
    )


def plot_efficiency(
    matched: dict[tuple[str, int], dict[str, str]], path: Path
) -> None:
    populations = (8, 16, 32)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))
    panels = (
        ("total_team_return", "Return per agent"),
        ("summed_total", "Task completions per agent"),
    )
    for ax, (field, ylabel) in zip(axes, panels, strict=True):
        for condition, label, color, marker, linestyle in (
            ("baseline", "Baseline", BASELINE, "o", "--"),
            ("team_formation", "Team-formation protocol", PROTOCOL, "s", "-"),
        ):
            values = [
                float(matched[(condition, n)][field]) / n for n in populations
            ]
            ax.plot(
                populations,
                values,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
            )
            for x, y in zip(populations, values, strict=True):
                ax.annotate(
                    f"{y:.2f}",
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.set_xticks(populations)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.82, wspace=0.32)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": "Per-Agent Efficiency Across Scale",
            "Subject": "Absolute output divided by physical agent count",
            "Creator": "AlemDICE",
        },
    )
    plt.close(fig)


def plot_composition(
    matched: dict[tuple[str, int], dict[str, str]], path: Path
) -> None:
    labels: list[str] = []
    base_values: list[float] = []
    coord_values: list[float] = []
    for agents in (8, 16, 32):
        for condition, short in (("baseline", "B"), ("team_formation", "P")):
            row = matched[(condition, agents)]
            labels.append(f"{short}{agents}")
            base_values.append(float(row["summed_base"]))
            coord_values.append(float(row["summed_coord"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 3.35))
    base_bars = ax.bar(
        x, base_values, color=BASE, label="Individual task completions"
    )
    coord_bars = ax.bar(
        x,
        coord_values,
        bottom=base_values,
        color=COORD,
        label="Coordinated task completions",
    )
    totals = [base + coord for base, coord in zip(base_values, coord_values, strict=True)]
    for xpos, total in zip(x, totals, strict=True):
        ax.annotate(
            f"{int(total)}",
            (xpos, total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Condition and agent count (B = baseline, P = protocol)")
    ax.set_ylabel("Task completions (sum across agents)")
    ax.set_ylim(0, 185)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    # Keep references alive for backends that inspect artists lazily.
    _ = (base_bars, coord_bars)
    finish_figure(
        fig,
        path,
        "Achievement Composition Across Scale",
        "Individual and coordinated first task completions summed across agents",
    )


def plot_marginal_yield(
    matched: dict[tuple[str, int], dict[str, str]], path: Path
) -> None:
    intervals = ((8, 16), (16, 32))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))
    panels = (
        ("total_team_return", "Added return per added agent"),
        ("summed_total", "Added task completions per added agent"),
    )
    x = np.arange(len(intervals))
    width = 0.34
    for ax, (field, ylabel) in zip(axes, panels, strict=True):
        values: dict[str, list[float]] = {}
        for condition in ("baseline", "team_formation"):
            values[condition] = [
                (
                    float(matched[(condition, end)][field])
                    - float(matched[(condition, start)][field])
                )
                / (end - start)
                for start, end in intervals
            ]
        baseline_bars = ax.bar(
            x - width / 2,
            values["baseline"],
            width,
            color=BASELINE,
            label="Baseline",
        )
        protocol_bars = ax.bar(
            x + width / 2,
            values["team_formation"],
            width,
            color=PROTOCOL,
            label="Team-formation protocol",
        )
        ax.bar_label(baseline_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(protocol_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(x, [f"{start}→{end}" for start, end in intervals])
        ax.set_xlabel("Scaling interval")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.82, wspace=0.34)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": "Marginal Yield of Added Agents",
            "Subject": "Additional absolute output per additional physical agent",
            "Creator": "AlemDICE",
        },
    )
    plt.close(fig)


def write_low_table(path: Path, low: list[dict[str, float | int]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$N$ & Team return & Return/agent & Agent unlocks & Unique types & Return CV \\",
        r"\midrule",
    ]
    for row in low:
        lines.append(
            f"{row['num_agents']} & "
            f"{row['total_return_mean']:.1f} [{row['total_return_min']:.1f}, {row['total_return_max']:.1f}] & "
            f"{row['return_per_agent']:.2f} & "
            f"{row['unlocks_mean']:.1f} [{row['unlocks_min']:.0f}, {row['unlocks_max']:.0f}] & "
            f"{row['unique_mean']:.1f} & "
            f"{100 * float(row['total_return_cv']):.1f}\\% \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Earlier baseline screen. Team return and agent unlocks are means with observed three-seed ranges in brackets. CV is the sample coefficient of variation of absolute team return. Each cell requested 200 steps; one three-agent episode terminated naturally at step 191.}",
            r"\label{tab:low-scale}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matched_table(
    path: Path, matched: dict[tuple[str, int], dict[str, str]]
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$N$ & Return B & Return P & $\Delta$ return & Unlocks B & Unlocks P & $\Delta$ unlocks & Unique B/P \\",
        r"\midrule",
    ]
    for agents in (8, 16, 32):
        baseline = matched[("baseline", agents)]
        protocol = matched[("team_formation", agents)]
        return_b = float(baseline["total_team_return"])
        return_p = float(protocol["total_team_return"])
        unlock_b = float(baseline["summed_total"])
        unlock_p = float(protocol["summed_total"])
        lines.append(
            f"{agents} & {return_b:.1f} & {return_p:.1f} & "
            f"{pct_change(return_p, return_b):+.1f}\\% & "
            f"{unlock_b:.0f} & {unlock_p:.0f} & "
            f"{pct_change(unlock_p, unlock_b):+.1f}\\% & "
            f"{baseline['team_unique_total']}/{protocol['team_unique_total']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Matched absolute performance. B denotes baseline and P denotes the team-formation protocol. Each value is one 200-step episode at the same seed; deltas are descriptive, not confidence intervals.}",
            r"\label{tab:matched-absolute}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_composition_table(
    path: Path, matched: dict[tuple[str, int], dict[str, str]]
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$N$ & Base B & Coord B & Share B & Base P & Coord P & Share P & Coord count ratio \\",
        r"\midrule",
    ]
    for agents in (8, 16, 32):
        baseline = matched[("baseline", agents)]
        protocol = matched[("team_formation", agents)]
        base_b = float(baseline["summed_base"])
        coord_b = float(baseline["summed_coord"])
        base_p = float(protocol["summed_base"])
        coord_p = float(protocol["summed_coord"])
        lines.append(
            f"{agents} & {base_b:.0f} & {coord_b:.0f} & "
            f"{100 * coord_b / (base_b + coord_b):.1f}\\% & "
            f"{base_p:.0f} & {coord_p:.0f} & "
            f"{100 * coord_p / (base_p + coord_p):.1f}\\% & "
            f"{coord_p / coord_b:.2f}$\\times$ \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Composition of summed agent first-unlocks. ``Base'' denotes individual-task achievements and ``Coord'' coordinated achievements.}",
            r"\label{tab:composition}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scaling_table(
    path: Path, matched: dict[tuple[str, int], dict[str, str]]
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Condition & Interval & $\alpha_R$ & $\alpha_A$ & Marginal return & Marginal unlocks \\",
        r"\midrule",
    ]
    for condition, label in (("baseline", "Baseline"), ("team_formation", "Protocol")):
        for start, end in ((8, 16), (16, 32)):
            r0 = matched[(condition, start)]
            r1 = matched[(condition, end)]
            return0 = float(r0["total_team_return"])
            return1 = float(r1["total_team_return"])
            unlock0 = float(r0["summed_total"])
            unlock1 = float(r1["summed_total"])
            lines.append(
                f"{label} & {start}$\\rightarrow${end} & "
                f"{elasticity(return0, return1, start, end):.3f} & "
                f"{elasticity(unlock0, unlock1, start, end):.3f} & "
                f"{(return1 - return0) / (end - start):.2f} & "
                f"{(unlock1 - unlock0) / (end - start):.2f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Local scaling diagnostics. Elasticity $\alpha=\log(Y_1/Y_0)/\log(N_1/N_0)$; $\alpha=1$ is linear absolute scaling. Marginal columns report added output per added agent.}",
            r"\label{tab:scaling-diagnostics}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    low_rows = read_csv(args.low_episodes)
    matched_rows = read_csv(args.matched)
    validate(low_rows, matched_rows)
    low = summarize_low(low_rows)
    matched = matched_index(matched_rows)

    generated = args.output_dir / "generated"
    figures = args.output_dir / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    write_derived_csv(generated / "derived_scaling_metrics.csv", low, matched)
    write_low_table(generated / "low_scale_table.tex", low)
    write_matched_table(generated / "matched_absolute_table.tex", matched)
    write_composition_table(generated / "composition_table.tex", matched)
    write_scaling_table(generated / "scaling_diagnostics_table.tex", matched)
    plot_uplift(matched, figures / "protocol_uplift.pdf")
    plot_efficiency(matched, figures / "per_agent_efficiency.pdf")
    plot_composition(matched, figures / "achievement_composition.pdf")
    plot_marginal_yield(matched, figures / "marginal_yield.pdf")
    print(args.output_dir)


if __name__ == "__main__":
    main()
