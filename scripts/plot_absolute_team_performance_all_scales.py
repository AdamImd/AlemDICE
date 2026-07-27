#!/usr/bin/env python3
"""Plot absolute team performance from one through 32 agents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE_COLOR = "#4C78A8"
PROTOCOL_COLOR = "#E45756"
AGENT_COUNTS = (1, 2, 3, 4, 6, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "Results/e3b1_scaling_matched/absolute_team_performance_all_scales.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Figures/20260721_absolute_team_performance_all_scales.pdf"
        ),
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        *(("baseline", agents) for agents in AGENT_COUNTS),
        *(("team_formation", agents) for agents in (8, 16, 32)),
    }
    observed = {(row["condition"], int(row["num_agents"])) for row in rows}
    if observed != expected:
        raise ValueError(f"unexpected condition/population cells: {sorted(observed)}")
    if any(int(row["requested_steps"]) != 200 for row in rows):
        raise ValueError("all cells must use the 200-step design")
    return rows


def select(
    rows: list[dict[str, str]], condition: str, populations: tuple[int, ...]
) -> list[dict[str, str]]:
    indexed = {
        int(row["num_agents"]): row
        for row in rows
        if row["condition"] == condition
    }
    return [indexed[population] for population in populations]


def values(rows: list[dict[str, str]], metric: str) -> tuple[list[int], list[float]]:
    return (
        [int(row["num_agents"]) for row in rows],
        [float(row[metric]) for row in rows],
    )


def plot_segment(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
    ranges: bool,
) -> None:
    xs, ys = values(rows, metric)
    if ranges:
        lower = [y - float(row[f"{metric}_min"]) for y, row in zip(ys, rows, strict=True)]
        upper = [float(row[f"{metric}_max"]) - y for y, row in zip(ys, rows, strict=True)]
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5.5,
            elinewidth=1,
            capsize=2.5,
            alpha=0.95,
            label=label,
        )
    else:
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            label=label,
        )


def annotate_high_scale(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    *,
    integer: bool,
    first_offset: int = 7,
) -> None:
    xs, ys = values(rows, metric)
    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        offset = first_offset if index == 0 else 7
        label = f"{int(round(y))}" if integer else f"{y:.1f}"
        ax.annotate(
            label,
            (x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset >= 0 else "top",
            fontsize=7.5,
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    baseline_low = select(rows, "baseline", (1, 2, 3, 4, 6))
    baseline_high = select(rows, "baseline", (8, 16, 32))
    protocol_high = select(rows, "team_formation", (8, 16, 32))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    panels = (
        ("total_team_return", "Total team return\n(sum across agents)", False),
        ("summed_total", "Task completions\n(sum across agents)", True),
    )

    for ax, (metric, ylabel, integer) in zip(axes, panels, strict=True):
        plot_segment(
            ax,
            baseline_low,
            metric,
            label="Baseline",
            color=BASELINE_COLOR,
            marker="o",
            linestyle="--",
            ranges=True,
        )
        plot_segment(
            ax,
            baseline_high,
            metric,
            label="_nolegend_",
            color=BASELINE_COLOR,
            marker="o",
            linestyle="--",
            ranges=False,
        )
        plot_segment(
            ax,
            protocol_high,
            metric,
            label="Team-formation protocol",
            color=PROTOCOL_COLOR,
            marker="s",
            linestyle="-",
            ranges=False,
        )
        annotate_high_scale(ax, baseline_high, metric, integer=integer)
        protocol_offset = -7 if metric == "summed_total" else 7
        annotate_high_scale(
            ax,
            protocol_high,
            metric,
            integer=integer,
            first_offset=protocol_offset,
        )
        ax.axvline(7, color="#B5B5B5", linewidth=0.8, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(AGENT_COUNTS, labels=[str(value) for value in AGENT_COUNTS])
        ax.set_xlim(0.85, 38)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    ordered = sorted(
        zip(handles, labels, strict=True),
        key=lambda item: 0 if item[1] == "Baseline" else 1,
    )
    fig.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.1)
    fig.savefig(
        args.output,
        bbox_inches="tight",
        metadata={
            "Title": "Absolute Team Performance from One to Thirty-Two Agents",
            "Subject": "Unnormalized population-level performance across scale",
            "Creator": "AlemDICE",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
