#!/usr/bin/env python3
"""Plot absolute, population-level performance for matched scaling runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001


CONDITIONS = {
    "baseline": {
        "label": "Baseline",
        "color": "#4C78A8",
        "marker": "o",
        "linestyle": "--",
    },
    "team_formation": {
        "label": "Team-formation protocol",
        "color": "#E45756",
        "marker": "s",
        "linestyle": "-",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Results/e3b1_scaling_matched/absolute_team_performance.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Figures/20260721_absolute_team_performance.pdf"),
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(condition, agents) for condition in CONDITIONS for agents in (8, 16, 32)}
    observed = {(row["condition"], int(row["num_agents"])) for row in rows}
    if observed != expected:
        raise ValueError(f"expected matched 8/16/32 cells, got {sorted(observed)}")
    if any(int(row["steps"]) != 200 or int(row["seed"]) != 14100 for row in rows):
        raise ValueError("all plotted cells must use the matched 200-step, seed-14100 design")
    return rows


def series(rows: list[dict[str, str]], condition: str, metric: str) -> tuple[list[int], list[float]]:
    selected = sorted(
        (row for row in rows if row["condition"] == condition),
        key=lambda row: int(row["num_agents"]),
    )
    return [int(row["num_agents"]) for row in selected], [float(row[metric]) for row in selected]


def annotate(
    ax: plt.Axes,
    xs: list[int],
    ys: list[float],
    *,
    integer: bool,
    first_offset: int = 7,
) -> None:
    for x, y in zip(xs, ys, strict=True):
        label = f"{int(round(y))}" if integer else f"{y:.1f}"
        offset = first_offset if x == xs[0] else 7
        ax.annotate(
            label,
            (x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset >= 0 else "top",
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    panels = (
        ("total_team_return", "Total team return\n(sum across agents)", False),
        ("summed_total", "Task completions\n(sum across agents)", True),
    )

    for ax, (metric, ylabel, integer) in zip(axes, panels, strict=True):
        for condition, style in CONDITIONS.items():
            xs, ys = series(rows, condition, metric)
            ax.plot(
                xs,
                ys,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=6,
            )
            first_offset = -7 if metric == "summed_total" and condition == "team_formation" else 7
            annotate(ax, xs, ys, integer=integer, first_offset=first_offset)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(ylabel)
        ax.set_xticks([8, 16, 32])
        ax.set_xlim(6, 34)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.2)
    fig.savefig(
        args.output,
        bbox_inches="tight",
        metadata={
            "Title": "Absolute Team Performance Across Population Scale",
            "Subject": "Matched scaling comparison using unnormalized population-level measures",
            "Creator": "AlemDICE",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
