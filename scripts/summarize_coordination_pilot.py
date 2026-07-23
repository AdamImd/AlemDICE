#!/usr/bin/env python3
"""Create compact JSON/CSV/Markdown summaries from a DCP1 pilot directory."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ARMS = ("free", "consensus", "roles", "cohesion", "integrated")
FIELDS = (
    "arm", "valid", "error", "num_steps", "episode_return", "total_achievements",
    "action_parse_rate", "protocol_parse_rate", "agreements",
    "proposal_agreement_rate", "valid_awards", "accepted_awards",
    "status_window_coverage", "input_tokens", "model_call_count",
    "output_tokens", "episode_wall_seconds", "coordination_attempts",
    "coordination_successes",
)


def row_for(root: Path, arm: str) -> dict:
    path = root / arm / "alem" / "default" / "default_run_00.json"
    if not path.exists():
        return {"arm": arm, "valid": False, "error": "missing episode artifact"}
    episode = json.loads(path.read_text(encoding="utf-8"))
    protocol = episode.get("coordination_protocol", {})
    stats = episode.get("user_info", {}) or {}
    achievements = stats.get(
        "Team/total_achievements",
        stats.get("team_achievements", stats.get("total_achievements", 0)),
    )
    return {
        "arm": arm,
        "valid": not bool(episode.get("error")),
        "error": episode.get("error"),
        "num_steps": episode.get("num_steps"),
        "episode_return": episode.get("episode_return"),
        "total_achievements": achievements,
        "action_parse_rate": episode.get("action_parse_rate"),
        "protocol_parse_rate": protocol.get("protocol_parse_rate"),
        "agreements": protocol.get("agreements"),
        "proposal_agreement_rate": protocol.get("proposal_agreement_rate"),
        "valid_awards": protocol.get("valid_awards"),
        "accepted_awards": protocol.get("accepted_awards"),
        "status_window_coverage": protocol.get("status_window_coverage"),
        "input_tokens": episode.get("input_tokens"),
        "output_tokens": episode.get("output_tokens"),
        "model_call_count": episode.get("model_call_count"),
        "episode_wall_seconds": episode.get("episode_wall_seconds"),
        "coordination_attempts": stats.get("Coordination/total_attempts"),
        "coordination_successes": stats.get("Coordination/total_successes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--copy-to", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    rows = [row_for(args.root, arm) for arm in args.arms]
    destination = args.copy_to or args.root
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pilot_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (destination / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)
    columns = ("arm", "valid", "num_steps", "episode_return", "action_parse_rate", "protocol_parse_rate", "agreements", "valid_awards", "accepted_awards", "status_window_coverage", "input_tokens")
    lines = ["# DCP1 local pilot results", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines += ["", "This is a one-seed mechanism pilot; differences are not efficacy estimates."]
    (destination / "pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ("manifest.json", "events.jsonl"):
        source = args.root / name
        if source.exists() and source.resolve() != (destination / name).resolve():
            shutil.copy2(source, destination / name)


if __name__ == "__main__":
    main()
