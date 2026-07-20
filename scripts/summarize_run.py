#!/usr/bin/env python3
"""Rebuild human-readable summaries for one Alem run or a matrix root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.llm.utils import (  # noqa: E402
    collect_and_summarize_results,
    print_summary_table,
    save_summary_stats,
)


def _run_directories(root: Path) -> list[tuple[str, Path]]:
    manifest = root / "matrix_manifest.json"
    if not manifest.is_file():
        return [(root.name, root)]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [
        (str(difficulty), root / str(difficulty))
        for difficulty in payload.get("difficulties", [])
        if (root / str(difficulty)).is_dir()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"run directory does not exist: {root}")

    matrix_summary: dict[str, object] = {}
    for label, run_dir in _run_directories(root):
        summary = collect_and_summarize_results(str(run_dir))
        if not summary:
            print(f"{label}: no completed episode results in {run_dir}")
            continue
        print(f"\n### {label.upper()} ###")
        print_summary_table(summary)
        save_summary_stats(summary, str(run_dir))
        summary_path = run_dir / "summary_stats.json"
        matrix_summary[label] = json.loads(summary_path.read_text(encoding="utf-8"))

    if not matrix_summary:
        return 1
    if (root / "matrix_manifest.json").is_file():
        destination = root / "matrix_summary.json"
        destination.write_text(
            json.dumps(matrix_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Matrix summary: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
