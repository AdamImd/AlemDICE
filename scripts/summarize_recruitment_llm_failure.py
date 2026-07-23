#!/usr/bin/env python3
"""Summarize a failed E2b campaign without mutating or promoting its artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA = "alem-dice-e2b-failed-diagnostic-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _cell_tuple(value: dict[str, Any]) -> tuple[int, str, str]:
    return int(value["seed"]), str(value["family"]), str(value["method"])


def _cell_id(cell: tuple[int, str, str]) -> str:
    seed, family, method = cell
    return f"{method}__{family}__seed_{seed}"


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: str(candidate.relative_to(root)),
    ):
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_debug(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    compressed = path.read_bytes()
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            if not line.endswith("\n"):
                raise ValueError(f"partial debug record in {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object debug record in {path}")
            rows.append(value)
    return rows, compressed


def _budget_abstentions(episode: dict[str, Any]) -> int:
    return sum(
        isinstance(decision.get("abstain_code"), str)
        and decision["abstain_code"].startswith("budget.")
        for round_record in episode.get("rounds", ())
        for decision in round_record.get("decisions", ())
    )


def _summarize_artifact_cell(
    *,
    root: Path,
    cell: tuple[int, str, str],
    classification_hint: str,
    manifest_marker: dict[str, Any] | None,
    max_output_tokens: int,
) -> dict[str, Any]:
    stem = _cell_id(cell)
    episode_path = root / "episodes" / f"{stem}.json"
    debug_path = root / "debug" / f"{stem}.calls.jsonl.gz"
    marker_path = root / "markers" / f"{stem}.complete.json"
    paths_present = {
        "episode": episode_path.is_file(),
        "debug": debug_path.is_file(),
        "marker": marker_path.is_file(),
    }
    result: dict[str, Any] = {
        "seed": cell[0],
        "family": cell[1],
        "method": cell[2],
        "classification": classification_hint,
        "paths_present": paths_present,
        "logical_calls": 0,
        "budget_abstentions": 0,
        "budget_exhausted_decisions": None,
        "validation_codes": {},
        "provider_statuses": {},
        "max_output_truncations": 0,
        "exact_cap_reasoning_only_truncations": 0,
        "semantic_sender_missing": 0,
        "analysis_only": None,
        "integrity_checks": {},
    }
    if not any(paths_present.values()):
        return result

    episode = _read_json(episode_path) if paths_present["episode"] else None
    rows, compressed = _read_debug(debug_path) if paths_present["debug"] else ([], b"")
    disk_marker = _read_json(marker_path) if paths_present["marker"] else None
    code_counts = Counter(
        str(row.get("normalized_parse", {}).get("validation_code")) for row in rows
    )
    status_counts = Counter(
        (
            str(row.get("response", {}).get("status")),
            str(row.get("response", {}).get("incomplete_reason")),
        )
        for row in rows
    )
    truncations = [
        row
        for row in rows
        if row.get("response", {}).get("status") == "incomplete"
        and row.get("response", {}).get("incomplete_reason") == "max_output_tokens"
    ]
    exact_cap = [
        row
        for row in truncations
        if row.get("response", {}).get("usage", {}).get("output_tokens")
        == max_output_tokens
        and row.get("response", {}).get("usage", {}).get("reasoning_tokens")
        == max_output_tokens
        and row.get("raw_completion") == ""
    ]
    result.update(
        {
            "logical_calls": len(rows),
            "budget_abstentions": (
                0 if episode is None else _budget_abstentions(episode)
            ),
            "budget_exhausted_decisions": (
                None
                if disk_marker is None
                else disk_marker.get("budget_exhausted_decisions")
            ),
            "validation_codes": dict(sorted(code_counts.items())),
            "provider_statuses": {
                f"{status}/{reason}": count
                for (status, reason), count in sorted(status_counts.items())
            },
            "max_output_truncations": len(truncations),
            "exact_cap_reasoning_only_truncations": len(exact_cap),
            "semantic_sender_missing": code_counts["semantic.sender_missing"],
            "analysis_only": (
                None if episode is None else episode.get("analysis_only")
            ),
        }
    )

    integrity: dict[str, bool] = {
        "artifact_triple_present": all(paths_present.values()),
    }
    if manifest_marker is not None and disk_marker is not None:
        decompressed = b"".join(
            (_canonical(row) + "\n").encode("ascii") for row in rows
        )
        integrity.update(
            {
                "manifest_marker_matches_disk": (
                    _canonical(manifest_marker) == _canonical(disk_marker)
                ),
                "artifact_sha256_matches": (
                    episode is not None
                    and manifest_marker.get("artifact_sha256")
                    == _sha256_file(episode_path)
                ),
                "debug_gzip_sha256_matches": (
                    manifest_marker.get("debug_gzip_sha256")
                    == _sha256_bytes(compressed)
                ),
                "debug_content_sha256_matches": (
                    manifest_marker.get("debug_content_sha256")
                    == _sha256_bytes(decompressed)
                ),
                "debug_record_count_matches": (
                    manifest_marker.get("debug_record_count") == len(rows)
                ),
                "logical_call_count_matches": (
                    manifest_marker.get("logical_calls") == len(rows)
                ),
                "directory_replay_matches": (
                    episode is not None and episode.get("replay_hash_match") is True
                ),
            }
        )
    result["integrity_checks"] = integrity

    if classification_hint == "manifest_completed":
        exhausted = int(result["budget_exhausted_decisions"] or 0)
        if len(rows) == 0:
            result["classification"] = "excluded_zero_call_poisoned"
        elif exhausted or result["budget_abstentions"]:
            result["classification"] = "excluded_poison_affected_completed"
        elif integrity and all(integrity.values()):
            result["classification"] = "eligible_partial_diagnostic"
        else:
            result["classification"] = "excluded_invalid_completed_artifact"
    return result


def summarize_failed_campaign(root: Path) -> dict[str, Any]:
    """Build an audit summary; the failed campaign remains non-promotable."""

    root = Path(os.path.abspath(root))
    full_path = root / "run_manifest_full.json"
    canary_path = root / "run_manifest_canary.json"
    gate_path = root / "canary_gate.json"
    ledger_path = root / "reservation_ledger.jsonl"
    full = _read_json(full_path)
    if full.get("status") != "failed":
        raise ValueError("the diagnostic summarizer requires a failed full manifest")
    protocol = full.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("full manifest has no bound protocol")
    max_output_tokens = int(protocol["max_output_tokens"])
    matrix = {
        (int(seed), str(family), str(method))
        for seed in protocol["seeds"]
        for family in protocol["scenario_families"]
        for method in protocol["methods"]
    }
    markers = {
        _cell_tuple(marker): marker for marker in full.get("markers", ())
    }
    failed = {
        _cell_tuple(cell) for cell in full.get("failed_episodes", ())
    }
    cancelled = {
        _cell_tuple(cell) for cell in full.get("cancelled_cells", ())
    }
    if (
        set(markers) & failed
        or set(markers) & cancelled
        or failed & cancelled
        or set(markers) | failed | cancelled != matrix
    ):
        raise ValueError("full manifest cell classifications do not partition the matrix")

    cells = []
    for cell in sorted(matrix):
        if cell in markers:
            hint = "manifest_completed"
        elif cell in failed:
            hint = "failed_diagnostic_only"
        else:
            hint = "cancelled_no_evidence"
        cells.append(
            _summarize_artifact_cell(
                root=root,
                cell=cell,
                classification_hint=hint,
                manifest_marker=markers.get(cell),
                max_output_tokens=max_output_tokens,
            )
        )

    classifications = Counter(cell["classification"] for cell in cells)
    code_counts = Counter()
    provider_statuses = Counter()
    for cell in cells:
        code_counts.update(cell["validation_codes"])
        provider_statuses.update(cell["provider_statuses"])
    eligible = [
        cell for cell in cells if cell["classification"] == "eligible_partial_diagnostic"
    ]
    poison_affected = [
        cell
        for cell in cells
        if cell["classification"] == "excluded_poison_affected_completed"
    ]
    semantic_examples = []
    for cell in cells:
        if not cell["semantic_sender_missing"]:
            continue
        debug_path = (
            root
            / "debug"
            / f"{_cell_id((cell['seed'], cell['family'], cell['method']))}.calls.jsonl.gz"
        )
        rows, _ = _read_debug(debug_path)
        for row in rows:
            if (
                row.get("normalized_parse", {}).get("validation_code")
                == "semantic.sender_missing"
            ):
                semantic_examples.append(
                    {
                        "seed": cell["seed"],
                        "family": cell["family"],
                        "method": cell["method"],
                        "round_index": row["round_index"],
                        "agent_id": row["agent_id"],
                        "attempt": row["attempt"],
                        "raw_completion": row["raw_completion"],
                    }
                )

    ledger_sha256 = _sha256_file(ledger_path)
    ledger_final = full["reservation_ledger_final"]
    ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    anchor_paths = sorted((root / "reservation_anchors").glob("*.json"))
    binding_checks = {
        "ledger_sha256_matches_manifest": (
            ledger_sha256 == ledger_final["ledger_sha256"]
        ),
        "ledger_record_count_matches_manifest": (
            len(ledger_lines) == int(ledger_final["records"])
        ),
        "anchor_count_matches_manifest": (
            len(anchor_paths) == int(ledger_final["anchor_count"])
        ),
        "no_unresolved_reservations": not ledger_final["unresolved"],
        "no_usage_overages": not ledger_final["overages"],
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "campaign_status": "failed_non_promotable",
        "interpretation": (
            "partial failed-run diagnostic only; excluded cells are never imputed "
            "and no between-method efficacy estimate is valid"
        ),
        "root_binding": {
            "path": str(root),
            "tree_sha256": _tree_sha256(root),
            "full_manifest_sha256": _sha256_file(full_path),
            "canary_manifest_sha256": (
                _sha256_file(canary_path) if canary_path.is_file() else None
            ),
            "canary_gate_sha256": (
                _sha256_file(gate_path) if gate_path.is_file() else None
            ),
            "ledger_sha256": ledger_sha256,
            "source_commit": full.get("source_commit"),
            "config_sha256": full.get("config_sha256"),
            "binding_checks": binding_checks,
        },
        "manifest_accounting": {
            "expected_cells": len(matrix),
            "manifest_completed_cells": len(markers),
            "failed_cells": len(failed),
            "cancelled_cells": len(cancelled),
            "logical_calls": full["campaign_budget"]["logical_used"],
            "provider_attempts_reserved": full["campaign_budget"][
                "provider_attempts_reserved"
            ],
            "tokens_reserved": full["campaign_budget"]["tokens_reserved"],
            "poisoned": full["campaign_budget"]["poisoned"],
            "poisoned_reason": full["campaign_budget"]["poisoned_reason"],
        },
        "classification_counts": dict(sorted(classifications.items())),
        "diagnostic_totals": {
            "debug_calls": sum(cell["logical_calls"] for cell in cells),
            "validation_codes": dict(sorted(code_counts.items())),
            "provider_statuses": dict(sorted(provider_statuses.items())),
            "max_output_truncations": sum(
                cell["max_output_truncations"] for cell in cells
            ),
            "exact_cap_reasoning_only_truncations": sum(
                cell["exact_cap_reasoning_only_truncations"] for cell in cells
            ),
            "semantic_sender_missing": sum(
                cell["semantic_sender_missing"] for cell in cells
            ),
            "configured_max_output_tokens": max_output_tokens,
        },
        "eligible_partial_outcomes": [
            {
                "seed": cell["seed"],
                "family": cell["family"],
                "method": cell["method"],
                "logical_calls": cell["logical_calls"],
                "analysis_only": cell["analysis_only"],
            }
            for cell in eligible
        ],
        "excluded_poison_affected_observations": [
            {
                "seed": cell["seed"],
                "family": cell["family"],
                "method": cell["method"],
                "logical_calls": cell["logical_calls"],
                "budget_abstentions": cell["budget_abstentions"],
                "analysis_only": cell["analysis_only"],
            }
            for cell in poison_affected
        ],
        "semantic_sender_missing_examples": semantic_examples,
        "cells": cells,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    accounting = summary["manifest_accounting"]
    totals = summary["diagnostic_totals"]
    lines = [
        "# E2b v2 failed hosted diagnostic",
        "",
        "> **Status: failed and permanently non-promotable.** This is partial",
        "> diagnostic evidence, not a between-method efficacy result. Cancelled,",
        "> failed, zero-call, and poison-affected cells are not imputed.",
        "",
        "## Bound source",
        "",
        f"- Preserved root: `{summary['root_binding']['path']}`",
        f"- Tree SHA-256: `{summary['root_binding']['tree_sha256']}`",
        f"- Full manifest SHA-256: `{summary['root_binding']['full_manifest_sha256']}`",
        f"- Ledger SHA-256: `{summary['root_binding']['ledger_sha256']}`",
        f"- Source commit: `{summary['root_binding']['source_commit']}`",
        "",
        "## Manifest accounting",
        "",
        f"- Matrix: {accounting['expected_cells']} cells.",
        f"- Manifest-completed: {accounting['manifest_completed_cells']}; failed: "
        f"{accounting['failed_cells']}; cancelled: {accounting['cancelled_cells']}.",
        f"- Budget: {accounting['logical_calls']} logical calls, "
        f"{accounting['provider_attempts_reserved']} reserved provider attempts, "
        f"{accounting['tokens_reserved']:,} reserved tokens.",
        f"- Poisoned: `{accounting['poisoned']}` "
        f"(`{accounting['poisoned_reason']}`).",
        "",
        "## Failure mechanism",
        "",
        f"- {totals['max_output_truncations']}/{totals['debug_calls']} calls returned "
        "`incomplete/max_output_tokens`.",
        f"- All {totals['exact_cap_reasoning_only_truncations']} truncations used "
        f"exactly {totals['configured_max_output_tokens']} output tokens entirely "
        "as reasoning and returned a blank visible completion.",
        f"- `semantic.sender_missing`: {totals['semantic_sender_missing']} calls.",
        f"- Validation codes: `{json.dumps(totals['validation_codes'], sort_keys=True)}`.",
        "",
        "## Evidence classification",
        "",
    ]
    for name, count in summary["classification_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "Only `eligible_partial_diagnostic` cells are shown below. They remain",
            "descriptive because the campaign failed and the method matrix is incomplete.",
            "",
            "| Seed | Family | Method | Calls | Coverage | Reward |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for cell in summary["eligible_partial_outcomes"]:
        analysis = cell["analysis_only"]
        lines.append(
            f"| {cell['seed']} | {cell['family']} | {cell['method']} | "
            f"{cell['logical_calls']} | "
            f"{analysis['oracle_allocation_coverage']:.3f} | "
            f"{analysis['achieved_reward']} |"
        )
    if summary["excluded_poison_affected_observations"]:
        lines.extend(
            [
                "",
                "A poison-affected completed artifact is retained only as an excluded",
                "observation. It is not included in the table or any aggregate:",
                "",
            ]
        )
        for cell in summary["excluded_poison_affected_observations"]:
            analysis = cell["analysis_only"]
            lines.append(
                f"- `{cell['method']}/{cell['family']}/seed-{cell['seed']}`: "
                f"{cell['logical_calls']} calls before/around poison, "
                f"{cell['budget_abstentions']} poison-budget abstentions, "
                f"coverage {analysis['oracle_allocation_coverage']:.3f}, "
                f"reward {analysis['achieved_reward']} (excluded)."
            )
    lines.extend(
        [
            "",
            "## Consequence for the next protocol",
            "",
            "The one-cell canary tested only Open Volunteer and therefore could not",
            "detect the Mutual Nomination failure mode. The successor must use a fresh",
            "root, increase the high-reasoning output allowance prospectively, and",
            "require every Open/Mutual × single/two-disjoint canary cell to pass before",
            "the remaining matrix can dispatch.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_outside_root(root: Path, path: Path, text: str) -> None:
    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(path))
    if os.path.commonpath((root, path)) == str(root):
        raise ValueError("diagnostic output must be outside the preserved campaign root")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    summary = summarize_failed_campaign(args.root)
    if args.json_output is None and args.markdown_output is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.json_output is not None:
        _write_text_outside_root(
            args.root,
            args.json_output,
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
    if args.markdown_output is not None:
        _write_text_outside_root(
            args.root,
            args.markdown_output,
            render_markdown(summary),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
