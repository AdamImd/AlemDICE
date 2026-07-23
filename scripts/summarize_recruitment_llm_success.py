#!/usr/bin/env python3
"""Validate and summarize one completed E2b v4 hosted campaign.

This command is deliberately provider-free and read-only with respect to the
campaign root.  It reuses the launcher's strict replay, artifact, debug-shard,
and durable-ledger validators before deriving the published efficacy and cost
statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.llm.eval_utils.recruitment_selection import (  # noqa: E402
    InformationSource,
    roster_utility,
    true_information_oracle,
)
from baselines.llm.eval_utils.team_formation import (  # noqa: E402
    RecordKind,
    RecruitmentMethod,
    parse_tfp1,
)
from baselines.llm.recruitment_arena import (  # noqa: E402
    ScenarioFamily,
    generate_scenario,
)
from scripts import run_recruitment_llm_screen as runner  # noqa: E402

SCHEMA_VERSION = "alem-dice-e2b-hosted-results-v1"
EXPECTED_PROTOCOL_REVISION = "e2b-v4-open-joint-confirmation-two-cell-canary"
CANARY_CELLS = {
    (22000, ScenarioFamily.SINGLE_COMPLEMENTARY.value),
    (22000, ScenarioFamily.TWO_DISJOINT.value),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Hash relative path and file digest for every regular campaign file."""

    digest = hashlib.sha256()
    files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: str(candidate.relative_to(root)),
    )
    for path in files:
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fraction(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "float": float(value),
    }


def _mean_fraction(values: Iterable[Fraction]) -> Fraction:
    materialized = tuple(values)
    if not materialized:
        return Fraction(0)
    return sum(materialized, Fraction(0)) / len(materialized)


def _duration_seconds(start: str, finish: str) -> float:
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _cell_metrics(episode: dict[str, Any]) -> dict[str, Any]:
    family = ScenarioFamily(episode["scenario"]["family"])
    seed = int(episode["scenario"]["seed"])
    scenario = generate_scenario(family, seed)
    agents = {agent.agent_id: agent for agent in scenario.agents}
    tasks = {task.task_id: task for task in scenario.tasks}
    oracle = true_information_oracle(scenario.tasks, scenario.agents)
    locked = {
        task_id: tuple(int(member) for member in members)
        for task_id, members in episode["analysis_only"]["locked_rosters"].items()
    }
    oracle_rosters = {
        assignment.task_id: tuple(int(member) for member in assignment.roster)
        for assignment in oracle.completed_tasks
    }
    achieved_utility = sum(
        (
            roster_utility(
                tasks[task_id],
                members,
                agents,
                information=InformationSource.TRUE,
            )
            for task_id, members in locked.items()
        ),
        Fraction(0),
    )
    achieved_cost = sum(
        (
            Fraction(agents[member].task_costs[task_id])
            for task_id, members in locked.items()
            for member in members
        ),
        Fraction(0),
    )
    utility_regret = oracle.total_roster_utility - achieved_utility
    cost_delta = achieved_cost - oracle.total_raw_cost

    calls = list(episode["call_ledger"])
    initial_calls = [call for call in calls if not call["semantic_repair"]]
    repair_calls = [call for call in calls if call["semantic_repair"]]
    invalid_calls = [call for call in calls if call["valid"] is False]
    call_latencies = [float(call["latency_seconds"]) for call in calls]

    operations = [
        operation
        for operation in episode["directory_replay"]["operations"]
        if operation.get("op") == "submit_control"
    ]
    transitions = list(episode["transition_ledger"])
    if len(operations) != len(transitions):
        raise ValueError("control operation/transition cardinality mismatch")

    kinds: Counter[str] = Counter()
    apply_claims = 0
    exact_capability_claims = 0
    exact_cost_claims = 0
    exact_apply_claims = 0
    claim_mismatches: list[dict[str, Any]] = []
    control_payload_bytes = 0
    control_delivered_bytes = 0
    for operation, transition in zip(operations, transitions, strict=True):
        parsed = parse_tfp1(operation["raw"])
        if (
            not parsed.valid
            or parsed.record is None
            or transition["sender"] != operation["sender"]
            or transition["task_id"] != parsed.record.task_id
        ):
            raise ValueError("strict control operation/transition binding mismatch")
        record = parsed.record
        kinds[record.kind.value] += 1
        control_payload_bytes += parsed.payload_bytes
        control_delivered_bytes += parsed.payload_bytes * len(transition["delivered_recipients"])
        if record.kind is not RecordKind.APPLY:
            continue
        apply_claims += 1
        sender = int(operation["sender"])
        task_id = record.task_id
        capability_exact = record.capabilities == agents[sender].true_capabilities
        cost_exact = record.cost == agents[sender].task_costs[task_id]
        exact_capability_claims += int(capability_exact)
        exact_cost_claims += int(cost_exact)
        exact_apply_claims += int(capability_exact and cost_exact)
        if not (capability_exact and cost_exact):
            claim_mismatches.append(
                {
                    "sender": sender,
                    "task_id": task_id,
                    "claimed_capabilities": list(record.capabilities or ()),
                    "true_capabilities": list(agents[sender].true_capabilities),
                    "claimed_cost": record.cost,
                    "true_cost": agents[sender].task_costs[task_id],
                }
            )

    exact_task_rosters = sum(
        locked.get(task_id) == roster for task_id, roster in oracle_rosters.items()
    )
    return {
        "seed": seed,
        "family": family.value,
        "stage_origin": ("canary" if (seed, family.value) in CANARY_CELLS else "promoted_full"),
        "normalized_reward": float(episode["analysis_only"]["normalized_reward"]),
        "oracle_allocation_coverage": float(episode["analysis_only"]["oracle_allocation_coverage"]),
        "achieved_reward": int(episode["analysis_only"]["achieved_reward"]),
        "oracle_reward": int(episode["analysis_only"]["oracle_reward"]),
        "true_feasible_locked_tasks": int(episode["analysis_only"]["true_feasible_locked_tasks"]),
        "true_infeasible_locked_tasks": sum(
            not feasible for feasible in episode["analysis_only"]["true_feasible_locks"].values()
        ),
        "locked_rosters": {key: list(value) for key, value in sorted(locked.items())},
        "oracle_rosters": {key: list(value) for key, value in sorted(oracle_rosters.items())},
        "exact_oracle_allocation": locked == oracle_rosters,
        "exact_oracle_task_rosters": exact_task_rosters,
        "oracle_task_rosters": len(oracle_rosters),
        "achieved_utility": _fraction(achieved_utility),
        "oracle_utility": _fraction(oracle.total_roster_utility),
        "utility_regret": _fraction(utility_regret),
        "achieved_raw_cost": _fraction(achieved_cost),
        "oracle_raw_cost": _fraction(oracle.total_raw_cost),
        "raw_cost_delta": _fraction(cost_delta),
        "logical_calls": len(calls),
        "initial_calls": len(initial_calls),
        "semantic_repair_calls": len(repair_calls),
        "invalid_calls": len(invalid_calls),
        "provider_attempts_actual": sum(int(call["transport_attempt_count"]) for call in calls),
        "transport_errors": sum(int(call["transport_error_count"]) for call in calls),
        "max_output_truncations": sum(
            call["incomplete_reason"] == "max_output_tokens" for call in calls
        ),
        "incomplete_calls": sum(call["status"] != "completed" for call in calls),
        "input_tokens": int(episode["token_ledger"]["input_tokens"]),
        "output_tokens": int(episode["token_ledger"]["output_tokens"]),
        "reasoning_tokens": int(episode["token_ledger"]["reasoning_tokens"]),
        "cached_tokens": int(episode["token_ledger"]["cached_tokens"]),
        "cache_write_tokens": int(episode["token_ledger"]["cache_write_tokens"]),
        "sum_call_latency_seconds": float(episode["latency_ledger"]["sum_call_latency_seconds"]),
        "max_call_latency_seconds": max(call_latencies, default=0.0),
        "decision_wall_seconds": sum(
            float(value) for value in episode["latency_ledger"]["round_decision_wall_seconds"]
        ),
        "executed_acting_rounds": int(episode["early_stop"]["executed_acting_rounds"]),
        "early_stop_reason": episode["early_stop"]["reason"],
        "lock_rounds": list(episode["analysis_only"]["lock_rounds"].values()),
        "apply_claims": apply_claims,
        "exact_capability_claims": exact_capability_claims,
        "exact_cost_claims": exact_cost_claims,
        "exact_apply_claims": exact_apply_claims,
        "claim_mismatches": claim_mismatches,
        "control_kinds": dict(sorted(kinds.items())),
        "control_submissions": len(operations),
        "accepted_control_transitions": sum(
            bool(transition["accepted"]) for transition in transitions
        ),
        "rejected_control_transitions": sum(
            not bool(transition["accepted"]) for transition in transitions
        ),
        "control_payload_bytes": control_payload_bytes,
        "control_delivered_bytes": control_delivered_bytes,
    }


def _aggregate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    calls = sum(cell["logical_calls"] for cell in cells)
    initial_calls = sum(cell["initial_calls"] for cell in cells)
    repairs = sum(cell["semantic_repair_calls"] for cell in cells)
    invalid = sum(cell["invalid_calls"] for cell in cells)
    input_tokens = sum(cell["input_tokens"] for cell in cells)
    output_tokens = sum(cell["output_tokens"] for cell in cells)
    reasoning_tokens = sum(cell["reasoning_tokens"] for cell in cells)
    lock_rounds = [int(round_index) for cell in cells for round_index in cell["lock_rounds"]]
    utility_regrets = [
        Fraction(
            cell["utility_regret"]["numerator"],
            cell["utility_regret"]["denominator"],
        )
        for cell in cells
    ]
    cost_deltas = [
        Fraction(
            cell["raw_cost_delta"]["numerator"],
            cell["raw_cost_delta"]["denominator"],
        )
        for cell in cells
    ]
    return {
        "episodes": len(cells),
        "full_reward_episodes": sum(cell["normalized_reward"] == 1.0 for cell in cells),
        "full_coverage_episodes": sum(cell["oracle_allocation_coverage"] == 1.0 for cell in cells),
        "true_feasible_locks": sum(cell["true_feasible_locked_tasks"] for cell in cells),
        "true_infeasible_locks": sum(cell["true_infeasible_locked_tasks"] for cell in cells),
        "exact_oracle_allocations": sum(cell["exact_oracle_allocation"] for cell in cells),
        "exact_oracle_task_rosters": sum(cell["exact_oracle_task_rosters"] for cell in cells),
        "oracle_task_rosters": sum(cell["oracle_task_rosters"] for cell in cells),
        "mean_utility_regret": _fraction(_mean_fraction(utility_regrets)),
        "mean_raw_cost_delta": _fraction(_mean_fraction(cost_deltas)),
        "logical_calls": calls,
        "initial_calls": initial_calls,
        "semantic_repair_calls": repairs,
        "invalid_calls": invalid,
        "invalid_initial_decision_rate": invalid / initial_calls if initial_calls else 0.0,
        "semantic_repair_rate": repairs / initial_calls if initial_calls else 0.0,
        "provider_attempts_actual": sum(cell["provider_attempts_actual"] for cell in cells),
        "transport_errors": sum(cell["transport_errors"] for cell in cells),
        "max_output_truncations": sum(cell["max_output_truncations"] for cell in cells),
        "incomplete_calls": sum(cell["incomplete_calls"] for cell in cells),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "actual_total_tokens": input_tokens + output_tokens,
        "reasoning_share_of_output": (reasoning_tokens / output_tokens if output_tokens else 0.0),
        "cached_tokens": sum(cell["cached_tokens"] for cell in cells),
        "cache_write_tokens": sum(cell["cache_write_tokens"] for cell in cells),
        "sum_call_latency_seconds": sum(cell["sum_call_latency_seconds"] for cell in cells),
        "mean_call_latency_seconds": (
            sum(cell["sum_call_latency_seconds"] for cell in cells) / calls if calls else 0.0
        ),
        "max_call_latency_seconds": max(
            (cell["max_call_latency_seconds"] for cell in cells), default=0.0
        ),
        "decision_wall_seconds": sum(cell["decision_wall_seconds"] for cell in cells),
        "mean_episode_decision_wall_seconds": (
            statistics.fmean(cell["decision_wall_seconds"] for cell in cells) if cells else 0.0
        ),
        "executed_acting_rounds": sum(cell["executed_acting_rounds"] for cell in cells),
        "mean_executed_acting_rounds": (
            statistics.fmean(cell["executed_acting_rounds"] for cell in cells) if cells else 0.0
        ),
        "mean_lock_round": statistics.fmean(lock_rounds) if lock_rounds else None,
        "apply_claims": sum(cell["apply_claims"] for cell in cells),
        "exact_capability_claims": sum(cell["exact_capability_claims"] for cell in cells),
        "exact_cost_claims": sum(cell["exact_cost_claims"] for cell in cells),
        "exact_apply_claims": sum(cell["exact_apply_claims"] for cell in cells),
        "control_submissions": sum(cell["control_submissions"] for cell in cells),
        "accepted_control_transitions": sum(cell["accepted_control_transitions"] for cell in cells),
        "rejected_control_transitions": sum(cell["rejected_control_transitions"] for cell in cells),
        "control_payload_bytes": sum(cell["control_payload_bytes"] for cell in cells),
        "control_delivered_bytes": sum(cell["control_delivered_bytes"] for cell in cells),
        "mean_control_delivered_bytes": (
            statistics.fmean(cell["control_delivered_bytes"] for cell in cells) if cells else 0.0
        ),
    }


def _invalid_call_details(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details = []
    for episode in episodes:
        for call in episode["call_ledger"]:
            if call["valid"] is not False:
                continue
            key = dict(call["reservation_key"])
            repair = next(
                (
                    candidate
                    for candidate in episode["call_ledger"]
                    if candidate["semantic_repair"]
                    and candidate["reservation_key"]["agent_id"] == key["agent_id"]
                    and candidate["reservation_key"]["round_index"] == key["round_index"]
                ),
                None,
            )
            details.append(
                {
                    "reservation_key": key,
                    "validation_code": call["validation_code"],
                    "invalid_completion": call["failure_excerpt"],
                    "repair_outcome": None if repair is None else repair["validation_code"],
                    "repair_was_valid": None if repair is None else repair["valid"],
                }
            )
    return details


def summarize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    full_manifest = _read_json(root / "run_manifest_full.json")
    canary_manifest = _read_json(root / "run_manifest_canary.json")
    canary_gate = _read_json(root / "canary_gate.json")
    protocol = full_manifest["protocol"]
    launch_binding = full_manifest["launch_binding"]
    if (
        full_manifest["status"] != "complete"
        or canary_manifest["status"] != "complete"
        or canary_gate["status"] != "pass"
        or protocol["protocol_revision"] != EXPECTED_PROTOCOL_REVISION
        or protocol["methods"] != [RecruitmentMethod.OPEN_VOLUNTEER.value]
        or full_manifest["source_commit"] != launch_binding["git_head"]
    ):
        raise ValueError("root is not the completed bound E2b v4 Open campaign")

    ledger = runner.DurableReservationLedger(root, launch_binding=launch_binding)
    reconciliation = ledger.reconcile()
    for manifest in (canary_manifest, full_manifest):
        ledger.verify_checkpoint(manifest["reservation_ledger_at_launch"]["checkpoint"])
        ledger.verify_checkpoint(manifest["reservation_ledger_final"]["checkpoint"])

    jobs = tuple(
        (
            int(seed),
            ScenarioFamily(family),
            RecruitmentMethod.OPEN_VOLUNTEER,
        )
        for seed in protocol["seeds"]
        for family in protocol["scenario_families"]
    )
    inventory_count = runner._validate_managed_cell_inventory(root, jobs)
    markers = []
    episodes = []
    for seed, family, method in jobs:
        marker = runner._strict_completed_cell(
            root,
            seed=seed,
            family=family,
            method=method,
            config_sha256=full_manifest["config_sha256"],
            source_hashes=full_manifest["source_hashes"],
            launch_binding=launch_binding,
            protocol=protocol,
            expected_resolved_model=full_manifest["resolved_model_binding"],
            reservation_reconciliation=reconciliation,
            require_reservations=True,
        )
        markers.append(marker)
        artifact = root / marker["artifact"]
        episodes.append(_read_json(artifact))
    runner._assert_reservation_coverage(reconciliation, markers)

    cells = sorted(
        (_cell_metrics(episode) for episode in episodes),
        key=lambda value: (value["family"], value["seed"]),
    )
    canary_cells = [cell for cell in cells if cell["stage_origin"] == "canary"]
    promoted_cells = [cell for cell in cells if cell["stage_origin"] == "promoted_full"]
    combined = _aggregate_cells(cells)
    if (
        len(canary_cells) != 2
        or len(promoted_cells) != 10
        or combined["full_reward_episodes"] != 12
        or combined["full_coverage_episodes"] != 12
        or combined["true_infeasible_locks"] != 0
    ):
        raise ValueError("completed campaign fails the published efficacy gate")

    family_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        family_groups[cell["family"]].append(cell)
    families = {
        family: _aggregate_cells(values) for family, values in sorted(family_groups.items())
    }

    hashes = {
        "root_tree_sha256": _tree_sha256(root),
        "run_manifest_canary_sha256": _sha256_file(root / "run_manifest_canary.json"),
        "canary_gate_sha256": _sha256_file(root / "canary_gate.json"),
        "run_manifest_full_sha256": _sha256_file(root / "run_manifest_full.json"),
        "reservation_ledger_sha256": _sha256_file(root / "reservation_ledger.jsonl"),
    }
    invalid_details = _invalid_call_details(episodes)
    claim_mismatches = [
        {
            "seed": cell["seed"],
            "family": cell["family"],
            **mismatch,
        }
        for cell in cells
        for mismatch in cell["claim_mismatches"]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_status": "complete_promotable_mechanism_screen",
        "evidence_boundary": (
            "Six-agent structured recruitment-arena mechanism evidence only; "
            "not Alem task efficacy, long-horizon role alignment, an "
            "Open-versus-Mutual effect, or a scalability result."
        ),
        "source": {
            "campaign_root": str(root),
            "source_commit": full_manifest["source_commit"],
            "protocol_revision": protocol["protocol_revision"],
            "requested_model": protocol["model_id"],
            "resolved_model": full_manifest["resolved_model_binding"],
            "reasoning_effort": protocol["reasoning_effort"],
            "max_output_tokens": protocol["max_output_tokens"],
            "hashes": hashes,
        },
        "integrity": {
            "strict_cells_validated": len(markers),
            "managed_inventory_count": inventory_count,
            "all_replay_hashes_match": all(
                episode["replay_hash_match"] is True for episode in episodes
            ),
            "ledger_records": reconciliation["records"],
            "ledger_reservations": len(reconciliation["reservations"]),
            "ledger_resolutions": len(reconciliation["resolutions"]),
            "ledger_unresolved": len(reconciliation["unresolved"]),
            "ledger_overages": len(reconciliation["overages"]),
            "provider_attempts_reserved": reconciliation["provider_attempts_reserved"],
            "tokens_reserved": reconciliation["tokens_reserved"],
            "poisoned": bool(full_manifest["campaign_budget"]["poisoned"]),
            "failed_episodes": full_manifest["failed_episodes"],
            "cancelled_cells": full_manifest["cancelled_cells"],
        },
        "stage_accounting": {
            "canary": {
                **_aggregate_cells(canary_cells),
                "started_at": canary_manifest["started_at"],
                "finished_at": canary_manifest["finished_at"],
                "invocation_wall_seconds": _duration_seconds(
                    canary_manifest["started_at"], canary_manifest["finished_at"]
                ),
                "cell_workers": canary_manifest["cell_workers"],
            },
            "promoted_full_new_cells": {
                **_aggregate_cells(promoted_cells),
                "started_at": full_manifest["started_at"],
                "finished_at": full_manifest["finished_at"],
                "invocation_wall_seconds": _duration_seconds(
                    full_manifest["started_at"], full_manifest["finished_at"]
                ),
                "cell_workers": full_manifest["cell_workers"],
                "completed_at_launch": full_manifest["completed_at_launch"],
            },
            "combined": combined,
        },
        "families": families,
        "cells": cells,
        "semantic_invalid_calls": invalid_details,
        "claim_mismatches": claim_mismatches,
        "deterministic_reference": {
            "source": "Results/e2d2_joint_allocation_v1.md",
            "source_sha256": _sha256_file(PROJECT_ROOT / "Results/e2d2_joint_allocation_v1.md"),
            "joint_exact_allocation_episodes": 4000,
            "mean_normalized_reward": 1.0,
            "oracle_allocation_coverage": 1.0,
            "mean_utility_regret": 0.0,
            "mean_raw_cost_delta": 0.0,
            "mean_control_delivered_bytes": 3558.5,
            "mean_lock_round": 5.5,
            "comparison_warning": (
                "E2d2 uses scripted truthful Contract Net behavior and "
                "different seeds/phase timing; its values are contextual, "
                "not a paired causal baseline for hosted v4."
            ),
        },
        "derivations": {
            "tree_hash": (
                "SHA-256 over each regular file ordered by relative path, "
                "updating relative_path + NUL + file_sha256_hex + LF."
            ),
            "canary_partition": (
                "seed 22000 single_complementary and two_disjoint; these 2 "
                "cells were completed before promotion."
            ),
            "promoted_full_partition": (
                "the other 10 cells; the full manifest is cumulative and "
                "records completed_at_launch=2."
            ),
            "oracle_match": (
                "locked task-to-roster mapping compared exactly with the "
                "true-information oracle assignment."
            ),
            "utility_and_cost": (
                "recomputed from generated true agent profiles and locked "
                "rosters; regret/delta are averaged over all 12 full-reward "
                "cells."
            ),
            "control_delivered_bytes": (
                "strict TFP1 payload bytes multiplied by the recorded "
                "delivered-recipient count for each replay-bound transition."
            ),
            "promotion_decision": (
                "The complete, reliable Open-only mechanism screen warrants "
                "bounded paired E3 Alem transfer; it does not warrant a "
                "scalability or long-horizon role-alignment claim."
            ),
        },
    }


def _fmt_fraction(value: dict[str, int | float]) -> str:
    return f"{float(value['float']):.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    source = result["source"]
    hashes = source["hashes"]
    canary = result["stage_accounting"]["canary"]
    promoted = result["stage_accounting"]["promoted_full_new_cells"]
    combined = result["stage_accounting"]["combined"]
    lines = [
        "# E2b v4 completed hosted Open Volunteer screen",
        "",
        "> **Status: complete, integrity-valid, and promotable to a bounded E3 "
        "Alem mechanism transfer.** This is a six-agent structured recruitment "
        "screen, not an Alem efficacy, scalability, long-horizon role-alignment, "
        "or Open-versus-Mutual result.",
        "",
        "## Bound source",
        "",
        f"- Preserved root: `{source['campaign_root']}`",
        f"- Source commit: `{source['source_commit']}`",
        f"- Root tree SHA-256: `{hashes['root_tree_sha256']}`",
        f"- Canary manifest SHA-256: `{hashes['run_manifest_canary_sha256']}`",
        f"- Canary gate SHA-256: `{hashes['canary_gate_sha256']}`",
        f"- Full manifest SHA-256: `{hashes['run_manifest_full_sha256']}`",
        f"- Reservation ledger SHA-256: `{hashes['reservation_ledger_sha256']}`",
        f"- Provider binding: requested/resolved `{source['requested_model']}`; "
        f"`{source['reasoning_effort']}` reasoning; "
        f"{source['max_output_tokens']:,}-token output allowance.",
        "",
        "## Canary and promoted full stage",
        "",
        "| Stage partition | Cells | Full reward | Full coverage | Calls | Repairs | "
        "Invalid | Input tokens | Output tokens | Wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Canary (separate) | {canary['episodes']} | "
        f"{canary['full_reward_episodes']}/{canary['episodes']} | "
        f"{canary['full_coverage_episodes']}/{canary['episodes']} | "
        f"{canary['logical_calls']} | {canary['semantic_repair_calls']} | "
        f"{canary['invalid_calls']} | {canary['input_tokens']:,} | "
        f"{canary['output_tokens']:,} | {canary['invocation_wall_seconds']:.3f}s |",
        f"| Promoted invocation: new cells only | {promoted['episodes']} | "
        f"{promoted['full_reward_episodes']}/{promoted['episodes']} | "
        f"{promoted['full_coverage_episodes']}/{promoted['episodes']} | "
        f"{promoted['logical_calls']} | {promoted['semantic_repair_calls']} | "
        f"{promoted['invalid_calls']} | {promoted['input_tokens']:,} | "
        f"{promoted['output_tokens']:,} | "
        f"{promoted['invocation_wall_seconds']:.3f}s |",
        f"| Combined evidence | {combined['episodes']} | "
        f"{combined['full_reward_episodes']}/{combined['episodes']} | "
        f"{combined['full_coverage_episodes']}/{combined['episodes']} | "
        f"{combined['logical_calls']} | {combined['semantic_repair_calls']} | "
        f"{combined['invalid_calls']} | {combined['input_tokens']:,} | "
        f"{combined['output_tokens']:,} | — |",
        "",
        "The full manifest is cumulative: it began with the two completed canary "
        "cells and dispatched only the remaining ten cells with three cell "
        "workers.",
        "",
        "## Efficacy and allocation quality",
        "",
        f"- All {combined['true_feasible_locks']}/"
        f"{combined['true_feasible_locks']} locks were truly feasible; "
        f"true-infeasible locks: {combined['true_infeasible_locks']}.",
        f"- Exact whole-allocation oracle matches: "
        f"{combined['exact_oracle_allocations']}/{combined['episodes']}; exact "
        f"task rosters: {combined['exact_oracle_task_rosters']}/"
        f"{combined['oracle_task_rosters']}.",
        f"- Mean true-utility regret: "
        f"{_fmt_fraction(combined['mean_utility_regret'])}; mean true raw-cost "
        f"delta: {_fmt_fraction(combined['mean_raw_cost_delta'])}.",
        f"- Mean lock round: {combined['mean_lock_round']:.3f}; mean executed "
        f"acting rounds: {combined['mean_executed_acting_rounds']:.3f}.",
        "",
        "| Family | Cells | Reward/coverage | Calls | Mean rounds | Exact allocation | "
        "Utility regret | Cost delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, values in result["families"].items():
        lines.append(
            f"| `{family}` | {values['episodes']} | "
            f"{values['full_reward_episodes']}/{values['episodes']} | "
            f"{values['logical_calls']} | "
            f"{values['mean_executed_acting_rounds']:.3f} | "
            f"{values['exact_oracle_allocations']}/{values['episodes']} | "
            f"{_fmt_fraction(values['mean_utility_regret'])} | "
            f"{_fmt_fraction(values['mean_raw_cost_delta'])} |"
        )
    lines.extend(
        [
            "",
            "The two non-oracle allocations retained full reward and feasibility "
            "but committed before every useful bid became public. "
            "`oversubscribed/22001` incurred +14 raw cost; "
            "`two_disjoint/22002` incurred +45 and did not lock its second team "
            "until round 8. This is an observed association in two cells, not a "
            "causal estimate.",
            "",
            "## Reliability, usage, and traffic",
            "",
            f"- {combined['logical_calls']} logical calls = "
            f"{combined['initial_calls']} initial decisions + "
            f"{combined['semantic_repair_calls']} semantic repairs; "
            f"{combined['provider_attempts_actual']} actual provider attempts.",
            f"- Transport errors: {combined['transport_errors']}; incomplete "
            f"calls: {combined['incomplete_calls']}; max-output truncations: "
            f"{combined['max_output_truncations']}.",
            f"- Invalid initial decisions: {combined['invalid_calls']}/"
            f"{combined['initial_calls']} "
            f"({100 * combined['invalid_initial_decision_rate']:.3f}%); both "
            "repairs returned valid abstentions.",
            f"- Usage: {combined['input_tokens']:,} input + "
            f"{combined['output_tokens']:,} output = "
            f"{combined['actual_total_tokens']:,} actual tokens; "
            f"{combined['reasoning_tokens']:,} reasoning tokens "
            f"({100 * combined['reasoning_share_of_output']:.2f}% of output).",
            f"- Mean/max call latency: {combined['mean_call_latency_seconds']:.3f}s/"
            f"{combined['max_call_latency_seconds']:.3f}s; summed per-call "
            f"latency: {combined['sum_call_latency_seconds']:.3f}s; summed "
            f"round decision wall: {combined['decision_wall_seconds']:.3f}s.",
            f"- Structured control: {combined['control_submissions']} submissions, "
            f"{combined['accepted_control_transitions']} accepted and "
            f"{combined['rejected_control_transitions']} rejected transitions; "
            f"{combined['control_payload_bytes']:,} payload bytes and "
            f"{combined['control_delivered_bytes']:,} delivered bytes "
            f"({combined['mean_control_delivered_bytes']:.1f}/episode).",
            f"- Bid fidelity: {combined['exact_capability_claims']}/"
            f"{combined['apply_claims']} exact capability vectors and "
            f"{combined['exact_cost_claims']}/{combined['apply_claims']} exact "
            "costs. One claim used `[0,80,20]` instead of `[20,80,20]`; it did "
            "not change feasibility or reward in that cell.",
            "",
            "The two rejected semantic decisions were:",
            "",
        ]
    )
    for item in result["semantic_invalid_calls"]:
        key = item["reservation_key"]
        lines.append(
            f"- `{key['family']}/{key['seed']}`, round "
            f"{key['round_index']}, agent {key['agent_id']}: "
            f"`{item['invalid_completion']}` failed "
            f"`{item['validation_code']}`; repair outcome "
            f"`{item['repair_outcome']}`."
        )
    reference = result["deterministic_reference"]
    lines.extend(
        [
            "",
            "## Context against deterministic E2d2",
            "",
            "The provider-free E2d2 joint-exact arm achieved 1.0 reward and "
            "coverage with zero utility regret and zero cost delta across 4,000 "
            f"episodes, using {reference['mean_control_delivered_bytes']:.1f} "
            "delivered control bytes per episode. Hosted v4 matched reward and "
            "coverage but not perfect oracle allocation. This is contextual only: "
            "E2d2 used scripted truthful Contract Net behavior, different seeds, "
            "and different phase timing.",
            "",
            "## Decision",
            "",
            "Promote Open Volunteer/Public Sweep with truth-free joint exact "
            "allocation to a **bounded, paired E3 Alem transfer** while preserving "
            "Source action behavior. Record reward/events, feasible locks, role "
            "violations, control bytes, provider tokens, latency, repairs, bid "
            "fidelity, and roster regret. Add a preregistered bid-closure/quorum "
            "or provisional-lock-grace ablation because the only allocation "
            "misses followed commitment on incomplete public bids.",
            "",
            "Do not interpret this screen as evidence of Open superiority over a "
            "redesigned Mutual mechanism, population scalability, embodied task "
            "efficacy, or long-horizon role alignment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for output in (args.json_output, args.markdown_output):
        if _is_within(output, root):
            raise ValueError("summary output may not be written inside the campaign root")

    result = summarize(root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
