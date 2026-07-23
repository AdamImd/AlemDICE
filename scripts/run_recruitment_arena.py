#!/usr/bin/env python3
"""Run the provider-free, parallel E2a RecruitmentArena campaign."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.llm.eval_utils.recruitment_selection import (  # noqa: E402
    true_information_oracle,
)
from baselines.llm.eval_utils.team_formation import RecruitmentMethod  # noqa: E402
from baselines.llm.recruitment_arena import (  # noqa: E402
    DEFAULT_ROUNDS,
    ControlArm,
    ScenarioFamily,
    TaskChoicePolicy,
    canonical_json,
    generate_scenario,
    run_control_episode,
    run_scripted_episode,
)

SCHEMA_VERSION = "alem-dice-e2a-campaign-v1"
DEFAULT_SEED_START = 20000
DEFAULT_NUM_SEEDS = 1000
DEFAULT_OUTPUT = Path("outputs/recruitment_arena/e2a_v1")
METHOD_CONFIGS = tuple(
    (method, task_choice) for method in RecruitmentMethod for task_choice in TaskChoicePolicy
)
METHODS = tuple(f"{method.value}__{task_choice.value}" for method, task_choice in METHOD_CONFIGS)
CONTROLS = tuple(arm.value for arm in ControlArm)
ARMS = (*METHODS, *CONTROLS)
FAMILIES = tuple(family.value for family in ScenarioFamily)


@dataclass(frozen=True)
class ShardJob:
    shard_id: int
    seeds: tuple[int, ...]
    output_path: str


@dataclass(frozen=True)
class ShardResult:
    shard_id: int
    seeds: tuple[int, ...]
    output_path: str
    content_sha256: str
    compressed_sha256: str
    episodes: tuple[dict[str, Any], ...]
    wall_seconds: float


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction_dict(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "float": float(value),
    }


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _episode_row(episode: Any, episode_sha256: str) -> dict[str, Any]:
    metrics = episode.metrics
    return {
        "seed": metrics.seed,
        "family": metrics.family.value,
        "arm": metrics.method,
        "episode_sha256": episode_sha256,
        "oracle_reward": metrics.oracle_reward,
        "achieved_reward": metrics.achieved_reward,
        "normalized_reward": _float_or_none(metrics.normalized_reward),
        "reward_regret": metrics.reward_regret,
        "oracle_utility": float(metrics.oracle_utility),
        "achieved_utility": _float_or_none(metrics.achieved_utility),
        "utility_regret": _float_or_none(metrics.utility_regret),
        "oracle_raw_cost": float(metrics.oracle_raw_cost),
        "achieved_raw_cost": _float_or_none(metrics.achieved_raw_cost),
        "oracle_assignment": canonical_json(metrics.oracle_assignment),
        "completed_tasks": metrics.completed_tasks,
        "formation_opportunities": metrics.formation_opportunities,
        "true_feasible_locks": metrics.true_feasible_locks,
        "true_infeasible_locks": metrics.true_infeasible_locks,
        "formation_rate": float(metrics.formation_rate),
        "oracle_allocation_opportunities": metrics.oracle_allocation_opportunities,
        "oracle_allocation_coverage": _float_or_none(metrics.oracle_allocation_coverage),
        "mean_lock_round": (mean(metrics.lock_rounds) if metrics.lock_rounds else None),
        "control_submissions": metrics.control_submissions,
        "valid_control_submissions": metrics.valid_control_submissions,
        "accepted_control_transitions": metrics.accepted_control_transitions,
        "rejected_control_transitions": metrics.rejected_control_transitions,
        "control_payload_bytes": metrics.control_payload_bytes,
        "control_delivered_bytes": metrics.control_delivered_bytes,
        "ordinary_deliveries": metrics.ordinary_deliveries,
        "ordinary_payload_bytes": metrics.ordinary_payload_bytes,
        "ordinary_delivered_bytes": metrics.ordinary_delivered_bytes,
        "unauthorized_ordinary_deliveries": (metrics.unauthorized_ordinary_deliveries),
        "roster_agreement_rate": _float_or_none(metrics.roster_agreement_rate),
        "overlapping_roster_rounds": metrics.overlapping_roster_rounds,
        "multi_offer_agent_rounds": metrics.multi_offer_agent_rounds,
        "roster_revision_count": metrics.roster_revision_count,
        "overstaff_agent_slots": metrics.overstaff_agent_slots,
        "exact_roster_comparator": metrics.exact_roster_comparator,
        "replay_hash_match": metrics.replay_hash_match,
        "terminal_state_hash": metrics.terminal_state_hash,
        "terminal_audit_chain_hash": metrics.terminal_audit_chain_hash,
        "model_calls": metrics.model_calls,
        "provider_requests": metrics.provider_requests,
        "transport_errors": metrics.transport_errors,
    }


def _run_shard(job: ShardJob) -> ShardResult:
    started = time.monotonic()
    path = Path(job.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content_digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="\n") as text:
                for seed in job.seeds:
                    for family_text in FAMILIES:
                        scenario = generate_scenario(family_text, seed)
                        oracle = true_information_oracle(
                            scenario.tasks,
                            scenario.agents,
                        )
                        for method, task_choice in METHOD_CONFIGS:
                            episode = run_scripted_episode(
                                scenario,
                                method,
                                oracle=oracle,
                                task_choice=task_choice,
                            )
                            encoded = canonical_json(episode.as_dict()).encode("ascii")
                            episode_sha256 = hashlib.sha256(encoded).hexdigest()
                            line = encoded + b"\n"
                            content_digest.update(line)
                            text.write(line.decode("ascii"))
                            rows.append(_episode_row(episode, episode_sha256))
                        for arm in ControlArm:
                            episode = run_control_episode(
                                scenario,
                                arm,
                                oracle=oracle,
                            )
                            encoded = canonical_json(episode.as_dict()).encode("ascii")
                            episode_sha256 = hashlib.sha256(encoded).hexdigest()
                            line = encoded + b"\n"
                            content_digest.update(line)
                            text.write(line.decode("ascii"))
                            rows.append(_episode_row(episode, episode_sha256))
    return ShardResult(
        shard_id=job.shard_id,
        seeds=job.seeds,
        output_path=str(path),
        content_sha256=content_digest.hexdigest(),
        compressed_sha256=_sha256(path),
        episodes=tuple(rows),
        wall_seconds=time.monotonic() - started,
    )


def _partition(seeds: tuple[int, ...], workers: int) -> tuple[tuple[int, ...], ...]:
    shard_count = min(workers, len(seeds))
    base, remainder = divmod(len(seeds), shard_count)
    shards = []
    offset = 0
    for shard_id in range(shard_count):
        size = base + (1 if shard_id < remainder else 0)
        shards.append(seeds[offset : offset + size])
        offset += size
    assert offset == len(seeds)
    return tuple(shards)


def _arm_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(rows)
    opportunities = sum(int(row["formation_opportunities"]) for row in values)
    feasible_locks = sum(int(row["true_feasible_locks"]) for row in values)
    oracle_opportunities = sum(
        int(row["oracle_allocation_opportunities"])
        for row in values
        if row["oracle_allocation_coverage"] is not None
    )
    completed_for_oracle = sum(
        int(row["completed_tasks"])
        for row in values
        if row["oracle_allocation_coverage"] is not None
    )
    normalized_values = [
        float(row["normalized_reward"]) for row in values if row["normalized_reward"] is not None
    ]
    regret_values = [
        float(row["reward_regret"]) for row in values if row["reward_regret"] is not None
    ]
    utility_regrets = [
        float(row["utility_regret"]) for row in values if row["utility_regret"] is not None
    ]
    allocation_coverages = [
        float(row["oracle_allocation_coverage"])
        for row in values
        if row["oracle_allocation_coverage"] is not None
    ]
    return {
        "episodes": len(values),
        "mean_normalized_reward": (mean(normalized_values) if normalized_values else None),
        "mean_reward_regret": mean(regret_values) if regret_values else None,
        "mean_utility_regret_at_equal_reward": (mean(utility_regrets) if utility_regrets else None),
        "mean_oracle_allocation_coverage": (
            mean(allocation_coverages) if allocation_coverages else None
        ),
        "aggregate_oracle_allocation_coverage": (
            completed_for_oracle / oracle_opportunities if oracle_opportunities else None
        ),
        "formation_opportunities": opportunities,
        "true_feasible_locks": feasible_locks,
        "aggregate_formation_rate": (feasible_locks / opportunities if opportunities else 1.0),
        "mean_lock_round": mean(
            float(row["mean_lock_round"]) for row in values if row["mean_lock_round"] is not None
        )
        if any(row["mean_lock_round"] is not None for row in values)
        else None,
        "control_submissions": sum(int(row["control_submissions"]) for row in values),
        "control_delivered_bytes": sum(int(row["control_delivered_bytes"]) for row in values),
        "ordinary_delivered_bytes": sum(int(row["ordinary_delivered_bytes"]) for row in values),
        "overlapping_roster_rounds": sum(int(row["overlapping_roster_rounds"]) for row in values),
        "multi_offer_agent_rounds": sum(int(row["multi_offer_agent_rounds"]) for row in values),
        "roster_revision_count": sum(int(row["roster_revision_count"]) for row in values),
        "overstaff_agent_slots": sum(int(row["overstaff_agent_slots"]) for row in values),
        "invalid_control_submissions": sum(
            int(row["control_submissions"]) - int(row["valid_control_submissions"])
            for row in values
        ),
        "rejected_control_transitions": sum(
            int(row["rejected_control_transitions"]) for row in values
        ),
        "unauthorized_ordinary_deliveries": sum(
            int(row["unauthorized_ordinary_deliveries"]) for row in values
        ),
        "replay_hash_failures": sum(row["replay_hash_match"] is False for row in values),
        "model_calls": sum(int(row["model_calls"]) for row in values),
        "provider_requests": sum(int(row["provider_requests"]) for row in values),
        "transport_errors": sum(int(row["transport_errors"]) for row in values),
    }


def _build_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    by_arm_family: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = tuple(row for row in rows if row["arm"] == arm)
        by_arm[arm] = _arm_summary(arm_rows)
        by_arm_family[arm] = {
            family: _arm_summary(row for row in arm_rows if row["family"] == family)
            for family in FAMILIES
        }

    integrity_gates = {
        "all_expected_episodes": len(rows)
        == len({row["seed"] for row in rows}) * len(FAMILIES) * len(ARMS),
        "unique_seed_family_arm_rows": len(rows)
        == len({(row["seed"], row["family"], row["arm"]) for row in rows}),
        "zero_invalid_scripted_records": all(
            summary["invalid_control_submissions"] == 0
            for arm, summary in by_arm.items()
            if arm in METHODS
        ),
        "zero_rejected_scripted_transitions": all(
            summary["rejected_control_transitions"] == 0
            for arm, summary in by_arm.items()
            if arm in METHODS
        ),
        "zero_unauthorized_ordinary_deliveries": all(
            summary["unauthorized_ordinary_deliveries"] == 0 for summary in by_arm.values()
        ),
        "all_replay_hashes_match": all(
            summary["replay_hash_failures"] == 0 for summary in by_arm.values()
        ),
        "zero_model_calls": all(summary["model_calls"] == 0 for summary in by_arm.values()),
        "zero_provider_requests": all(
            summary["provider_requests"] == 0 for summary in by_arm.values()
        ),
        "zero_transport_errors": all(
            summary["transport_errors"] == 0 for summary in by_arm.values()
        ),
        "full_roster_agreement": all(
            row["roster_agreement_rate"] is None or float(row["roster_agreement_rate"]) == 1.0
            for row in rows
        ),
    }
    promotion = {
        method: {
            "oracle_allocation_coverage": by_arm[method]["aggregate_oracle_allocation_coverage"],
            "feasible_card_formation_rate": by_arm[method]["aggregate_formation_rate"],
            "passes_80_percent_formation_gate": (
                by_arm[method]["aggregate_oracle_allocation_coverage"] >= 0.8
            ),
        }
        for method in METHODS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len(rows),
        "seed_count": len({row["seed"] for row in rows}),
        "arms": by_arm,
        "arms_by_family": by_arm_family,
        "integrity_gates": integrity_gates,
        "integrity_passed": all(integrity_gates.values()),
        "method_promotion_gates": promotion,
        "at_least_one_method_promotable": any(
            value["passes_80_percent_formation_gate"] for value in promotion.values()
        ),
    }


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    if not rows:
        raise ValueError("cannot write an empty episode table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    seeds: tuple[int, ...],
    workers: int,
) -> None:
    lines = [
        "# E2a RecruitmentArena-6 Results",
        "",
        (
            f"Provider-free scripted campaign: {len(seeds)} seeds "
            f"({seeds[0]}–{seeds[-1]}), {len(FAMILIES)} scenario families, "
            f"{len(ARMS)} arms, {workers} process workers."
        ),
        "",
        "| Arm | Episodes | Normalized reward | Formation rate | Mean lock round | "
        "Control delivered bytes | Invalid/rejected | Promotion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for arm in ARMS:
        value = summary["arms"][arm]
        promotion = summary["method_promotion_gates"].get(arm)
        promotion_text = (
            "PASS"
            if promotion and promotion["passes_80_percent_formation_gate"]
            else "FAIL"
            if promotion
            else "reference"
        )
        mean_lock = value["mean_lock_round"]
        normalized_text = (
            f"{value['mean_normalized_reward']:.3f}"
            if value["mean_normalized_reward"] is not None
            else "—"
        )
        mean_lock_text = f"{mean_lock:.2f}" if mean_lock is not None else "—"
        lines.append(
            f"| `{arm}` | {value['episodes']} | "
            f"{normalized_text} | "
            f"{value['aggregate_formation_rate']:.3f} | "
            f"{mean_lock_text} | "
            f"{value['control_delivered_bytes']:,} | "
            f"{value['invalid_control_submissions']}/"
            f"{value['rejected_control_transitions']} | {promotion_text} |"
        )
    lines.extend(
        [
            "",
            "## Integrity gates",
            "",
            *[
                f"- {'PASS' if passed else 'FAIL'} — `{name}`"
                for name, passed in summary["integrity_gates"].items()
            ],
            "",
            (
                "The 80% formation threshold is a method-level promotion gate, "
                "not an artifact-integrity condition. Non-passing methods and "
                "topology controls remain in the audit record."
            ),
            "",
            "All task truth used for completion and the oracle is absent from "
            "live protocol selection. E2a makes zero model/provider calls.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_hashes() -> dict[str, str]:
    paths = (
        Path("baselines/llm/eval_utils/team_formation.py"),
        Path("baselines/llm/eval_utils/recruitment_selection.py"),
        Path("baselines/llm/recruitment_arena.py"),
        Path("scripts/run_recruitment_arena.py"),
    )
    return {str(path): _sha256(PROJECT_ROOT / path) for path in paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.rounds != DEFAULT_ROUNDS:
        raise ValueError(f"E2a is frozen at --rounds {DEFAULT_ROUNDS}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    partial = output.with_name(f"{output.name}.partial-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")
    partial.mkdir(parents=True)
    (partial / "events").mkdir()
    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    shards = _partition(seeds, args.workers)
    jobs = tuple(
        ShardJob(
            shard_id=shard_id,
            seeds=shard,
            output_path=str(partial / "events" / f"episodes_{shard_id:03d}.jsonl.gz"),
        )
        for shard_id, shard in enumerate(shards)
    )
    started_at = datetime.now(UTC)
    started = time.monotonic()
    run_state = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": started_at.isoformat(),
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "rounds": args.rounds,
        "workers": len(jobs),
        "families": list(FAMILIES),
        "methods": list(METHODS),
        "controls": list(CONTROLS),
        "model_calls": 0,
        "provider_requests": 0,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_status": _git("status", "--short"),
        "source_hashes": _source_hashes(),
        "python": platform.python_version(),
    }
    (partial / "run_state.json").write_text(
        json.dumps(run_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        results: list[ShardResult] = []
        if len(jobs) == 1:
            results.append(_run_shard(jobs[0]))
        else:
            with ProcessPoolExecutor(
                max_workers=len(jobs),
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = {executor.submit(_run_shard, job): job for job in jobs}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(
                        f"completed shard {result.shard_id}: "
                        f"{len(result.seeds)} seeds, "
                        f"{len(result.episodes)} episodes, "
                        f"{result.wall_seconds:.2f}s",
                        flush=True,
                    )
        results.sort(key=lambda result: result.shard_id)
        rows = tuple(
            sorted(
                (row for result in results for row in result.episodes),
                key=lambda row: (
                    int(row["seed"]),
                    FAMILIES.index(str(row["family"])),
                    ARMS.index(str(row["arm"])),
                ),
            )
        )
        summary = _build_summary(rows)
        _write_csv(partial / "episodes.csv", rows)
        (partial / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_markdown(
            partial / "summary.md",
            summary=summary,
            seeds=seeds,
            workers=len(jobs),
        )
        ended_at = datetime.now(UTC)
        manifest = {
            **run_state,
            "status": "complete",
            "ended_at": ended_at.isoformat(),
            "wall_seconds": time.monotonic() - started,
            "episode_count": len(rows),
            "integrity_passed": summary["integrity_passed"],
            "shards": [
                {
                    "shard_id": result.shard_id,
                    "seeds": list(result.seeds),
                    "path": str(Path(result.output_path).relative_to(partial)),
                    "content_sha256": result.content_sha256,
                    "compressed_sha256": result.compressed_sha256,
                    "episode_count": len(result.episodes),
                    "wall_seconds": result.wall_seconds,
                }
                for result in results
            ],
        }
        artifact_paths = (
            partial / "episodes.csv",
            partial / "summary.json",
            partial / "summary.md",
        )
        manifest["artifact_sha256"] = {path.name: _sha256(path) for path in artifact_paths}
        (partial / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (partial / "run_state.json").unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(output)
        print(
            f"E2a wrote {len(rows)} episodes to {output} "
            f"in {manifest['wall_seconds']:.2f}s; "
            f"integrity={'PASS' if summary['integrity_passed'] else 'FAIL'}",
            flush=True,
        )
        return 0 if summary["integrity_passed"] else 2
    except BaseException:
        failed = partial.with_name(
            f"{output.name}.failed-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        if partial.exists():
            if failed.exists():
                shutil.rmtree(failed)
            partial.replace(failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
