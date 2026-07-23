#!/usr/bin/env python3
"""Run the provider-free E2d Contract Net roster-selector ablation."""

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
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.llm.eval_utils.recruitment_selection import (  # noqa: E402
    SelectionMethod,
    true_information_oracle,
)
from baselines.llm.eval_utils.team_formation import RecruitmentMethod  # noqa: E402
from baselines.llm.recruitment_arena import (  # noqa: E402
    DEFAULT_ROUNDS,
    ScenarioFamily,
    TaskChoicePolicy,
    canonical_json,
    generate_scenario,
    run_scripted_episode,
)
from scripts.run_recruitment_arena import (  # noqa: E402
    _episode_row,
    _partition,
    _sha256,
)

SCHEMA_VERSION = "alem-dice-e2d-selector-campaign-v1"
DEFAULT_SEED_START = 22200
DEFAULT_NUM_SEEDS = 1000
DEFAULT_OUTPUT = Path("outputs/recruitment_arena/e2d_selector_v1")
SELECTORS = tuple(SelectionMethod)
ARMS = tuple(
    f"{RecruitmentMethod.CONTRACT_NET.value}__"
    f"{TaskChoicePolicy.PUBLIC_SWEEP.value}__{selector.value}"
    for selector in SELECTORS
)
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


def _selector_audit(episode: Any) -> dict[str, Any]:
    event = next(item for item in episode.events if item["phase"] == "selector_audit")
    decisions = tuple(event["decisions"])
    return {
        "selector_decisions": len(decisions),
        "selector_claimed_only": (
            event["information_source"] == "claimed"
            and all(decision["information_source"] == "claimed" for decision in decisions)
            and "true_" not in canonical_json(event)
        ),
        "random_seed_count": sum(decision["random_seed"] is not None for decision in decisions),
    }


def _run_shard(job: ShardJob) -> ShardResult:
    started = time.monotonic()
    path = Path(job.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content_digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="\n") as text:
                for seed in job.seeds:
                    for family_text in FAMILIES:
                        scenario = generate_scenario(family_text, seed)
                        oracle = true_information_oracle(scenario.tasks, scenario.agents)
                        for selector in SELECTORS:
                            episode = run_scripted_episode(
                                scenario,
                                RecruitmentMethod.CONTRACT_NET,
                                oracle=oracle,
                                task_choice=TaskChoicePolicy.PUBLIC_SWEEP,
                                selector=selector,
                            )
                            encoded = canonical_json(episode.as_dict()).encode("ascii")
                            episode_sha256 = hashlib.sha256(encoded).hexdigest()
                            line = encoded + b"\n"
                            content_digest.update(line)
                            text.write(line.decode("ascii"))
                            rows.append(
                                {
                                    **_episode_row(episode, episode_sha256),
                                    **_selector_audit(episode),
                                }
                            )
    return ShardResult(
        shard_id=job.shard_id,
        seeds=job.seeds,
        output_path=str(path),
        content_sha256=content_digest.hexdigest(),
        compressed_sha256=_sha256(path),
        episodes=tuple(rows),
        wall_seconds=time.monotonic() - started,
    )


def _selector_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(rows)
    equal_reward = tuple(row for row in values if row["achieved_reward"] == row["oracle_reward"])
    oracle_opportunities = sum(int(row["oracle_allocation_opportunities"]) for row in values)
    completed = sum(int(row["completed_tasks"]) for row in values)
    achieved_costs = [
        float(row["achieved_raw_cost"]) for row in values if row["achieved_raw_cost"] is not None
    ]
    completed_tasks = sum(int(row["completed_tasks"]) for row in values)
    return {
        "episodes": len(values),
        "equal_reward_episodes": len(equal_reward),
        "mean_normalized_reward": mean(float(row["normalized_reward"]) for row in values),
        "aggregate_oracle_allocation_coverage": (
            completed / oracle_opportunities if oracle_opportunities else 1.0
        ),
        "mean_utility_regret_at_equal_reward": (
            mean(float(row["utility_regret"]) for row in equal_reward) if equal_reward else None
        ),
        "mean_achieved_utility_at_equal_reward": (
            mean(float(row["achieved_utility"]) for row in equal_reward) if equal_reward else None
        ),
        "mean_oracle_utility_at_equal_reward": (
            mean(float(row["oracle_utility"]) for row in equal_reward) if equal_reward else None
        ),
        "mean_achieved_raw_cost": mean(achieved_costs) if achieved_costs else None,
        "mean_achieved_raw_cost_at_equal_reward": (
            mean(float(row["achieved_raw_cost"]) for row in equal_reward) if equal_reward else None
        ),
        "mean_oracle_raw_cost_at_equal_reward": (
            mean(float(row["oracle_raw_cost"]) for row in equal_reward) if equal_reward else None
        ),
        "mean_raw_cost_delta_at_equal_reward": (
            mean(
                float(row["achieved_raw_cost"]) - float(row["oracle_raw_cost"])
                for row in equal_reward
            )
            if equal_reward
            else None
        ),
        "mean_raw_cost_per_completed_task": (
            sum(achieved_costs) / completed_tasks if completed_tasks else None
        ),
        "mean_lock_round": mean(
            float(row["mean_lock_round"]) for row in values if row["mean_lock_round"] is not None
        ),
        "control_submissions": sum(int(row["control_submissions"]) for row in values),
        "control_payload_bytes": sum(int(row["control_payload_bytes"]) for row in values),
        "control_delivered_bytes": sum(int(row["control_delivered_bytes"]) for row in values),
        "mean_control_delivered_bytes": mean(int(row["control_delivered_bytes"]) for row in values),
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
        "overstaff_agent_slots": sum(int(row["overstaff_agent_slots"]) for row in values),
        "replay_hash_failures": sum(row["replay_hash_match"] is False for row in values),
        "roster_agreement_failures": sum(
            row["roster_agreement_rate"] is not None and float(row["roster_agreement_rate"]) != 1.0
            for row in values
        ),
        "selector_decisions": sum(int(row["selector_decisions"]) for row in values),
        "selector_information_boundary_failures": sum(
            not bool(row["selector_claimed_only"]) for row in values
        ),
        "random_seed_count": sum(int(row["random_seed_count"]) for row in values),
        "model_calls": sum(int(row["model_calls"]) for row in values),
        "provider_requests": sum(int(row["provider_requests"]) for row in values),
        "transport_errors": sum(int(row["transport_errors"]) for row in values),
    }


def _build_summary(rows: tuple[dict[str, Any], ...], seed_count: int) -> dict[str, Any]:
    by_arm = {arm: _selector_summary(row for row in rows if row["arm"] == arm) for arm in ARMS}
    by_arm_family = {
        arm: {
            family: _selector_summary(
                row for row in rows if row["arm"] == arm and row["family"] == family
            )
            for family in FAMILIES
        }
        for arm in ARMS
    }
    integrity_gates = {
        "all_expected_episodes": len(rows) == seed_count * len(FAMILIES) * len(ARMS),
        "unique_seed_family_arm_rows": len(rows)
        == len({(row["seed"], row["family"], row["arm"]) for row in rows}),
        "zero_invalid_records": all(
            value["invalid_control_submissions"] == 0 for value in by_arm.values()
        ),
        "zero_rejected_transitions": all(
            value["rejected_control_transitions"] == 0 for value in by_arm.values()
        ),
        "zero_unauthorized_ordinary_deliveries": all(
            value["unauthorized_ordinary_deliveries"] == 0 for value in by_arm.values()
        ),
        "exact_six_agent_exclusivity": all(
            value["overstaff_agent_slots"] == 0 for value in by_arm.values()
        ),
        "all_replay_hashes_match": all(
            value["replay_hash_failures"] == 0 for value in by_arm.values()
        ),
        "full_roster_agreement": all(
            value["roster_agreement_failures"] == 0 for value in by_arm.values()
        ),
        "claimed_information_only": all(
            value["selector_information_boundary_failures"] == 0 for value in by_arm.values()
        ),
        "all_episodes_audited": all(
            value["selector_decisions"] >= value["episodes"] for value in by_arm.values()
        ),
        "zero_model_provider_transport": all(
            value["model_calls"] == value["provider_requests"] == value["transport_errors"] == 0
            for value in by_arm.values()
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len(rows),
        "seed_count": seed_count,
        "arms": by_arm,
        "arms_by_family": by_arm_family,
        "integrity_gates": integrity_gates,
        "integrity_passed": all(integrity_gates.values()),
        "interpretation_guardrail": (
            "exact_utility is a task-local claimed-information selector, not the "
            "global true-information allocation oracle"
        ),
    }


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
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
        "# E2d Contract Net selector ablation",
        "",
        (
            f"Provider-free paired campaign: {len(seeds)} seeds "
            f"({seeds[0]}–{seeds[-1]}), {len(FAMILIES)} scenario families, "
            f"{len(ARMS)} claimed-information selector arms, {workers} process workers."
        ),
        "",
        "| Selector | Episodes | Reward | Oracle cover | Utility regret* | "
        "Cost delta* | Bytes/ep | Lock round |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector, arm in zip(SELECTORS, ARMS, strict=True):
        value = summary["arms"][arm]
        lines.append(
            f"| `{selector.value}` | {value['episodes']} | "
            f"{value['mean_normalized_reward']:.4f} | "
            f"{value['aggregate_oracle_allocation_coverage']:.4f} | "
            f"{value['mean_utility_regret_at_equal_reward']:.6f} | "
            f"{value['mean_raw_cost_delta_at_equal_reward']:.3f} | "
            f"{value['mean_control_delivered_bytes']:.1f} | "
            f"{value['mean_lock_round']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "\\* Utility regret and raw-cost delta are conditioned on episodes "
                "that match the oracle's completed reward. Cost delta is descriptive: "
                "the oracle maximizes utility before minimizing cost."
            ),
            "",
            "## Integrity gates",
            "",
            *[
                f"- {'PASS' if passed else 'FAIL'} — `{name}`"
                for name, passed in summary["integrity_gates"].items()
            ],
            "",
            (
                "`exact_utility` is a task-local selector over delivered bids and "
                "claimed capabilities/costs. It is not the global true-information oracle."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_hashes() -> dict[str, str]:
    paths = (
        Path("baselines/llm/eval_utils/team_formation.py"),
        Path("baselines/llm/eval_utils/recruitment_selection.py"),
        Path("baselines/llm/recruitment_arena.py"),
        Path("scripts/run_recruitment_selector_ablation.py"),
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
        raise ValueError(f"E2d is frozen at --rounds {DEFAULT_ROUNDS}")
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
        "selectors": [selector.value for selector in SELECTORS],
        "arms": list(ARMS),
        "information_source": "claimed",
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
        summary = _build_summary(rows, len(seeds))
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
        manifest = {
            **run_state,
            "status": "complete",
            "ended_at": datetime.now(UTC).isoformat(),
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
            f"E2d wrote {len(rows)} episodes to {output} "
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
