#!/usr/bin/env python3
"""Run named Alem OpenAI experiment matrices one difficulty at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.llm.experiment_config import (  # noqa: E402
    ABLATION_NAMES,
    PROFILE_NAMES,
    ExperimentConfigError,
    ExperimentSpec,
    compose_experiment,
    validate_experiment_config,
)

MANIFEST_NAME = "matrix_manifest.json"
RESOLVED_CONFIG_NAME = "resolved_config.yaml"


class MatrixLaunchError(RuntimeError):
    """Raised for invalid output or resume state."""


@dataclass(frozen=True)
class DifficultyRun:
    difficulty: str
    output_dir: Path
    completed_episode_units: int
    total_episode_units: int
    remaining_decision_call_cap: int
    remaining_debrief_call_cap: int
    command: tuple[str, ...]

    @property
    def remaining_episode_units(self) -> int:
        return self.total_episode_units - self.completed_episode_units


@dataclass(frozen=True)
class MatrixRun:
    profile: str
    ablation: str | None
    output_root: Path
    config_sha256: str
    source_commit: str | None
    uv_lock_sha256: str | None
    source_status: tuple[str, ...]
    spec: ExperimentSpec
    difficulties: tuple[DifficultyRun, ...]

    @property
    def remaining_decision_call_cap(self) -> int:
        return sum(item.remaining_decision_call_cap for item in self.difficulties)

    @property
    def remaining_debrief_call_cap(self) -> int:
        return sum(item.remaining_debrief_call_cap for item in self.difficulties)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validated Alem LLM matrix sequentially by difficulty. "
            "Completed episode JSON files are skipped when resuming."
        )
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="openai_reduced",
        help="Named experiment profile (default: openai_reduced).",
    )
    parser.add_argument(
        "--ablation",
        choices=ABLATION_NAMES,
        help="Optional hard-difficulty ablation manifest.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="Resume a matrix directory created by this launcher.",
    )
    output_group.add_argument(
        "--output-root",
        type=Path,
        metavar="RUN_DIR",
        help="Use this directory for a new run instead of the timestamped default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print directories/commands without creating files or calling an API.",
    )
    return parser


def _default_output_root(config, profile: str, ablation: str | None) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = f"_{ablation}" if ablation else ""
    base = Path(str(config.eval.output_dir))
    if not base.is_absolute():
        base = REPO_ROOT / base
    return (base / f"{timestamp}_{profile}{suffix}").resolve()


def _config_sha256(config) -> str:
    """Hash the fully resolved profile so resume cannot mix configurations."""

    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(config, resolve=True)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _git_status_lines() -> tuple[str, ...]:
    return tuple((_git_output("status", "--short") or "").splitlines())


def _lock_sha256() -> str | None:
    lock_path = REPO_ROOT / "uv.lock"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None


def _run_eval(command: Sequence[str]) -> subprocess.CompletedProcess:
    """Run one difficulty; isolated for tests without patching Git subprocesses."""

    return subprocess.run(command, cwd=REPO_ROOT, check=True)


def _episode_markers(config, output_dir: Path) -> tuple[Path, ...]:
    markers: list[Path] = []
    env_names = [name.strip() for name in str(config.envs.names).split("-") if name.strip()]
    for env_name in env_names:
        task_key = f"{env_name}_tasks"
        tasks = config.tasks.get(task_key)
        if tasks is None:
            raise MatrixLaunchError(f"Missing tasks.{task_key} for environment {env_name!r}")
        episodes = int(config.eval.num_episodes[env_name])
        for task in tasks:
            task_name = str(task)
            for episode_index in range(episodes):
                markers.append(
                    output_dir
                    / env_name
                    / task_name
                    / f"{task_name}_run_{episode_index:02d}.json"
                )
    return tuple(markers)


def _episode_marker_is_complete(path: Path) -> bool:
    """Mirror the evaluator's terminal-marker rule without importing JAX."""

    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != "alem-dice-episode-v1"
        or result.get("artifact_status") != "complete"
        or result.get("error")
        or not result.get("termination_reason")
    ):
        return False
    return result.get("early_stop_reason") != "consecutive_length_incomplete_responses"


def _command_for(
    *, profile: str, ablation: str | None, difficulty: str, output_dir: Path
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(REPO_ROOT / "baselines" / "llm" / "eval_alem.py"),
        f"experiment={profile}",
    ]
    if ablation:
        command.append(f"ablation={ablation}")
    command.extend(
        (
            f"alem.coordination_difficulty={difficulty}",
            f"eval.resume_from={output_dir}",
        )
    )
    return tuple(command)


def build_matrix_run(
    *,
    profile: str,
    ablation: str | None = None,
    output_root: Path | None = None,
) -> tuple[MatrixRun, object]:
    """Compose a profile and calculate its remaining episode/call budget."""

    config = compose_experiment(profile, ablation=ablation)
    spec = validate_experiment_config(config, expected_profile=profile)
    root = (output_root or _default_output_root(config, profile, ablation)).resolve()

    runs: list[DifficultyRun] = []
    for difficulty in spec.difficulties:
        difficulty_dir = root / difficulty
        markers = _episode_markers(config, difficulty_dir)
        completed = sum(_episode_marker_is_complete(marker) for marker in markers)
        remaining = len(markers) - completed
        runs.append(
            DifficultyRun(
                difficulty=difficulty,
                output_dir=difficulty_dir,
                completed_episode_units=completed,
                total_episode_units=len(markers),
                remaining_decision_call_cap=(
                    remaining * spec.max_steps_per_episode * spec.num_agents
                    if spec.requires_api_key
                    else 0
                ),
                remaining_debrief_call_cap=(remaining * spec.num_agents)
                if spec.generate_debriefs
                else 0,
                command=_command_for(
                    profile=profile,
                    ablation=ablation,
                    difficulty=difficulty,
                    output_dir=difficulty_dir,
                ),
            )
        )
    return (
        MatrixRun(
            profile=profile,
            ablation=ablation,
            output_root=root,
            config_sha256=_config_sha256(config),
            source_commit=_git_output("rev-parse", "HEAD"),
            uv_lock_sha256=_lock_sha256(),
            source_status=_git_status_lines(),
            spec=spec,
            difficulties=tuple(runs),
        ),
        config,
    )


def _manifest_payload(matrix: MatrixRun) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": matrix.profile,
        "ablation": matrix.ablation,
        "resolved_config_sha256": matrix.config_sha256,
        "source_commit": matrix.source_commit,
        "uv_lock_sha256": matrix.uv_lock_sha256,
        "difficulties": list(matrix.spec.difficulties),
        "seeds": list(matrix.spec.seeds),
        "num_agents": matrix.spec.num_agents,
        "max_steps_per_episode": matrix.spec.max_steps_per_episode,
        "num_workers": matrix.spec.num_workers,
        "model_ids": list(matrix.spec.model_ids),
        "generate_debriefs": matrix.spec.generate_debriefs,
    }


def _load_resume_manifest(output_root: Path) -> dict[str, object]:
    if not output_root.is_dir():
        raise MatrixLaunchError(f"Resume directory does not exist: {output_root}")
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise MatrixLaunchError(
            f"Resume directory is missing {MANIFEST_NAME}: {output_root}. "
            "Use the matrix root printed by this launcher, not a difficulty subdirectory."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixLaunchError(f"Could not read resume manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MatrixLaunchError(f"Resume manifest must contain a JSON object: {manifest_path}")
    return payload


def _validate_resume_manifest(matrix: MatrixRun, actual: dict[str, object]) -> None:
    expected = _manifest_payload(matrix)
    mismatches = [
        key for key, value in expected.items() if actual.get(key) != value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: expected {expected[key]!r}, found {actual.get(key)!r}"
            for key in mismatches
        )
        raise MatrixLaunchError(f"Resume manifest does not match requested matrix ({details})")


def _prepare_new_output(matrix: MatrixRun, config) -> None:
    if matrix.output_root.exists() and not matrix.output_root.is_dir():
        raise MatrixLaunchError(
            f"New-run output path exists but is not a directory: {matrix.output_root}"
        )
    if matrix.output_root.exists() and any(matrix.output_root.iterdir()):
        raise MatrixLaunchError(
            f"New-run output directory is not empty: {matrix.output_root}. "
            "Use --resume to continue an existing matrix."
        )
    matrix.output_root.mkdir(parents=True, exist_ok=True)
    payload = _manifest_payload(matrix)
    payload["created_at_utc"] = datetime.now(UTC).isoformat()
    (matrix.output_root / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from omegaconf import OmegaConf

    OmegaConf.save(config=config, f=str(matrix.output_root / RESOLVED_CONFIG_NAME), resolve=True)


def _print_matrix(matrix: MatrixRun) -> None:
    ablation = matrix.ablation or "none"
    max_concurrent = matrix.spec.num_workers * matrix.spec.num_agents
    print(f"Profile: {matrix.profile}")
    print(f"Ablation: {ablation}")
    print(f"Output root: {matrix.output_root}")
    print(f"Source commit: {matrix.source_commit or 'unknown'}")
    print(f"UV lock SHA-256: {matrix.uv_lock_sha256 or 'missing'}")
    print(
        "Nominal decision-call cap: "
        f"{matrix.spec.nominal_decision_call_cap:,} (excluding transport retries)"
    )
    print(
        f"Remaining decision-call cap: {matrix.remaining_decision_call_cap:,} "
        "(excluding transport retries)"
    )
    if matrix.spec.generate_debriefs:
        print(
            f"Remaining debrief-call cap: {matrix.remaining_debrief_call_cap:,}"
        )
    print(f"Maximum concurrent decision calls: {max_concurrent}")
    for run in matrix.difficulties:
        print(
            f"  {run.difficulty}: {run.output_dir} | completed "
            f"{run.completed_episode_units}/{run.total_episode_units} | "
            f"remaining decision-call cap {run.remaining_decision_call_cap:,}"
        )
        print(f"    {shlex.join(run.command)}")


def run_matrix(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        resume_root = args.resume.resolve() if args.resume else None
        requested_root = (
            resume_root
            if resume_root is not None
            else args.output_root.resolve()
            if args.output_root
            else None
        )
        matrix, config = build_matrix_run(
            profile=args.profile,
            ablation=args.ablation,
            output_root=requested_root,
        )

        if resume_root is not None:
            _validate_resume_manifest(matrix, _load_resume_manifest(resume_root))

        _print_matrix(matrix)
        if args.dry_run:
            print("Dry run complete; no directories were created and no API calls were made.")
            return 0

        if matrix.spec.requires_api_key and not os.environ.get("OPENAI_API_KEY", "").strip():
            raise MatrixLaunchError(
                "OPENAI_API_KEY is required for this profile. Export it in the current shell "
                "or use --dry-run to inspect the matrix without API calls."
            )
        if matrix.spec.requires_api_key and matrix.source_status:
            changed = "\n  ".join(matrix.source_status)
            raise MatrixLaunchError(
                "Paid profiles require a clean Git worktree so results are reproducible. "
                "Commit or stash these changes before launching:\n  " + changed
            )
        if matrix.spec.requires_api_key and (
            matrix.source_commit is None or matrix.uv_lock_sha256 is None
        ):
            raise MatrixLaunchError(
                "Paid profiles require both a Git source commit and uv.lock."
            )

        if resume_root is None:
            _prepare_new_output(matrix, config)

        for run in matrix.difficulties:
            if run.remaining_episode_units == 0:
                print(f"Skipping {run.difficulty}: all episode outputs already exist.")
                continue
            run.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running {run.difficulty}; output: {run.output_dir}", flush=True)
            _run_eval(run.command)
        print(f"Matrix complete. Saved at: {matrix.output_root}")
        return 0
    except (ExperimentConfigError, MatrixLaunchError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run_matrix())


if __name__ == "__main__":
    main()
