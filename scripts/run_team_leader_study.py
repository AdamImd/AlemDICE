#!/usr/bin/env python3
"""Run the matched three-arm bodyless-team-leader pilot."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.llm.experiment_config import compose_experiment  # noqa: E402
from scripts.run_openai_matrix import (  # noqa: E402
    _config_sha256,
    _episode_marker_is_complete,
    _git_output,
    _git_status_lines,
    _lock_sha256,
)

PROFILE = "team_leader_200"
ARMS = ("baseline", "leader_peer", "leader_no_peer")
MANIFEST_NAME = "study_manifest.json"
RESOLVED_CONFIG_NAME = "resolved_config.yaml"
WORKER_CALLS_PER_ARM = 3 * 200
LEADER_CALLS_PER_ARM = 200 // 5
LOGICAL_CALL_CAP = WORKER_CALLS_PER_ARM + 2 * (WORKER_CALLS_PER_ARM + LEADER_CALLS_PER_ARM)


class StudyLaunchError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--resume", type=Path, metavar="RUN_DIR")
    output.add_argument("--output-root", type=Path, metavar="RUN_DIR")
    output.add_argument("--summarize", type=Path, metavar="RUN_DIR")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Make exactly one worker and one leader compatibility request, then exit.",
    )
    return parser


def _default_root(config) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = Path(str(config.eval.output_dir))
    if not base.is_absolute():
        base = REPO_ROOT / base
    return (base / f"{timestamp}_team_leader_200").resolve()


def _cache_key(arm: str, role: str) -> str:
    arm_codes = {
        "baseline": "b",
        "leader_peer": "lp",
        "leader_no_peer": "ln",
        "preflight": "p",
    }
    role_codes = {
        "warrior": "w",
        "forager": "f",
        "miner": "m",
        "leader": "l",
        "worker": "w",
    }
    return f"ad:g54h:tl1:{arm_codes[arm]}:{role_codes[role]}"


def _command(arm: str, arm_dir: Path) -> tuple[str, ...]:
    roles = ("warrior", "forager", "miner", "leader")
    command = [
        sys.executable,
        str(REPO_ROOT / "baselines" / "llm" / "eval_alem.py"),
        f"experiment={PROFILE}",
        f"team.topology={arm}",
        "alem.coordination_difficulty=easy",
        f"eval.resume_from={arm_dir}",
    ]
    command.extend(
        f"clients.{index}.generate_kwargs.prompt_cache_key={_cache_key(arm, role)}"
        for index, role in enumerate(roles)
    )
    return tuple(command)


def _episode_marker(root: Path, arm: str) -> Path:
    return root / arm / "easy" / "alem" / "default" / "default_run_00.json"


def _manifest(config, root: Path) -> dict[str, object]:
    source_commit = _git_output("rev-parse", "HEAD")
    return {
        "schema_version": "alem-dice-team-leader-study-v1",
        "profile": PROFILE,
        "arms": list(ARMS),
        "arm_order": list(ARMS),
        "arm_definitions": {
            "baseline": "three workers; direct peer broadcasts; no leader",
            "leader_peer": "bodyless leader plus direct peer broadcasts",
            "leader_no_peer": "bodyless leader; worker reports delivered only to leader",
        },
        "physical_workers": 3,
        "leader_logical_id": 3,
        "seed": 9999,
        "difficulty": "easy",
        "max_steps_per_episode": 200,
        "environment_max_timesteps": 10000,
        "leader_replan_interval": 5,
        "logical_call_cap": LOGICAL_CALL_CAP,
        "model": "gpt-5.4-2026-03-05",
        "reasoning_effort": "high",
        "max_output_tokens": 8192,
        "non_specialist_efficiency": 0.70,
        "source_commit": source_commit,
        "source_branch": _git_output("branch", "--show-current"),
        "uv_lock_sha256": _lock_sha256(),
        "resolved_config_sha256": _config_sha256(config),
        "cache_keys": {
            arm: {
                role: _cache_key(arm, role)
                for role in ("warrior", "forager", "miner", "leader")
            }
            for arm in ARMS
        },
        "pricing_usd_per_million": {
            "gpt-5.4": {"uncached_input": 2.50, "cached_input": 0.25, "output": 15.00},
            "gpt-5.6-luna": {"uncached_input": 1.00, "cached_input": 0.10, "output": 6.00},
        },
        "analysis_version": 1,
        "output_root": str(root),
    }


def _validate_resume(expected: dict[str, object], root: Path) -> None:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise StudyLaunchError(f"Resume root is missing {MANIFEST_NAME}: {root}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "schema_version",
        "profile",
        "arms",
        "seed",
        "max_steps_per_episode",
        "environment_max_timesteps",
        "leader_replan_interval",
        "model",
        "reasoning_effort",
        "source_commit",
        "uv_lock_sha256",
        "resolved_config_sha256",
        "cache_keys",
    )
    mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
    if mismatches:
        raise StudyLaunchError("Resume manifest mismatch: " + ", ".join(mismatches))


def _summarize(root: Path, results_dir: Path) -> int:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "summarize_team_leader_study.py"),
        str(root),
        "--results-dir",
        str(results_dir),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _run_preflight(config) -> int:
    """Verify GPT-5.4 request compatibility with exactly two provider calls."""

    from omegaconf import OmegaConf

    from baselines.llm.eval_utils.agents.robust_all import _extract_safe_action
    from baselines.llm.eval_utils.client import create_llm_client
    from baselines.llm.eval_utils.prompt_builder import Message
    from baselines.llm.eval_utils.team_leader import TeamLeaderAgent, fallback_plan

    def configured_client(index: int, role: str):
        payload = OmegaConf.to_container(config.clients[index], resolve=True)
        payload["generate_kwargs"]["prompt_cache_key"] = _cache_key("preflight", role)
        return OmegaConf.create(payload)

    worker = create_llm_client(configured_client(0, "worker"))()
    worker_response = worker.generate(
        [
            Message(
                role="system",
                content=(
                    "Alem transport compatibility check. Return exactly one valid tagged action "
                    "and no other text: <action>Noop</action>."
                ),
            ),
            Message(role="user", content="Choose the Noop action now."),
        ]
    )
    if _extract_safe_action(worker_response) is None:
        raise StudyLaunchError("Worker preflight returned no parseable action")

    leader_factory = create_llm_client(configured_client(3, "leader"))
    leader = TeamLeaderAgent(leader_factory)
    leader.set_instruction_prompt("<game_rules>Survive and cooperate.</game_rules>")
    leader_response, plan = leader.plan(
        step=0,
        observations=[
            {"obs_long_term": f"Agent {idx} is safe.", "obs_short_term": "No inventory."}
            for idx in range(3)
        ],
        reports={0: [], 1: [], 2: []},
        previous_plan=fallback_plan(),
        previous_version=0,
    )
    if plan is None:
        raise StudyLaunchError("Leader preflight returned no valid team plan")

    responses = (worker_response, leader_response)
    input_tokens = sum(int(response.input_tokens or 0) for response in responses)
    cached_tokens = sum(int(response.cached_tokens or 0) for response in responses)
    output_tokens = sum(int(response.output_tokens or 0) for response in responses)
    reasoning_tokens = sum(int(response.reasoning_tokens or 0) for response in responses)
    cost = (
        max(input_tokens - cached_tokens, 0) * 2.50
        + cached_tokens * 0.25
        + output_tokens * 15.00
    ) / 1_000_000
    print("Preflight complete: 2 logical calls")
    print(
        f"Tokens: input={input_tokens:,}, cached={cached_tokens:,}, "
        f"output={output_tokens:,}, reasoning={reasoning_tokens:,}"
    )
    print(f"Estimated GPT-5.4 cost: ${cost:.4f}")
    return 0


def run_study(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.summarize:
        return _summarize(args.summarize.resolve(), args.results_dir.resolve())

    config = compose_experiment(PROFILE)
    root = (
        args.resume.resolve()
        if args.resume
        else args.output_root.resolve()
        if args.output_root
        else _default_root(config)
    )
    manifest = _manifest(config, root)
    print(f"Profile: {PROFILE}")
    print(f"Output root: {root}")
    print(f"Source commit: {manifest['source_commit'] or 'unknown'}")
    print(f"Nominal logical-call cap: {LOGICAL_CALL_CAP:,} (excluding transport retries)")
    print("Maximum concurrent provider calls: 3")
    for arm in ARMS:
        complete = _episode_marker_is_complete(_episode_marker(root, arm))
        print(f"  {arm}: {'complete' if complete else 'pending'}")
        print("    " + shlex.join(_command(arm, root / arm / "easy")))
    if args.dry_run:
        print("Dry run complete; no files or API calls were made.")
        return 0

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise StudyLaunchError("OPENAI_API_KEY is required")
    status = _git_status_lines()
    if status:
        raise StudyLaunchError(
            "Paid study requires a clean Git worktree:\n  " + "\n  ".join(status)
        )
    if not manifest["source_commit"] or not manifest["uv_lock_sha256"]:
        raise StudyLaunchError("Paid study requires a Git commit and uv.lock")

    if args.preflight:
        return _run_preflight(config)

    if args.resume:
        _validate_resume(manifest, root)
    else:
        if root.exists() and any(root.iterdir()):
            raise StudyLaunchError(f"New output root is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        manifest["created_at_utc"] = datetime.now(UTC).isoformat()
        (root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from omegaconf import OmegaConf

        OmegaConf.save(config=config, f=str(root / RESOLVED_CONFIG_NAME), resolve=True)

    for arm in ARMS:
        if _episode_marker_is_complete(_episode_marker(root, arm)):
            print(f"Skipping {arm}: complete.")
            continue
        arm_dir = root / arm / "easy"
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {arm}; output: {arm_dir}", flush=True)
        subprocess.run(_command(arm, arm_dir), cwd=REPO_ROOT, check=True)

    result = _summarize(root, args.results_dir.resolve())
    if result == 0:
        print(f"Study complete: {root}")
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(run_study())
    except StudyLaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
