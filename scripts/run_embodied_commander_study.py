#!/usr/bin/env python3
"""Run the staged, matched Luna Source-vs-embodied-commander study."""

from __future__ import annotations

import argparse
import hashlib
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

from baselines.llm.experiment_config import (  # noqa: E402
    compose_experiment,
    validate_experiment_config,
)
from scripts.run_openai_matrix import (  # noqa: E402
    _config_sha256,
    _episode_marker_is_complete,
    _git_output,
    _git_status_lines,
    _lock_sha256,
)

ARMS = ("baseline", "embodied_commander_broadcast")
OPTIONAL_STAR_ARM = "embodied_commander_star"
MANIFEST_NAME = "commander_study_manifest.json"
RESOLVED_CONFIG_NAME = "resolved_base_config.yaml"
PROMPT_CONTRACT_FILES = (
    "baselines/llm/eval_utils/agents/robust_all.py",
    "baselines/llm/eval_utils/evaluator.py",
    "baselines/llm/eval_utils/team_commander.py",
)


class CommanderStudyError(RuntimeError):
    """Raised before or during an unsafe/inconsistent study launch."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("30", "100", "200"), default="30")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--resume", type=Path, metavar="RUN_DIR")
    output.add_argument("--output-root", type=Path, metavar="RUN_DIR")
    output.add_argument("--summarize", type=Path, metavar="RUN_DIR")
    parser.add_argument(
        "--include-star",
        action="store_true",
        help="Append the commander-star routing ablation after the preregistered arms.",
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Make exactly one action and one planning request, then exit.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a dirty worktree while recording its status in the manifest.",
    )
    return parser


def _profile(stage: str) -> str:
    return f"embodied_commander_{stage}"


def _arms(include_star: bool) -> tuple[str, ...]:
    return ARMS + ((OPTIONAL_STAR_ARM,) if include_star else ())


def _default_root(config, stage: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = Path(str(config.eval.output_dir))
    if not base.is_absolute():
        base = REPO_ROOT / base
    return (base / f"{timestamp}_embodied_commander_{stage}").resolve()


def _cache_key(stage: str, arm: str, role: str) -> str:
    arm_code = {
        "baseline": "src",
        "embodied_commander_broadcast": "cmd",
        "embodied_commander_star": "star",
        "preflight": "pf",
    }[arm]
    role_code = {"warrior": "w", "forager": "f", "miner": "m"}[role]
    return f"alem:g56:b1344e4:sq1:{stage}:{arm_code}:{role_code}"


def _arm_overrides(stage: str, arm: str, arm_dir: Path) -> tuple[str, ...]:
    roles = ("warrior", "forager", "miner")
    overrides = [
        f"team.topology={arm}",
        "team.commander_agent_id=0",
        "alem.coordination_difficulty=easy",
        f"eval.resume_from={arm_dir}",
        "WANDB_MODE=disabled",
    ]
    overrides.extend(
        f"clients.{index}.generate_kwargs.prompt_cache_key={_cache_key(stage, arm, role)}"
        for index, role in enumerate(roles)
    )
    return tuple(overrides)


def _command(stage: str, arm: str, arm_dir: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        str(REPO_ROOT / "baselines" / "llm" / "eval_alem.py"),
        f"experiment={_profile(stage)}",
        *_arm_overrides(stage, arm, arm_dir),
    )


def _episode_markers(root: Path, arm: str, episodes: int) -> tuple[Path, ...]:
    base = root / arm / "easy" / "alem" / "default"
    return tuple(base / f"default_run_{index:02d}.json" for index in range(episodes))


def _sha256_files(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _arm_config_hash(stage: str, arm: str, root: Path) -> str:
    config = compose_experiment(
        _profile(stage),
        overrides=_arm_overrides(stage, arm, root / arm / "easy"),
    )
    return _config_sha256(config)


def _manifest(config, root: Path, stage: str, arms: tuple[str, ...]) -> dict[str, object]:
    spec = validate_experiment_config(config)
    source_commit = _git_output("rev-parse", "HEAD")
    scheduled_calls = (
        spec.max_steps_per_episode + int(config.team.commander_review_interval) - 1
    ) // int(config.team.commander_review_interval)
    event_calls = (spec.max_steps_per_episode + 9) // 10
    treatment_plan_cap = len(spec.seeds) * (scheduled_calls + event_calls)
    action_cap_per_arm = len(spec.seeds) * spec.max_steps_per_episode * spec.num_agents
    return {
        "schema_version": "alem-dice-embodied-commander-study-v1",
        "stage": stage,
        "profile": _profile(stage),
        "arms": list(arms),
        "arm_order": list(arms),
        "arm_definitions": {
            "baseline": "unchanged Source protocol with one-tick peer broadcast",
            "embodied_commander_broadcast": (
                "Agent 0 serial plan phase; leased plan delivered before three parallel "
                "action calls; SCP1 statuses also peer-broadcast"
            ),
            "embodied_commander_star": (
                "same plans and timing; SCP1 statuses delivered only to Agent 0"
            ),
        },
        "seeds": list(spec.seeds),
        "difficulty": "easy",
        "max_steps_per_episode": spec.max_steps_per_episode,
        "physical_agents": spec.num_agents,
        "logical_participants": spec.num_agents,
        "commander_agent_id": 0,
        "review_interval": int(config.team.commander_review_interval),
        "lease_steps": int(config.team.commander_lease_steps),
        "event_replan_cap_per_episode": event_calls,
        "action_call_cap_per_arm": action_cap_per_arm,
        "commander_plan_call_cap_per_treatment_arm": treatment_plan_cap,
        "logical_call_cap": (
            action_cap_per_arm * len(arms)
            + treatment_plan_cap * sum(arm != "baseline" for arm in arms)
        ),
        "maximum_parallel_provider_calls": spec.num_agents * spec.num_workers,
        "model": "gpt-5.6-luna",
        "api": "openai_responses",
        "reasoning_effort": "none",
        "max_output_tokens": 8192,
        "source_commit": source_commit,
        "source_branch": _git_output("branch", "--show-current"),
        "git_status": list(_git_status_lines()),
        "uv_lock_sha256": _lock_sha256(),
        "resolved_base_config_sha256": _config_sha256(config),
        "resolved_arm_config_sha256": {arm: _arm_config_hash(stage, arm, root) for arm in arms},
        "prompt_contract_sha256": _sha256_files(PROMPT_CONTRACT_FILES),
        "prompt_contract_files": list(PROMPT_CONTRACT_FILES),
        "cache_keys": {
            arm: {role: _cache_key(stage, arm, role) for role in ("warrior", "forager", "miner")}
            for arm in arms
        },
        "commands": {arm: list(_command(stage, arm, root / arm / "easy")) for arm in arms},
        "output_root": str(root),
        "preregistration": {
            "stage_30": {
                "complete_no_transport_errors": True,
                "valid_plan_calls_min": 1,
                "active_plan_coverage_min": 0.80,
                "action_parse_rate_min": 0.95,
                "status_parse_rate_min": 0.90,
                "accepted_authority_or_stale_or_leak_violations_max": 0,
            },
            "stage_100": {
                "complete_no_transport_errors": True,
                "valid_plan_and_status_rate_min": 0.90,
                "valid_status_coverage_min": 0.80,
                "active_plan_coverage_min": 0.90,
                "action_parse_rate_min": 0.95,
                "accepted_authority_or_stale_or_leak_violations_max": 0,
            },
            "stage_200": {
                "complete_pairs": 3,
                "achievement_improvement_min_pairs": 2,
                "action_parse_drop_over_2pp_max_pairs": 1,
                "plan_and_status_validity_min": 0.90,
                "active_plan_coverage_min": 0.90,
                "accepted_authority_or_stale_violations_max": 0,
                "mean_token_or_wall_regression_max": 0.35,
            },
        },
    }


def _validate_resume(expected: dict[str, object], root: Path) -> None:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise CommanderStudyError(f"Resume root is missing {MANIFEST_NAME}: {root}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "schema_version",
        "stage",
        "profile",
        "arms",
        "seeds",
        "max_steps_per_episode",
        "source_commit",
        "uv_lock_sha256",
        "resolved_base_config_sha256",
        "resolved_arm_config_sha256",
        "prompt_contract_sha256",
        "cache_keys",
    )
    mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
    if mismatches:
        raise CommanderStudyError("Resume manifest mismatch: " + ", ".join(mismatches))


def _summarize(root: Path, results_dir: Path) -> int:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "summarize_embodied_commander_study.py"),
        str(root),
        "--results-dir",
        str(results_dir),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _run_preflight(config, stage: str) -> int:
    """Make exactly two bounded Luna calls: one action and one squad plan."""

    from omegaconf import OmegaConf

    from baselines.llm.eval_utils.agents.robust_all import _extract_safe_action
    from baselines.llm.eval_utils.client import create_llm_client
    from baselines.llm.eval_utils.prompt_builder import Message
    from baselines.llm.eval_utils.team_commander import (
        EmbodiedCommanderPlanner,
        ReviewRequest,
        SquadSpec,
    )

    def configured_client(index: int, role: str, max_tokens: int):
        payload = OmegaConf.to_container(config.clients[index], resolve=True)
        payload["generate_kwargs"]["prompt_cache_key"] = _cache_key(stage, "preflight", role)
        payload["generate_kwargs"]["max_tokens"] = max_tokens
        return OmegaConf.create(payload)

    worker = create_llm_client(configured_client(0, "warrior", 64))()
    worker_response = worker.generate(
        [
            Message(
                role="system",
                content="Return exactly <action>Noop</action> and no other text.",
            ),
            Message(role="user", content="Choose Noop now."),
        ]
    )
    if _extract_safe_action(worker_response) is None:
        raise CommanderStudyError("Luna action preflight returned no parseable action")

    spec = SquadSpec(
        team_id="preflight",
        members=(0, 1, 2),
        commander_id=0,
        max_steps=int(config.eval.max_steps_per_episode),
    )
    planner = EmbodiedCommanderPlanner(
        create_llm_client(configured_client(0, "warrior", 1024)),
        spec=spec,
    )
    planner.set_instruction_prompt("<game_rules>Survive and cooperate.</game_rules>")
    planner_response, proposal = planner.plan(
        step=0,
        review=ReviewRequest("scheduled", True),
        commander_observation={
            "obs_long_term": "All three agents are safe at the starting area.",
            "obs_short_term": "No immediate threats are visible.",
        },
        reports=[],
        active_plan=None,
        canonical_actions={"Noop", "Move North", "Move South", "Move East", "Move West"},
    )
    if proposal is None or proposal.operation != "REPLACE":
        raise CommanderStudyError("Luna planning preflight returned no valid REPLACE plan")

    responses = (worker_response, planner_response)
    print("Preflight complete: exactly 2 logical Luna calls")
    print(
        f"Tokens: input={sum(int(response.input_tokens or 0) for response in responses):,}, cached={sum(int(response.cached_tokens or 0) for response in responses):,}, output={sum(int(response.output_tokens or 0) for response in responses):,}, reasoning={sum(int(response.reasoning_tokens or 0) for response in responses):,}"
    )
    return 0


def run_study(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.summarize:
        return _summarize(args.summarize.resolve(), args.results_dir.resolve())

    stage = args.stage
    profile = _profile(stage)
    config = compose_experiment(profile)
    arms = _arms(args.include_star)
    root = (
        args.resume.resolve()
        if args.resume
        else args.output_root.resolve()
        if args.output_root
        else _default_root(config, stage)
    )
    manifest = _manifest(config, root, stage, arms)
    episodes = len(manifest["seeds"])

    print(f"Profile: {profile}")
    print(f"Output root: {root}")
    print(f"Source commit: {manifest['source_commit'] or 'unknown'}")
    print(
        "Nominal logical-call cap: {:,} (excluding transport retries)".format(
            int(manifest["logical_call_cap"])
        )
    )
    print(f"Maximum concurrent provider calls: {manifest['maximum_parallel_provider_calls']}")
    for arm in arms:
        markers = _episode_markers(root, arm, episodes)
        completed = sum(_episode_marker_is_complete(marker) for marker in markers)
        print(f"  {arm}: {completed}/{episodes} complete")
        print("    " + shlex.join(_command(stage, arm, root / arm / "easy")))
    if args.dry_run:
        print("Dry run complete; no files or API calls were made.")
        return 0

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise CommanderStudyError("OPENAI_API_KEY is required")
    status = _git_status_lines()
    if status and not args.allow_dirty:
        raise CommanderStudyError(
            "Paid study requires a clean Git worktree (or explicit --allow-dirty):\n  "
            + "\n  ".join(status)
        )
    if not manifest["source_commit"] or not manifest["uv_lock_sha256"]:
        raise CommanderStudyError("Paid study requires a Git commit and uv.lock")

    if args.preflight:
        return _run_preflight(config, stage)

    if args.resume:
        _validate_resume(manifest, root)
    else:
        if root.exists() and any(root.iterdir()):
            raise CommanderStudyError(f"New output root is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        manifest["created_at_utc"] = datetime.now(UTC).isoformat()
        (root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from omegaconf import OmegaConf

        OmegaConf.save(
            config=config,
            f=str(root / RESOLVED_CONFIG_NAME),
            resolve=True,
        )

    for arm in arms:
        markers = _episode_markers(root, arm, episodes)
        if all(_episode_marker_is_complete(marker) for marker in markers):
            print(f"Skipping {arm}: complete.")
            continue
        arm_dir = root / arm / "easy"
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running {arm}; output: {arm_dir}", flush=True)
        subprocess.run(_command(stage, arm, arm_dir), cwd=REPO_ROOT, check=True)

    result = _summarize(root, args.results_dir.resolve())
    if result == 0:
        print(f"Study complete: {root}")
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(run_study())
    except CommanderStudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
