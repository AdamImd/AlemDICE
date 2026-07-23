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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "baselines/llm/eval_utils/agents/__init__.py",
    "baselines/llm/eval_utils/agents/robust_all.py",
    "baselines/llm/eval_utils/evaluator.py",
    "baselines/llm/eval_utils/team_commander.py",
)


class CommanderStudyError(RuntimeError):
    """Raised before or during an unsafe/inconsistent study launch."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("30", "100", "200"), default="30")
    parser.add_argument(
        "--model-regime",
        choices=("luna-luna", "nano-luna"),
        default="luna-luna",
    )
    parser.add_argument(
        "--action-reasoning-efforts",
        nargs="+",
        choices=("none", "high"),
        default=None,
    )
    parser.add_argument("--parallel-arms", type=int, default=1)
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


def _profile(stage: str, model_regime: str = "luna-luna") -> str:
    if model_regime == "nano-luna":
        if stage != "100":
            raise CommanderStudyError("nano-luna currently supports only --stage 100")
        return "embodied_commander_nano_luna_100"
    return f"embodied_commander_{stage}"


def _arm_topology(arm: str) -> str:
    for topology in (*ARMS, OPTIONAL_STAR_ARM):
        if arm == topology or arm.endswith(f"__{topology}"):
            return topology
    raise CommanderStudyError(f"Unknown study arm: {arm}")


def _arm_effort(arm: str) -> str | None:
    if arm.startswith("nano_none__"):
        return "none"
    if arm.startswith("nano_high__"):
        return "high"
    return None


def _arms(
    include_star: bool,
    model_regime: str = "luna-luna",
    efforts: tuple[str, ...] = ("none",),
) -> tuple[str, ...]:
    topologies = ARMS + ((OPTIONAL_STAR_ARM,) if include_star else ())
    if model_regime == "luna-luna":
        return topologies
    return tuple(f"nano_{effort}__{topology}" for effort in efforts for topology in topologies)


def _default_root(config, stage: str, model_regime: str = "luna-luna") -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = Path(str(config.eval.output_dir))
    if not base.is_absolute():
        base = REPO_ROOT / base
    label = (
        f"nano_luna_2x2_{stage}"
        if model_regime == "nano-luna"
        else f"embodied_commander_{stage}"
    )
    return (base / f"{timestamp}_{label}").resolve()


def _cache_key(stage: str, arm: str, role: str) -> str:
    topology = _arm_topology(arm) if arm != "preflight" else "preflight"
    arm_code = {
        "baseline": "src",
        "embodied_commander_broadcast": "cmd",
        "embodied_commander_star": "star",
        "preflight": "pf",
    }[topology]
    role_code = {"warrior": "w", "forager": "f", "miner": "m"}[role]
    effort = _arm_effort(arm)
    model_code = "g54n" if effort is not None else "g56"
    effort_code = {"none": "n", "high": "h", None: "x"}[effort]
    return f"alem:{model_code}:sq2:{stage}:{effort_code}:{arm_code}:{role_code}"


def _planner_cache_key(stage: str, arm: str) -> str:
    effort = _arm_effort(arm) or "x"
    topology = _arm_topology(arm)
    topology_code = "cmd" if topology == "embodied_commander_broadcast" else "star"
    return f"alem:g56l:sq2:{stage}:{effort[0]}:{topology_code}:p"


def _arm_overrides(stage: str, arm: str, arm_dir: Path) -> tuple[str, ...]:
    roles = ("warrior", "forager", "miner")
    topology = _arm_topology(arm)
    overrides = [
        f"team.topology={topology}",
        "team.commander_agent_id=0",
        "alem.coordination_difficulty=easy",
        f"eval.resume_from={arm_dir}",
        "WANDB_MODE=disabled",
    ]
    overrides.extend(
        f"clients.{index}.generate_kwargs.prompt_cache_key={_cache_key(stage, arm, role)}"
        for index, role in enumerate(roles)
    )
    effort = _arm_effort(arm)
    if effort is not None:
        overrides.extend(
            f"clients.{index}.generate_kwargs.reasoning_effort={effort}"
            for index in range(3)
        )
        overrides.append(
            "team.commander_planner_client.generate_kwargs.reasoning_effort=high"
        )
        if topology != "baseline":
            overrides.append(
                "team.commander_planner_client.generate_kwargs.prompt_cache_key="
                + _planner_cache_key(stage, arm)
            )
    return tuple(overrides)


def _command(
    stage: str,
    arm: str,
    arm_dir: Path,
    model_regime: str = "luna-luna",
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(REPO_ROOT / "baselines" / "llm" / "eval_alem.py"),
        f"experiment={_profile(stage, model_regime)}",
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


def _arm_config_hash(
    stage: str,
    arm: str,
    root: Path,
    model_regime: str = "luna-luna",
) -> str:
    config = compose_experiment(
        _profile(stage, model_regime),
        overrides=_arm_overrides(stage, arm, root / arm / "easy"),
    )
    return _config_sha256(config)


def _manifest(
    config,
    root: Path,
    stage: str,
    arms: tuple[str, ...],
    *,
    model_regime: str = "luna-luna",
    parallel_arms: int = 1,
) -> dict[str, object]:
    spec = validate_experiment_config(config)
    source_commit = _git_output("rev-parse", "HEAD")
    scheduled_calls = (
        spec.max_steps_per_episode + int(config.team.commander_review_interval) - 1
    ) // int(config.team.commander_review_interval)
    event_calls = (spec.max_steps_per_episode + 9) // 10
    treatment_plan_cap = len(spec.seeds) * (scheduled_calls + event_calls)
    action_cap_per_arm = len(spec.seeds) * spec.max_steps_per_episode * spec.num_agents
    treatment_arm_count = sum(_arm_topology(arm) != "baseline" for arm in arms)
    arm_configs = {
        arm: {
            "topology": _arm_topology(arm),
            "worker_model": spec.model_ids[0],
            "worker_reasoning_effort": (
                _arm_effort(arm)
                or str(config.clients[0].generate_kwargs.reasoning_effort)
            ),
            "commander_planner_model": (
                spec.commander_planner_model_id
                if _arm_topology(arm) != "baseline"
                else None
            ),
            "commander_planner_reasoning_effort": (
                spec.commander_planner_reasoning_effort
                if _arm_topology(arm) != "baseline"
                else None
            ),
        }
        for arm in arms
    }
    return {
        "schema_version": (
            "alem-dice-embodied-commander-study-v2"
            if model_regime == "nano-luna"
            else "alem-dice-embodied-commander-study-v1"
        ),
        "stage": stage,
        "profile": _profile(stage, model_regime),
        "model_regime": model_regime,
        "arms": list(arms),
        "arm_order": list(arms),
        "arm_configs": arm_configs,
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
            + treatment_plan_cap * treatment_arm_count
        ),
        "logical_call_cap_by_phase_model": {
            f"decision:{spec.model_ids[0]}": action_cap_per_arm * len(arms),
            f"commander_plan:{spec.commander_planner_model_id or spec.model_ids[0]}": (
                treatment_plan_cap * treatment_arm_count
            ),
        },
        "parallel_arms": parallel_arms,
        "maximum_parallel_provider_calls": (
            spec.num_agents * spec.num_workers * min(parallel_arms, len(arms))
        ),
        "worker_models": list(spec.model_ids),
        "commander_planner_model": spec.commander_planner_model_id,
        "api": "openai_responses",
        "action_reasoning_efforts": sorted(
            {
                value["worker_reasoning_effort"]
                for value in arm_configs.values()
            }
        ),
        "commander_planner_reasoning_effort": spec.commander_planner_reasoning_effort,
        "max_output_tokens": 8192,
        "source_commit": source_commit,
        "source_branch": _git_output("branch", "--show-current"),
        "git_status": list(_git_status_lines()),
        "uv_lock_sha256": _lock_sha256(),
        "resolved_base_config_sha256": _config_sha256(config),
        "resolved_arm_config_sha256": {
            arm: _arm_config_hash(stage, arm, root, model_regime) for arm in arms
        },
        "prompt_contract_sha256": _sha256_files(PROMPT_CONTRACT_FILES),
        "prompt_contract_files": list(PROMPT_CONTRACT_FILES),
        "cache_keys": {
            arm: {
                "workers": {
                    role: _cache_key(stage, arm, role)
                    for role in ("warrior", "forager", "miner")
                },
                "planner": (
                    _planner_cache_key(stage, arm)
                    if _arm_topology(arm) != "baseline"
                    and _arm_effort(arm) is not None
                    else None
                ),
            }
            for arm in arms
        },
        "commands": {
            arm: list(
                _command(
                    stage,
                    arm,
                    root / arm / "easy",
                    model_regime,
                )
            )
            for arm in arms
        },
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
        "model_regime",
        "arms",
        "arm_configs",
        "parallel_arms",
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
    if any(root.rglob("*_commander_calls.jsonl")):
        audit_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_commander_calls.py"),
            str(root),
        ]
        audit_result = subprocess.run(
            audit_command,
            cwd=REPO_ROOT,
            check=False,
        )
        if audit_result.returncode != 0:
            return audit_result.returncode
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "summarize_embodied_commander_study.py"),
        str(root),
        "--results-dir",
        str(results_dir),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _run_preflight(
    config,
    stage: str,
    *,
    model_regime: str = "luna-luna",
    efforts: tuple[str, ...] = ("none",),
) -> int:
    """Validate each action effort and the independent planner route."""

    from omegaconf import OmegaConf

    from baselines.llm.eval_utils.agents.robust_all import _extract_safe_action
    from baselines.llm.eval_utils.client import create_llm_client
    from baselines.llm.eval_utils.prompt_builder import Message
    from baselines.llm.eval_utils.team_commander import (
        EmbodiedCommanderPlanner,
        ReviewRequest,
        SquadSpec,
    )

    def configured_worker(effort: str, max_tokens: int):
        payload = OmegaConf.to_container(config.clients[0], resolve=True)
        payload["generate_kwargs"]["reasoning_effort"] = effort
        payload["generate_kwargs"]["prompt_cache_key"] = (
            f"alem:g54n:sq2:{stage}:{effort[0]}:pf:w"
            if model_regime == "nano-luna"
            else f"alem:g56:sq2:{stage}:x:pf:w"
        )
        payload["generate_kwargs"]["max_tokens"] = max_tokens
        return OmegaConf.create(payload)

    worker_responses = []
    for effort in efforts:
        worker = create_llm_client(configured_worker(effort, 8192))()
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
            raise CommanderStudyError(
                f"Action preflight at reasoning_effort={effort} returned no parseable action"
            )
        worker_responses.append(worker_response)

    spec = SquadSpec(
        team_id="preflight",
        members=(0, 1, 2),
        commander_id=0,
        max_steps=int(config.eval.max_steps_per_episode),
    )
    planner_payload = OmegaConf.to_container(
        config.team.get("commander_planner_client") or config.clients[0],
        resolve=True,
    )
    planner_payload["generate_kwargs"]["reasoning_effort"] = "high"
    planner_payload["generate_kwargs"]["prompt_cache_key"] = (
        f"alem:g56l:sq2:{stage}:h:pf:p"
    )
    planner_payload["generate_kwargs"]["max_tokens"] = 8192
    planner = EmbodiedCommanderPlanner(
        create_llm_client(OmegaConf.create(planner_payload)),
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
        raise CommanderStudyError(
            "Luna planning preflight returned no valid REPLACE plan. Raw output: "
            + repr(planner.last_raw_completion)
        )

    responses = (*worker_responses, planner_response)
    print(
        f"Preflight complete: exactly {len(responses)} logical calls "
        f"({len(worker_responses)} action, 1 planner)"
    )
    print(
        f"Tokens: input={sum(int(response.input_tokens or 0) for response in responses):,}, cached={sum(int(response.cached_tokens or 0) for response in responses):,}, output={sum(int(response.output_tokens or 0) for response in responses):,}, reasoning={sum(int(response.reasoning_tokens or 0) for response in responses):,}"
    )
    return 0


def run_study(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.summarize:
        return _summarize(args.summarize.resolve(), args.results_dir.resolve())

    stage = args.stage
    if args.parallel_arms < 1:
        raise CommanderStudyError("--parallel-arms must be positive")
    model_regime = args.model_regime
    efforts = tuple(args.action_reasoning_efforts or ("none",))
    if model_regime == "luna-luna" and args.action_reasoning_efforts:
        raise CommanderStudyError(
            "--action-reasoning-efforts is only supported with --model-regime nano-luna"
        )
    profile = _profile(stage, model_regime)
    config = compose_experiment(profile)
    arms = _arms(args.include_star, model_regime, efforts)
    root = (
        args.resume.resolve()
        if args.resume
        else args.output_root.resolve()
        if args.output_root
        else _default_root(config, stage, model_regime)
    )
    manifest = _manifest(
        config,
        root,
        stage,
        arms,
        model_regime=model_regime,
        parallel_arms=args.parallel_arms,
    )
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
        print(
            "    "
            + shlex.join(
                _command(
                    stage,
                    arm,
                    root / arm / "easy",
                    model_regime,
                )
            )
        )
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
        return _run_preflight(
            config,
            stage,
            model_regime=model_regime,
            efforts=efforts,
        )

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

    pending_arms = []
    for arm in arms:
        markers = _episode_markers(root, arm, episodes)
        if all(_episode_marker_is_complete(marker) for marker in markers):
            print(f"Skipping {arm}: complete.")
            continue
        arm_dir = root / arm / "easy"
        arm_dir.mkdir(parents=True, exist_ok=True)
        pending_arms.append((arm, arm_dir))

    def run_arm(arm: str, arm_dir: Path) -> tuple[str, int, Path]:
        log_path = arm_dir.parent / "launcher.log"
        print(f"Running {arm}; output: {arm_dir}; log: {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                _command(stage, arm, arm_dir, model_regime),
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return arm, result.returncode, log_path

    failures = []
    worker_count = min(args.parallel_arms, len(pending_arms)) if pending_arms else 0
    if worker_count:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_arm, arm, arm_dir): arm
                for arm, arm_dir in pending_arms
            }
            for future in as_completed(futures):
                arm, returncode, log_path = future.result()
                print(f"Finished {arm}: exit={returncode}; log={log_path}", flush=True)
                if returncode != 0:
                    failures.append((arm, returncode, log_path))
    if failures:
        details = ", ".join(
            f"{arm} (exit {returncode}, {log_path})"
            for arm, returncode, log_path in failures
        )
        raise CommanderStudyError(
            "One or more arms failed after all parallel arms settled: "
            + details
            + f". Resume with --resume {root}"
        )

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
