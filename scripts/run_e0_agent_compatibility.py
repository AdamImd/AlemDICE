#!/usr/bin/env python3
"""Run the provider-free E0 variable-agent compatibility gate.

E0 deliberately exercises the canonical LLM evaluator and artifact writers with
scripted agents.  It makes no provider requests and is not a behavioral model
evaluation.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alem.alem_coop.constants import Action, Specialization  # noqa: E402
from alem.llm.alem_env import make_env  # noqa: E402
from baselines.llm.eval_utils.client import ModelResponse  # noqa: E402
from baselines.llm.eval_utils.evaluator import EvaluatorManager  # noqa: E402
from baselines.llm.experiment_config import (  # noqa: E402
    compose_experiment,
    validate_experiment_config,
)

SCHEMA_VERSION = "alem-dice-e0-agent-compatibility-v1"
DEFAULT_COUNTS = (1, 2, 3, 4, 6)
DEFAULT_SEEDS = (13000, 13001)
DEFAULT_STEPS = 10
EXPECTED_ROLE_VALUES = (
    Specialization.WARRIOR.value,
    Specialization.FORAGER.value,
    Specialization.MINER.value,
)


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


def _state_projection_hash(state: Any) -> str:
    """Hash stable arrays sufficient to verify deterministic E0 reconstruction."""

    digest = hashlib.sha256()
    for name in (
        "map",
        "item_map",
        "player_position",
        "player_specialization",
        "coordination_map",
    ):
        array = np.asarray(getattr(state, name))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class ScriptedProbeAgent:
    """Minimal deterministic actor used only by the free E0 gate."""

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0
        self.current_communication: str | None = None
        self.communication_history: list[dict[int, str]] = []
        self._last_parse_failed = False
        self._last_comm_failed = False
        self._last_scratchpad_failed = False
        self._last_raw_completion: str | None = None
        self._last_raw_reasoning: str | None = None

    def reset(self) -> None:
        self.step_count = 0
        self.current_communication = None
        self.communication_history = []

    def receive_communication(self, communication: dict[int, str]) -> None:
        self.communication_history.append(dict(communication))

    def act(self, _observation: Any, prev_action: str | None = None) -> ModelResponse:
        del prev_action
        self.current_communication = (
            f"E0|AGENT={self.agent_id}|TICK={self.step_count}|STATE=ACTIVE"
        )
        self._last_raw_completion = (
            f"<action>Noop</action><communication>{self.current_communication}</communication>"
        )
        self.step_count += 1
        return ModelResponse(
            model_id="e0-scripted",
            completion="Noop",
            stop_reason="stop",
            input_tokens=0,
            output_tokens=0,
            reasoning=None,
            reasoning_tokens=0,
            status="completed",
        )


class ScriptedProbeFactory:
    """Factory matching the evaluator's physical-agent interface."""

    def create_agent(self, agent_idx: int | None = None) -> ScriptedProbeAgent:
        if agent_idx is None:
            raise ValueError("E0 requires an explicit physical agent index")
        return ScriptedProbeAgent(agent_idx)

    def create_leader(self) -> None:
        return None

    def create_commander_planner(self, _spec: Any) -> None:
        return None


def build_e0_config(
    *,
    num_agents: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
) -> DictConfig:
    """Compose a validated provider-free config for one E0 population."""

    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    if not seeds or tuple(seeds) != tuple(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError("E0 seeds must be a non-empty contiguous sequence")
    if steps < 1:
        raise ValueError("steps must be positive")

    config = compose_experiment("fake_smoke")
    base_client = OmegaConf.to_container(config.clients[0], resolve=True)
    if not isinstance(base_client, dict):
        raise TypeError("fake_smoke client must resolve to a mapping")
    clients = []
    for agent_id in range(num_agents):
        client = copy.deepcopy(base_client)
        client["generate_kwargs"]["prompt_cache_key"] = (
            f"alem-dice:e0:scripted:n{num_agents}:agent-{agent_id}"
        )
        clients.append(client)

    with open_dict(config):
        config.experiment.seeds = list(seeds)
        config.EVAL_SEED = seeds[0]
        config.eval.num_episodes.alem = len(seeds)
        config.eval.num_workers = 1
        config.eval.max_steps_per_episode = steps
        config.eval.save_images = False
        config.eval.save_images_every_episode = False
        config.eval.debug = False
        config.eval.generate_debriefs = False
        config.agent.use_communication = True
        config.agent.use_scratchpad = False
        config.agent.use_cot = False
        config.alem.num_agents = num_agents
        config.alem.max_timesteps = steps
        config.team.topology = "baseline"
        config.clients = OmegaConf.create(clients)
        config.WANDB_MODE = "disabled"

    validate_experiment_config(config, expected_profile="fake_smoke")
    return config


def _load_episode(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Episode artifact is not an object: {path}")
    return value


def _expected_roles(num_agents: int) -> list[int]:
    return [EXPECTED_ROLE_VALUES[index % len(EXPECTED_ROLE_VALUES)] for index in range(num_agents)]


def _validate_give_targets(env: Any, num_agents: int) -> list[dict[str, int]]:
    mappings = []
    for giver in range(num_agents):
        for target in range(num_agents):
            if giver == target:
                continue
            label = f"Give to Agent {target}"
            actual = int(env.wrapper.get_action_index(label, giver))
            slot = target if target < giver else target - 1
            expected = int(Action.GIVE.value + slot)
            if actual != expected:
                raise AssertionError(
                    f"N={num_agents} giver={giver} target={target}: "
                    f"expected action {expected}, got {actual}"
                )
            canonical = env.check_action_validity(label, giver)
            if canonical != label:
                raise AssertionError(
                    f"N={num_agents} giver={giver} target={target}: "
                    f"canonicalized to {canonical!r}"
                )
            mappings.append(
                {
                    "giver": giver,
                    "target": target,
                    "action_index": actual,
                }
            )
    return mappings


@dataclass(frozen=True)
class PopulationResult:
    num_agents: int
    passed: bool
    episodes: tuple[dict[str, Any], ...]
    observation_shape: tuple[int, ...]
    action_space_size: int
    give_mapping_count: int
    role_values: tuple[int, ...]
    expected_role_values: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "passed": self.passed,
            "episodes": list(self.episodes),
            "observation_shape": list(self.observation_shape),
            "action_space_size": self.action_space_size,
            "give_mapping_count": self.give_mapping_count,
            "role_values": list(self.role_values),
            "expected_role_values": list(self.expected_role_values),
        }


def validate_population(
    *,
    output_dir: Path,
    config: DictConfig,
    num_agents: int,
    seeds: tuple[int, ...],
    steps: int,
) -> PopulationResult:
    """Validate artifacts and reconstruct each seed from the saved state bundle."""

    task_dir = output_dir / "alem" / "default"
    probe_env = make_env("alem", "default", config)
    probe_env.reset(seed=seeds[0])
    initial_observations = probe_env.env.get_obs(probe_env.state)
    observation_shape = tuple(np.asarray(initial_observations["agent_0"]).shape)
    action_space_size = int(probe_env.env.action_space("agent_0").n)
    give_mappings = _validate_give_targets(probe_env, num_agents)

    episode_records: list[dict[str, Any]] = []
    observed_roles: tuple[int, ...] | None = None
    for episode_index, seed in enumerate(seeds):
        prefix = f"default_run_{episode_index:02d}"
        episode_path = task_dir / f"{prefix}.json"
        trajectory_path = task_dir / f"{prefix}_trajectory.npz"
        states_path = task_dir / f"{prefix}_states.pkl.gz"
        debug_path = task_dir / f"{prefix}_debug.jsonl"
        for required in (episode_path, trajectory_path, states_path, debug_path):
            if not required.is_file():
                raise AssertionError(f"Missing E0 artifact: {required}")

        episode = _load_episode(episode_path)
        if episode.get("artifact_status") != "complete" or episode.get("error"):
            raise AssertionError(f"Incomplete E0 episode: {episode_path}")
        if int(episode.get("num_steps", -1)) != steps:
            raise AssertionError(f"Unexpected E0 step count: {episode_path}")
        if int(episode.get("num_agents", -1)) != num_agents:
            raise AssertionError(f"Unexpected E0 agent count: {episode_path}")
        if int(episode.get("model_call_count", -1)) != 0:
            raise AssertionError(f"E0 made a model call: {episode_path}")
        if int(episode.get("provider_request_count", -1)) != 0:
            raise AssertionError(f"E0 made a provider request: {episode_path}")
        if int(episode.get("transport_error_count", -1)) != 0:
            raise AssertionError(f"E0 recorded a transport error: {episode_path}")
        if float(episode.get("action_parse_rate", 0.0)) != 1.0:
            raise AssertionError(f"E0 action parse rate is not 1: {episode_path}")

        worker_peer = episode.get("communication_metrics", {}).get("worker_peer", {})
        expected_emitted = num_agents * steps
        expected_delivered = num_agents * max(num_agents - 1, 0) * steps
        if int(worker_peer.get("emitted_messages", -1)) != expected_emitted:
            raise AssertionError(f"Incorrect emitted-message accounting: {episode_path}")
        if int(worker_peer.get("delivered_messages", -1)) != expected_delivered:
            raise AssertionError(f"Incorrect delivered-message accounting: {episode_path}")

        with np.load(trajectory_path, allow_pickle=True) as trajectory:
            for key in ("obs", "actions", "rewards", "text_obs", "text_actions"):
                if trajectory[key].shape[0] != steps:
                    raise AssertionError(f"{trajectory_path}: {key} has wrong horizon")
                if trajectory[key].shape[1] != num_agents:
                    raise AssertionError(f"{trajectory_path}: {key} has wrong agent axis")
            if tuple(trajectory["obs"].shape[2:]) != observation_shape:
                raise AssertionError(f"{trajectory_path}: observation dimensions drifted")

        with gzip.open(states_path, "rb") as handle:
            state_payload = pickle.load(handle)
        saved_states = state_payload.get("states", [])
        if len(saved_states) != steps:
            raise AssertionError(f"{states_path}: wrong saved-state count")
        saved_initial_state = saved_states[0]
        roles = tuple(int(value) for value in np.asarray(saved_initial_state.player_specialization))
        if observed_roles is None:
            observed_roles = roles
        if roles != tuple(_expected_roles(num_agents)):
            raise AssertionError(f"{states_path}: specialization cycle is {roles}")

        replay_env = make_env("alem", "default", config)
        replay_env.reset(seed=seed)
        saved_hash = _state_projection_hash(saved_initial_state)
        replay_hash = _state_projection_hash(replay_env.state)
        if saved_hash != replay_hash:
            raise AssertionError(f"{states_path}: deterministic replay hash mismatch")

        episode_records.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "steps": steps,
                "artifact_status": episode["artifact_status"],
                "action_parse_rate": episode["action_parse_rate"],
                "model_calls": episode["model_call_count"],
                "provider_requests": episode["provider_request_count"],
                "transport_errors": episode["transport_error_count"],
                "emitted_messages": worker_peer["emitted_messages"],
                "delivered_messages": worker_peer["delivered_messages"],
                "delivery_bytes": worker_peer["delivery_bytes"],
                "saved_state_count": len(saved_states),
                "saved_state_sha256": _sha256(states_path),
                "state_projection_sha256": saved_hash,
                "replay_projection_sha256": replay_hash,
                "trajectory_sha256": _sha256(trajectory_path),
            }
        )

    return PopulationResult(
        num_agents=num_agents,
        passed=True,
        episodes=tuple(episode_records),
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        give_mapping_count=len(give_mappings),
        role_values=observed_roles or (),
        expected_role_values=tuple(_expected_roles(num_agents)),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# E0 Variable-Agent Compatibility Gate",
        "",
        f"Overall result: **{'PASS' if summary['passed'] else 'FAIL'}**.",
        "",
        "| Agents | Episodes | Steps | Obs shape | Actions | Give mappings | "
        "Emitted | Delivered | Result |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for population in summary["populations"]:
        episodes = population["episodes"]
        lines.append(
            "| {n} | {episodes} | {steps} | `{shape}` | {actions} | {give} | "
            "{emitted} | {delivered} | {result} |".format(
                n=population["num_agents"],
                episodes=len(episodes),
                steps=sum(int(row["steps"]) for row in episodes),
                shape="×".join(str(value) for value in population["observation_shape"]),
                actions=population["action_space_size"],
                give=population["give_mapping_count"],
                emitted=sum(int(row["emitted_messages"]) for row in episodes),
                delivered=sum(int(row["delivered_messages"]) for row in episodes),
                result="PASS" if population["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "This is a provider-free engineering gate. Scripted agents always select "
            "`Noop` and emit deterministic messages; the result is not evidence of "
            "behavioral scaling.",
            "",
            f"Raw run root: `{summary['run_root']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_e0(
    *,
    output_root: Path,
    counts: tuple[int, ...] = DEFAULT_COUNTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"E0 output root already exists and is non-empty: {output_root}. "
            "Choose a new --output path."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    population_results: list[PopulationResult] = []

    for num_agents in counts:
        config = build_e0_config(num_agents=num_agents, seeds=seeds, steps=steps)
        population_dir = output_root / f"n{num_agents}"
        population_dir.mkdir(parents=True, exist_ok=False)
        (population_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(config, resolve=True),
            encoding="utf-8",
        )
        manager = EvaluatorManager(
            config,
            original_cwd=str(PROJECT_ROOT),
            output_dir=str(population_dir),
        )
        manager.run(ScriptedProbeFactory())
        population_results.append(
            validate_population(
                output_dir=population_dir,
                config=config,
                num_agents=num_agents,
                seeds=seeds,
                steps=steps,
            )
        )

    finished_at = datetime.now(UTC)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "passed": all(result.passed for result in population_results),
        "engineering_only": True,
        "provider_calls_authorized": False,
        "counts": list(counts),
        "seeds": list(seeds),
        "steps_per_episode": steps,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "run_root": str(output_root),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "source_status": (_git("status", "--short") or "").splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "populations": [result.as_dict() for result in population_results],
        "gates": {
            "all_populations_complete": True,
            "zero_model_calls": True,
            "zero_provider_requests": True,
            "zero_transport_errors": True,
            "all_action_parse_rates_one": True,
            "trajectory_agent_axes_match": True,
            "give_targets_round_trip": True,
            "specialization_cycle_matches": True,
            "broadcast_accounting_exact": True,
            "saved_state_replay_hashes_match": True,
        },
    }
    _write_json(output_root / "e0_results.json", summary)
    (output_root / "e0_results.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _parse_int_tuple(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/alem_eval/e0_agent_compatibility_v1"),
    )
    parser.add_argument("--counts", default="1,2,3,4,6")
    parser.add_argument("--seeds", default="13000,13001")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configurations and print the run matrix without creating artifacts.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    counts = _parse_int_tuple(args.counts, name="counts")
    seeds = _parse_int_tuple(args.seeds, name="seeds")
    for num_agents in counts:
        build_e0_config(num_agents=num_agents, seeds=seeds, steps=args.steps)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dry_run": True,
                    "counts": list(counts),
                    "seeds": list(seeds),
                    "steps_per_episode": args.steps,
                    "episodes": len(counts) * len(seeds),
                    "scripted_agent_ticks": sum(counts) * len(seeds) * args.steps,
                    "model_calls": 0,
                    "provider_requests": 0,
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    summary = run_e0(
        output_root=args.output.resolve(),
        counts=counts,
        seeds=seeds,
        steps=args.steps,
    )
    print(_summary_markdown(summary), end="")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
