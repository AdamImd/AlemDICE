#!/usr/bin/env python3
"""Run the auditable E1 Source-baseline population-scaling study."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, TextIO

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

PROFILE = "source_scaling_200"
PLANNED_COUNTS = (1, 2, 3, 4, 6)
STAGE_COUNTS = {
    "e1a": (1, 2, 3, 4),
    "e1b": (6,),
    "all": PLANNED_COUNTS,
}
SEEDS = (13100, 13101, 13102)
MAX_STEPS = 200
EPISODES_PER_COUNT = len(SEEDS)
DEFAULT_CAMPAIGN_LOGICAL_CALL_CAP = 12_001
DEFAULT_CAMPAIGN_PROVIDER_ATTEMPT_CAP = 15_000
MANIFEST_NAME = "study_manifest.json"
PREFLIGHT_NAME = "preflight.json"
RESOLVED_CONFIG_NAME = "resolved_config.yaml"
LOCK_NAME = ".study.lock"
ATTEMPT_PATTERN = re.compile(r"attempt_(\d+)\.json$")
ALLOWED_SOURCE_STATUS = {
    "?? Results/replays/nano_high_source_full_world.mp4",
}


class SourceScalingLaunchError(RuntimeError):
    """Raised when a paid run would be ambiguous or irreproducible."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--resume", type=Path, metavar="RUN_DIR")
    output.add_argument("--output-root", type=Path, metavar="RUN_DIR")
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_COUNTS),
        default="e1a",
        help="Run E1a (N=1--4), E1b (N=6), or all planned counts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose, validate, and print every command without writing or calling an API.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Make exactly one GPT-5.4 nano high request, record it, and exit.",
    )
    parser.add_argument(
        "--max-arm-attempts",
        type=int,
        default=1,
        help="Maximum evaluator invocations per incomplete population (default: 1).",
    )
    parser.add_argument(
        "--campaign-logical-call-cap",
        type=int,
        default=DEFAULT_CAMPAIGN_LOGICAL_CALL_CAP,
        help=(
            "Hard cumulative successful-response ceiling including preflight "
            f"(default: {DEFAULT_CAMPAIGN_LOGICAL_CALL_CAP:,})."
        ),
    )
    parser.add_argument(
        "--campaign-provider-attempt-cap",
        type=int,
        default=DEFAULT_CAMPAIGN_PROVIDER_ATTEMPT_CAP,
        help=(
            "Hard cumulative successful-request plus observed transport-error "
            f"ceiling (default: {DEFAULT_CAMPAIGN_PROVIDER_ATTEMPT_CAP:,})."
        ),
    )
    return parser


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably replace one small control file without exposing partial JSON/YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


@contextlib.contextmanager
def _study_lock(root: Path):
    """Prevent duplicate paid launchers from sharing a study root."""

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_NAME
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise SourceScalingLaunchError(
                f"Study root is already locked ({owner}): {root}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "acquired_at_utc": _now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _allowed_source_artifacts() -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for status_line in sorted(ALLOWED_SOURCE_STATUS):
        relative = status_line.removeprefix("?? ")
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        artifacts[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return artifacts


def _default_root(config) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = Path(str(config.eval.output_dir))
    if not base.is_absolute():
        base = REPO_ROOT / base
    return (base / f"{stamp}_e1_source_scaling_3seed_200").resolve()


def _cache_key(num_agents: int, agent_index: int) -> str:
    return f"alem:e1:g54n:n{num_agents}:a{agent_index}"


def _overrides(num_agents: int) -> tuple[str, ...]:
    values = [f"alem.num_agents={num_agents}"]
    values.extend(
        (f"clients.{index}.generate_kwargs.prompt_cache_key={_cache_key(num_agents, index)}")
        for index in range(6)
    )
    return tuple(values)


def _compose(num_agents: int):
    if num_agents not in PLANNED_COUNTS:
        raise SourceScalingLaunchError(f"Unsupported population N={num_agents}")
    config = compose_experiment(PROFILE, overrides=_overrides(num_agents))
    spec = validate_experiment_config(config, expected_profile=PROFILE)
    if (
        spec.num_agents != num_agents
        or spec.seeds != SEEDS
        or spec.max_steps_per_episode != MAX_STEPS
        or spec.num_workers != 1
    ):
        raise SourceScalingLaunchError(f"Resolved N={num_agents} profile violates E1")
    return config


def _arm_root(root: Path, num_agents: int) -> Path:
    return root / f"n{num_agents}"


def _difficulty_root(root: Path, num_agents: int) -> Path:
    return _arm_root(root, num_agents) / "easy"


def _episode_markers(root: Path, num_agents: int) -> tuple[Path, ...]:
    base = _difficulty_root(root, num_agents) / "alem" / "default"
    return tuple(base / f"default_run_{index:02d}.json" for index in range(3))


def _episode_marker_matches(
    path: Path,
    *,
    num_agents: int,
    episode_index: int,
) -> bool:
    if not _episode_marker_is_complete(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        payload.get("seed") != SEEDS[episode_index]
        or payload.get("physical_worker_count") != num_agents
        or payload.get("coordination_strategy") != "free"
        or not isinstance(payload.get("team"), dict)
        or payload["team"].get("topology") != "baseline"
        or not isinstance(payload.get("performance_metrics"), dict)
        or payload["performance_metrics"].get("schema_version") != "alem-dice-performance-v1"
    ):
        return False
    steps = payload.get("num_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not (1 <= steps <= MAX_STEPS):
        return False
    clients = payload.get("clients")
    if not isinstance(clients, list) or len(clients) < num_agents:
        return False
    for index in range(num_agents):
        client = clients[index]
        if (
            not isinstance(client, dict)
            or client.get("client_name") != "openai_responses"
            or client.get("model_id") != "gpt-5.4-nano"
            or client.get("prompt_cache_key_resolved")
            != f"{_cache_key(num_agents, index)}:traffic-0"
            or client.get("prompt_cache_traffic_shard_resolved") != 0
        ):
            return False
    stem = path.name.removesuffix(".json")
    companions = (
        path.with_name(f"{stem}.csv"),
        path.with_name(f"{stem}_trajectory.npz"),
        path.with_name(f"{stem}_states.pkl.gz"),
        path.with_name(f"{stem}_debug.jsonl"),
    )
    return all(companion.is_file() and companion.stat().st_size > 0 for companion in companions)


def _run_manifest_matches(root: Path, num_agents: int) -> bool:
    path = _difficulty_root(root, num_agents) / "run_manifest.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_source = _git_output("rev-parse", "HEAD")
    expected_lock = _lock_sha256()
    expected_config = _runtime_config_sha256(str(root), num_agents)
    if (
        payload.get("schema_version") != "alem-dice-run-manifest-v1"
        or payload.get("profile") != PROFILE
        or payload.get("difficulty") != "easy"
        or payload.get("source_commit") != expected_source
        or payload.get("uv_lock_sha256") != expected_lock
        or not isinstance(payload.get("resolved_config_sha256"), str)
        or payload.get("models") != ["gpt-5.4-nano"] * 6
        or not isinstance(payload.get("resume_history"), list)
    ):
        return False
    invocations = [
        {
            "source_commit": payload.get("source_commit"),
            "uv_lock_sha256": payload.get("uv_lock_sha256"),
            "resolved_config_file": payload.get("resolved_config_file"),
            "resolved_config_sha256": payload.get("resolved_config_sha256"),
        },
        *payload["resume_history"],
    ]
    for invocation in invocations:
        if (
            not isinstance(invocation, dict)
            or invocation.get("source_commit") != expected_source
            or invocation.get("uv_lock_sha256") != expected_lock
            or not isinstance(invocation.get("resolved_config_sha256"), str)
        ):
            return False
        filename = invocation.get("resolved_config_file")
        config_path = (
            _difficulty_root(root, num_agents) / filename if isinstance(filename, str) else None
        )
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or config_path is None
            or not config_path.is_file()
            or _sha256_file(config_path) != invocation.get("resolved_config_sha256")
            or _normalized_config_file_sha256(config_path) != expected_config
        ):
            return False
    return True


def _completed_count(root: Path, num_agents: int) -> int:
    markers = _episode_markers(root, num_agents)
    matched = sum(
        _episode_marker_matches(
            marker,
            num_agents=num_agents,
            episode_index=episode_index,
        )
        for episode_index, marker in enumerate(markers)
    )
    return matched if matched == 0 or _run_manifest_matches(root, num_agents) else 0


def _command(root: Path, num_agents: int) -> tuple[str, ...]:
    return (
        sys.executable,
        str(REPO_ROOT / "baselines" / "llm" / "eval_alem.py"),
        f"experiment={PROFILE}",
        *_overrides(num_agents),
        "alem.coordination_difficulty=easy",
        f"eval.resume_from={_difficulty_root(root, num_agents)}",
    )


@cache
def _runtime_config_sha256(root_text: str, num_agents: int) -> str:
    root = Path(root_text)
    config = compose_experiment(
        PROFILE,
        overrides=(
            *_overrides(num_agents),
            "alem.coordination_difficulty=easy",
            f"eval.resume_from={_difficulty_root(root, num_agents)}",
        ),
    )
    return _normalized_config_sha256(config)


def _normalized_config_sha256(config: object) -> str:
    """Hash experiment semantics after removing W&B's generated run identifier."""

    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise SourceScalingLaunchError("Resolved runtime configuration is not a mapping")
    wandb_payload = payload.get("wandb")
    if isinstance(wandb_payload, dict):
        wandb_payload.pop("run_id", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _normalized_config_file_sha256(path: Path) -> str:
    from omegaconf import OmegaConf

    return _normalized_config_sha256(OmegaConf.load(path))


def _resolved_config_text(config: object) -> str:
    from omegaconf import OmegaConf

    return OmegaConf.to_yaml(config, resolve=True)


def _manifest(
    root: Path,
    configs: dict[int, object],
    *,
    campaign_logical_call_cap: int,
    campaign_provider_attempt_cap: int,
) -> dict[str, object]:
    source_status = list(_git_status_lines())
    return {
        "schema_version": "alem-dice-source-scaling-e1-v1",
        "profile": PROFILE,
        "planned_counts": list(PLANNED_COUNTS),
        "stage_definitions": {stage: list(counts) for stage, counts in STAGE_COUNTS.items()},
        "seeds": list(SEEDS),
        "difficulty": "easy",
        "max_steps_per_episode": MAX_STEPS,
        "episodes_per_count": EPISODES_PER_COUNT,
        "episode_workers": 1,
        "agent_calls_within_tick": "concurrent",
        "model": "gpt-5.4-nano",
        "reasoning_effort": "high",
        "topology": "baseline",
        "coordination_strategy": "free",
        "agent_type": "robust_all",
        "prompt_mode": "specific_collaborative",
        "nominal_decision_call_cap": sum(
            count * EPISODES_PER_COUNT * MAX_STEPS for count in PLANNED_COUNTS
        ),
        "campaign_logical_call_cap": campaign_logical_call_cap,
        "campaign_provider_attempt_cap": campaign_provider_attempt_cap,
        "nominal_call_cap_by_count": {
            str(count): count * EPISODES_PER_COUNT * MAX_STEPS for count in PLANNED_COUNTS
        },
        "cache_keys": {
            str(count): [_cache_key(count, index) for index in range(count)]
            for count in PLANNED_COUNTS
        },
        "resolved_config_sha256": {
            str(count): _config_sha256(config) for count, config in configs.items()
        },
        "resolved_config_file_sha256": {
            str(count): _sha256_bytes(_resolved_config_text(config).encode("utf-8"))
            for count, config in configs.items()
        },
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_branch": _git_output("branch", "--show-current"),
        "source_status_at_creation": source_status,
        "allowed_preexisting_source_status": sorted(ALLOWED_SOURCE_STATUS),
        "allowed_preexisting_artifacts": _allowed_source_artifacts(),
        "uv_lock_sha256": _lock_sha256(),
        "output_root": str(root),
    }


def _manifest_invariants(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "schema_version",
        "profile",
        "planned_counts",
        "stage_definitions",
        "seeds",
        "difficulty",
        "max_steps_per_episode",
        "episodes_per_count",
        "episode_workers",
        "model",
        "reasoning_effort",
        "topology",
        "coordination_strategy",
        "agent_type",
        "prompt_mode",
        "nominal_decision_call_cap",
        "campaign_logical_call_cap",
        "campaign_provider_attempt_cap",
        "nominal_call_cap_by_count",
        "cache_keys",
        "resolved_config_sha256",
        "resolved_config_file_sha256",
        "source_commit",
        "source_branch",
        "allowed_preexisting_artifacts",
        "uv_lock_sha256",
        "output_root",
    )
    return {key: payload.get(key) for key in keys}


def _validate_resume(root: Path, expected: dict[str, object]) -> dict[str, object]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise SourceScalingLaunchError(f"Resume root is missing {MANIFEST_NAME}: {root}")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceScalingLaunchError(f"Cannot read resume manifest: {exc}") from exc
    if _manifest_invariants(actual) != _manifest_invariants(expected):
        raise SourceScalingLaunchError(
            "Resume manifest does not match the current source/configuration"
        )
    expected_file_hashes = expected["resolved_config_file_sha256"]
    if not isinstance(expected_file_hashes, dict):
        raise SourceScalingLaunchError("Invalid resolved-config hash manifest")
    for count in PLANNED_COUNTS:
        path = _arm_root(root, count) / RESOLVED_CONFIG_NAME
        expected_hash = expected_file_hashes.get(str(count))
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise SourceScalingLaunchError(
                f"Saved N={count} resolved configuration is missing or changed"
            )
    return actual


def _blocking_source_status() -> tuple[str, ...]:
    return tuple(line for line in _git_status_lines() if line not in ALLOWED_SOURCE_STATUS)


def _campaign_usage(root: Path) -> dict[str, int]:
    totals = {
        "logical_responses": 0,
        "provider_attempts_observed": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "transport_errors": 0,
    }
    for ledger in root.glob("n*/easy/alem/default/attempt_ledger.jsonl"):
        try:
            lines = ledger.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SourceScalingLaunchError(f"Cannot read usage ledger {ledger}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceScalingLaunchError(
                    f"Invalid usage ledger JSON at {ledger}:{line_number}"
                ) from exc
            model_calls = int(row.get("model_call_count", 0) or 0)
            provider_requests = int(row.get("provider_request_count", 0) or 0)
            transport_errors = int(row.get("transport_error_count", 0) or 0)
            totals["logical_responses"] += model_calls
            # provider_request_count already includes successful and failed
            # transport attempts; transport_error_count is a labeled subset.
            totals["provider_attempts_observed"] += provider_requests
            totals["transport_errors"] += transport_errors
            for key in (
                "input_tokens",
                "cached_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                totals[key] += int(row.get(key, 0) or 0)
    for attempt_path in root.glob("n*/attempts/attempt_*.json"):
        try:
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceScalingLaunchError(f"Cannot audit launcher attempt {attempt_path}") from exc
        if attempt.get("status") in {"running", "interrupted"}:
            raise SourceScalingLaunchError(
                "Paid usage may be missing after an unfinalized/interrupted evaluator; "
                f"manual ledger reconciliation is required: {attempt_path}"
            )
    preflight_dir = root / "preflight_attempts"
    if preflight_dir.is_dir():
        for path in sorted(preflight_dir.glob("preflight_*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SourceScalingLaunchError(
                    f"Cannot read preflight usage artifact {path}"
                ) from exc
            if row.get("status") == "running":
                raise SourceScalingLaunchError(
                    f"Unfinalized preflight artifact requires audit: {path}"
                )
            totals["logical_responses"] += int(row.get("logical_response_count", 0) or 0)
            totals["provider_attempts_observed"] += int(row.get("transport_attempt_count", 0) or 0)
            totals["transport_errors"] += int(row.get("transport_error_count", 0) or 0)
            for key in (
                "input_tokens",
                "cached_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                totals[key] += int(row.get(key, 0) or 0)
    return totals


def _remaining_nominal_calls(root: Path, counts: Sequence[int]) -> int:
    return sum(
        (EPISODES_PER_COUNT - _completed_count(root, count)) * count * MAX_STEPS for count in counts
    )


def _enforce_campaign_budget(
    root: Path,
    manifest: dict[str, object],
    *,
    next_nominal_calls: int,
) -> dict[str, int]:
    usage = _campaign_usage(root)
    logical_cap = int(manifest["campaign_logical_call_cap"])
    provider_cap = int(manifest["campaign_provider_attempt_cap"])
    projected_logical = usage["logical_responses"] + next_nominal_calls
    projected_provider = usage["provider_attempts_observed"] + next_nominal_calls
    if projected_logical > logical_cap:
        raise SourceScalingLaunchError(
            "Campaign logical-response ceiling would be exceeded: "
            f"spent={usage['logical_responses']:,}, "
            f"next nominal={next_nominal_calls:,}, cap={logical_cap:,}"
        )
    if projected_provider > provider_cap:
        raise SourceScalingLaunchError(
            "Campaign provider-attempt ceiling would be exceeded: "
            f"observed={usage['provider_attempts_observed']:,}, "
            f"next nominal={next_nominal_calls:,}, cap={provider_cap:,}"
        )
    return usage


def _validate_paid_run(manifest: dict[str, object]) -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SourceScalingLaunchError("OPENAI_API_KEY is required")
    blocking = _blocking_source_status()
    if blocking:
        raise SourceScalingLaunchError(
            "Paid study requires a reproducible tracked source tree:\n  " + "\n  ".join(blocking)
        )
    if not manifest["source_commit"] or not manifest["uv_lock_sha256"]:
        raise SourceScalingLaunchError("Paid study requires a Git commit and uv.lock")


def _prepare_new(
    root: Path, manifest: dict[str, object], configs: dict[int, object]
) -> dict[str, object]:
    existing = [path for path in root.iterdir() if path.name != LOCK_NAME] if root.is_dir() else []
    if root.exists() and (not root.is_dir() or existing):
        raise SourceScalingLaunchError(f"New output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["created_at_utc"] = _now()
    _atomic_write_json(root / MANIFEST_NAME, payload)

    for count, config in configs.items():
        arm = _arm_root(root, count)
        arm.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            arm / RESOLVED_CONFIG_NAME,
            _resolved_config_text(config),
        )
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_json(path, payload)


def _preflight_expectation(manifest: dict[str, object]) -> dict[str, object]:
    config_hashes = manifest.get("resolved_config_sha256")
    if not isinstance(config_hashes, dict):
        raise SourceScalingLaunchError("Manifest is missing resolved configuration hashes")
    return {
        "schema_version": "alem-dice-source-scaling-preflight-v1",
        "source_commit": manifest["source_commit"],
        "uv_lock_sha256": manifest["uv_lock_sha256"],
        "resolved_config_sha256": config_hashes["1"],
        "requested_model": "gpt-5.4-nano",
        "reasoning_effort": "high",
        "configured_prompt_cache_key": "alem:e1:g54n:preflight",
        "effective_prompt_cache_key": "alem:e1:g54n:preflight:traffic-0",
        "prompt_cache_traffic_shard": 0,
    }


def _valid_preflight(
    payload: Any,
    expectation: dict[str, object],
) -> bool:
    if not isinstance(payload, dict):
        return False
    response_model = payload.get("model_id")
    requested_model = expectation["requested_model"]
    model_matches = response_model == requested_model or (
        isinstance(response_model, str)
        and isinstance(requested_model, str)
        and response_model.startswith(f"{requested_model}-")
    )
    return (
        all(payload.get(key) == value for key, value in expectation.items())
        and payload.get("status") == "passed"
        and payload.get("parse_success") is True
        and payload.get("parsed_action") == "Noop"
        and model_matches
        and payload.get("incomplete_reason") is None
        and payload.get("transport_error_count") == 0
        and payload.get("logical_response_count") == 1
        and isinstance(payload.get("response_id"), str)
        and bool(payload["response_id"])
    )


def _next_preflight_attempt(root: Path) -> int:
    attempt_dir = root / "preflight_attempts"
    numbers = []
    if attempt_dir.is_dir():
        for path in attempt_dir.glob("preflight_*.json"):
            match = re.fullmatch(r"preflight_(\d+)\.json", path.name)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _run_preflight(root: Path, config, manifest: dict[str, object]) -> int:
    from omegaconf import OmegaConf

    from baselines.llm.eval_utils.agents.robust_all import _extract_safe_action
    from baselines.llm.eval_utils.client import create_llm_client
    from baselines.llm.eval_utils.prompt_builder import Message

    canonical_path = root / PREFLIGHT_NAME
    expectation = _preflight_expectation(manifest)
    if canonical_path.exists():
        try:
            canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceScalingLaunchError(
                f"Cannot validate canonical preflight {canonical_path}: {exc}"
            ) from exc
        if _valid_preflight(canonical, expectation):
            print(f"Preflight already passed: {canonical_path}")
            return 0
        raise SourceScalingLaunchError(
            f"Canonical preflight is invalid or does not match this source: {canonical_path}"
        )

    attempt = _next_preflight_attempt(root)
    if attempt > 3:
        raise SourceScalingLaunchError(
            "Preflight retry ceiling reached; inspect preflight_attempts before continuing"
        )
    _enforce_campaign_budget(root, manifest, next_nominal_calls=1)
    attempt_path = root / "preflight_attempts" / f"preflight_{attempt:02d}.json"
    client_payload = OmegaConf.to_container(config.clients[0], resolve=True)
    if not isinstance(client_payload, dict):
        raise SourceScalingLaunchError("N=1 client configuration is not a mapping")
    generate_kwargs = dict(client_payload.get("generate_kwargs", {}))
    generate_kwargs["prompt_cache_key"] = expectation["configured_prompt_cache_key"]
    generate_kwargs["prompt_cache_traffic_shards"] = 1
    client_payload["generate_kwargs"] = generate_kwargs
    client = create_llm_client(OmegaConf.create(client_payload))()
    effective_cache_key = getattr(client, "effective_prompt_cache_key", None)
    effective_cache_shard = getattr(client, "prompt_cache_traffic_shard", None)
    if (
        effective_cache_key != expectation["effective_prompt_cache_key"]
        or effective_cache_shard != expectation["prompt_cache_traffic_shard"]
    ):
        raise SourceScalingLaunchError("Preflight cache route did not resolve deterministically")
    started = _now()
    running = {
        **expectation,
        "attempt": attempt,
        "started_at_utc": started,
        "status": "running",
        "logical_response_count": 0,
    }
    _atomic_write_json(attempt_path, running)
    try:
        response = client.generate(
            [
                Message(
                    role="system",
                    content=(
                        "Alem E1 transport and action-format compatibility check. "
                        "Return exactly <action>Noop</action>."
                    ),
                ),
                Message(role="user", content="Select Noop now."),
            ]
        )
        parsed = _extract_safe_action(response)
        passed = (
            parsed == "Noop"
            and response.incomplete_reason is None
            and int(response.transport_error_count or 0) == 0
        )
        payload = {
            **expectation,
            "attempt": attempt,
            "started_at_utc": started,
            "completed_at_utc": _now(),
            "status": "passed" if passed else "failed",
            "model_id": response.model_id,
            "completion": response.completion,
            "parsed_action": parsed,
            "parse_success": parsed == "Noop",
            "stop_reason": response.stop_reason,
            "provider_status": response.status,
            "incomplete_reason": response.incomplete_reason,
            "response_id": response.response_id,
            "input_tokens": int(response.input_tokens or 0),
            "cached_tokens": int(response.cached_tokens or 0),
            "output_tokens": int(response.output_tokens or 0),
            "reasoning_tokens": int(response.reasoning_tokens or 0),
            "latency_seconds": float(response.latency_seconds or 0.0),
            "logical_response_count": 1,
            "transport_attempt_count": int(response.transport_attempt_count or 1),
            "transport_error_count": int(response.transport_error_count or 0),
            "transport_error_types": list(response.transport_error_types or ()),
        }
    except Exception as exc:
        transport_attempts = int(getattr(client, "last_transport_attempt_count", 0) or 0)
        transport_errors = int(
            getattr(client, "last_transport_error_count", transport_attempts) or transport_attempts
        )
        payload = {
            **running,
            "completed_at_utc": _now(),
            "status": "exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "transport_attempt_count": transport_attempts,
            "transport_error_count": transport_errors,
            "transport_error_types": list(getattr(client, "last_transport_error_types", ()) or ()),
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        _atomic_write_json(attempt_path, payload)
        raise SourceScalingLaunchError(
            f"Preflight provider call failed; see {attempt_path}"
        ) from exc
    _atomic_write_json(attempt_path, payload)
    if not _valid_preflight(payload, expectation):
        raise SourceScalingLaunchError(
            f"Preflight response failed its compatibility contract; see {attempt_path}"
        )
    _atomic_write_json(canonical_path, payload)
    print("Preflight complete: one parseable GPT-5.4 nano high call")
    print(
        "Tokens: "
        f"input={payload['input_tokens']:,}, cached={payload['cached_tokens']:,}, "
        f"output={payload['output_tokens']:,}, "
        f"reasoning={payload['reasoning_tokens']:,}"
    )
    print(f"Artifact: {canonical_path}")
    return 0


def _next_attempt(arm: Path) -> int:
    attempt_dir = arm / "attempts"
    numbers = []
    if attempt_dir.is_dir():
        for path in attempt_dir.glob("attempt_*.json"):
            match = ATTEMPT_PATTERN.fullmatch(path.name)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def _run_logged(command: Sequence[str], arm: Path) -> int:
    attempt = _next_attempt(arm)
    attempt_dir = arm / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stem = attempt_dir / f"attempt_{attempt:02d}"
    log_path = stem.with_suffix(".log")
    metadata_path = stem.with_suffix(".json")
    if metadata_path.exists() or log_path.exists():
        raise SourceScalingLaunchError(f"Attempt artifact already exists: {stem}")
    started = _now()
    root = arm.parent
    before = _completed_count(root, int(arm.name.removeprefix("n")))
    usage_before = _campaign_usage(root)
    _write_json(
        metadata_path,
        {
            "schema_version": "alem-dice-source-scaling-attempt-v1",
            "attempt": attempt,
            "started_at_utc": started,
            "status": "running",
            "command": list(command),
            "command_shell": shlex.join(command),
            "completed_episodes_before": before,
            "campaign_usage_before": usage_before,
            "log_path": str(log_path),
        },
    )
    process: subprocess.Popen | None = None
    return_code = -1
    interrupted: BaseException | None = None
    previous_handlers: dict[int, Any] = {}

    def _interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _interrupt)
        with log_path.open("x", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
            return_code = process.wait()
            log_file.flush()
            os.fsync(log_file.fileno())
    except BaseException as exc:
        interrupted = exc
        if process is not None:
            _terminate_process_group(process)
            return_code = process.returncode if process.returncode is not None else -1
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    num_agents = int(arm.name.removeprefix("n"))
    after = _completed_count(root, num_agents)
    final_payload = {
        "schema_version": "alem-dice-source-scaling-attempt-v1",
        "attempt": attempt,
        "started_at_utc": started,
        "completed_at_utc": _now(),
        "status": (
            "interrupted"
            if interrupted is not None
            else "process_exited"
            if return_code == 0
            else "evaluator_error"
        ),
        "return_code": return_code,
        "command": list(command),
        "command_shell": shlex.join(command),
        "completed_episodes_before": before,
        "completed_episodes_after": after,
        "campaign_usage_before": usage_before,
        "interruption_type": (type(interrupted).__name__ if interrupted is not None else None),
        "interruption_message": str(interrupted) if interrupted is not None else None,
        "log_path": str(log_path),
    }
    _write_json(metadata_path, final_payload)
    try:
        usage_after: dict[str, int] | dict[str, str] = _campaign_usage(root)
    except SourceScalingLaunchError as exc:
        usage_after = {"audit_error": str(exc)}
    final_payload["campaign_usage_after"] = usage_after
    _write_json(metadata_path, final_payload)
    if interrupted is not None:
        raise interrupted
    return return_code


def _print_plan(
    root: Path, stage: str, configs: dict[int, object], manifest: dict[str, object]
) -> None:
    counts = STAGE_COUNTS[stage]
    remaining = sum(
        (EPISODES_PER_COUNT - _completed_count(root, count)) * count * MAX_STEPS for count in counts
    )
    print(f"Profile: {PROFILE}")
    print(f"Stage: {stage}; populations: {','.join(map(str, counts))}")
    print(f"Output root: {root}")
    print(f"Source commit: {manifest['source_commit'] or 'unknown'}")
    print(f"UV lock SHA-256: {manifest['uv_lock_sha256'] or 'missing'}")
    print(f"Planned E1 decision-call cap: {manifest['nominal_decision_call_cap']:,}")
    print(f"Hard campaign logical-response ceiling: {manifest['campaign_logical_call_cap']:,}")
    print(f"Hard campaign provider-attempt ceiling: {manifest['campaign_provider_attempt_cap']:,}")
    print(f"Remaining stage decision-call cap: {remaining:,}")
    for count in counts:
        completed = _completed_count(root, count)
        spec = validate_experiment_config(configs[count])
        print(
            f"  N={count}: {completed}/{EPISODES_PER_COUNT} complete; "
            f"at most {spec.num_agents} simultaneous model calls"
        )
        print("    " + shlex.join(_command(root, count)))


def run_study(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_arm_attempts < 1:
        raise SourceScalingLaunchError("--max-arm-attempts must be positive")
    if args.campaign_logical_call_cap < 9_601:
        raise SourceScalingLaunchError(
            "--campaign-logical-call-cap must cover 9,600 planned episode calls "
            "plus the provider preflight"
        )
    if args.campaign_provider_attempt_cap < 9_601:
        raise SourceScalingLaunchError(
            "--campaign-provider-attempt-cap must cover the planned campaign"
        )

    configs = {count: _compose(count) for count in PLANNED_COUNTS}
    root = (
        args.resume.resolve()
        if args.resume
        else args.output_root.resolve()
        if args.output_root
        else _default_root(configs[1])
    )
    expected = _manifest(
        root,
        configs,
        campaign_logical_call_cap=args.campaign_logical_call_cap,
        campaign_provider_attempt_cap=args.campaign_provider_attempt_cap,
    )
    _print_plan(root, args.stage, configs, expected)
    if args.dry_run:
        print("Dry run complete; no files or API calls were made.")
        return 0

    _validate_paid_run(expected)
    with _study_lock(root):
        if args.resume:
            manifest = _validate_resume(root, expected)
        else:
            manifest = _prepare_new(root, expected, configs)

        if args.preflight:
            result = _run_preflight(root, configs[1], manifest)
            manifest["preflight_sha256"] = _sha256_file(root / PREFLIGHT_NAME)
            manifest["campaign_usage"] = _campaign_usage(root)
            manifest["last_run_completed_at_utc"] = _now()
            _atomic_write_json(root / MANIFEST_NAME, manifest)
            return result

        preflight_path = root / PREFLIGHT_NAME
        if not preflight_path.is_file():
            raise SourceScalingLaunchError(
                f"Missing {PREFLIGHT_NAME}; run this launcher with --resume {root} "
                "--preflight before the study"
            )
        try:
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceScalingLaunchError(f"Cannot read canonical preflight: {exc}") from exc
        if not _valid_preflight(preflight, _preflight_expectation(manifest)):
            raise SourceScalingLaunchError("Recorded preflight does not match this study")
        if manifest.get("preflight_sha256") != _sha256_file(preflight_path):
            raise SourceScalingLaunchError("Canonical preflight changed after manifest binding")

        if args.stage == "e1b" and any(
            _completed_count(root, count) != EPISODES_PER_COUNT for count in STAGE_COUNTS["e1a"]
        ):
            raise SourceScalingLaunchError("E1b requires a complete E1a screen")

        stage_counts = STAGE_COUNTS[args.stage]
        stage_remaining = _remaining_nominal_calls(root, stage_counts)
        usage = _enforce_campaign_budget(
            root,
            manifest,
            next_nominal_calls=stage_remaining,
        )
        print(
            "Campaign usage before stage: "
            f"{usage['logical_responses']:,} logical responses, "
            f"{usage['input_tokens']:,} input tokens, "
            f"{usage['output_tokens']:,} output tokens"
        )

        failures: list[int] = []
        for count in stage_counts:
            if _completed_count(root, count) == EPISODES_PER_COUNT:
                print(f"Skipping N={count}: all episodes complete.")
                continue
            arm = _arm_root(root, count)
            for _ in range(args.max_arm_attempts):
                completed = _completed_count(root, count)
                if completed == EPISODES_PER_COUNT:
                    break
                next_nominal = (EPISODES_PER_COUNT - completed) * count * MAX_STEPS
                _enforce_campaign_budget(
                    root,
                    manifest,
                    next_nominal_calls=next_nominal,
                )
                print(
                    f"Running N={count}: {completed}/{EPISODES_PER_COUNT} complete",
                    flush=True,
                )
                return_code = _run_logged(_command(root, count), arm)
                completed_after = _completed_count(root, count)
                if return_code != 0:
                    print(
                        f"N={count} evaluator returned {return_code}; "
                        f"{completed_after}/{EPISODES_PER_COUNT} canonical episodes complete.",
                        flush=True,
                    )
            if _completed_count(root, count) != EPISODES_PER_COUNT:
                failures.append(count)
            manifest["completed_episodes_by_count"] = {
                str(value): _completed_count(root, value) for value in PLANNED_COUNTS
            }
            manifest["campaign_usage"] = _campaign_usage(root)
            manifest["last_requested_stage"] = args.stage
            manifest["last_progress_at_utc"] = _now()
            _atomic_write_json(root / MANIFEST_NAME, manifest)

        manifest["last_run_completed_at_utc"] = _now()
        manifest["campaign_usage"] = _campaign_usage(root)
        _atomic_write_json(root / MANIFEST_NAME, manifest)
        if failures:
            raise SourceScalingLaunchError(
                "Incomplete populations after retry budget: "
                + ", ".join(f"N={count}" for count in failures)
            )
        print(f"Stage {args.stage} complete: {root}")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_study())
    except SourceScalingLaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
