#!/usr/bin/env python3
"""Dry-run or explicitly execute the frozen, resumable E2b Luna screen."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.llm.eval_utils.client import create_llm_client  # noqa: E402
from baselines.llm.eval_utils.team_formation import RecruitmentMethod  # noqa: E402
from baselines.llm.recruitment_arena import (  # noqa: E402
    DEFAULT_ROUNDS,
    ScenarioFamily,
    generate_scenario,
)
from baselines.llm.recruitment_llm_screen import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SEMANTIC_REPAIRS,
    DEFAULT_STALL_ROUNDS,
    DEFAULT_TRANSPORT_RETRIES,
    SCHEMA_VERSION,
    SUPPORTED_METHODS,
    CampaignBudget,
    ScreenConfig,
    canonical_json,
    estimate_campaign,
    run_llm_recruitment_episode,
    sha256_text,
)

FROZEN_SEEDS = (22000, 22001, 22002)
FROZEN_FAMILIES = tuple(ScenarioFamily)
FROZEN_METHODS = SUPPORTED_METHODS
CANARY_CELL = (
    22000,
    ScenarioFamily.SINGLE_COMPLEMENTARY,
    RecruitmentMethod.OPEN_VOLUNTEER,
)
DEFAULT_LOGICAL_CALL_CAP = 2_160
DEFAULT_PROVIDER_ATTEMPT_CAP = 4_320
DEFAULT_TOKEN_EXPOSURE_CAP = 73_543_680
DEFAULT_OUTPUT = Path("outputs/recruitment_llm/e2b_luna_screen_v1")
PROTOCOL_PATH = Path("reports/agent_scaling_recruitment/e2b_protocol.md")
SOURCE_PATHS = (
    Path("baselines/llm/eval_utils/client.py"),
    Path("baselines/llm/eval_utils/openai_responses.py"),
    Path("baselines/llm/eval_utils/prompt_builder.py"),
    Path("baselines/llm/eval_utils/recruitment_selection.py"),
    Path("baselines/llm/eval_utils/team_formation.py"),
    Path("baselines/llm/recruitment_arena.py"),
    Path("baselines/llm/recruitment_llm_screen.py"),
    Path("scripts/run_recruitment_llm_screen.py"),
    PROTOCOL_PATH,
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


def _source_hashes() -> dict[str, str]:
    return {
        str(path): _sha256(PROJECT_ROOT / path)
        for path in SOURCE_PATHS
        if (PROJECT_ROOT / path).exists()
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON and atomically replace the destination in one directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{id(payload)}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact_paths(
    output: Path,
    *,
    seed: int,
    family: ScenarioFamily,
    method: RecruitmentMethod,
) -> tuple[Path, Path, Path]:
    stem = f"{method.value}__{family.value}__seed_{seed}"
    return (
        output / "episodes" / f"{stem}.json",
        output / "debug" / f"{stem}.calls.jsonl.gz",
        output / "markers" / f"{stem}.complete.json",
    )


def _write_debug_shard(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Atomically write deterministic gzip JSONL and return both content hashes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{id(records)}")
    content_digest = hashlib.sha256()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    line = canonical_json(record) + "\n"
                    content_digest.update(line.encode("ascii"))
                    text.write(line)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return {
        "debug_record_count": len(records),
        "debug_content_sha256": content_digest.hexdigest(),
        "debug_gzip_sha256": _sha256(path),
    }


def _validate_debug_shard(
    path: Path,
    *,
    expected_count: int,
    expected_content_sha256: str,
    expected_gzip_sha256: str,
) -> bool:
    """Replay embedded hashes without contacting a provider."""

    if not path.exists() or _sha256(path) != expected_gzip_sha256:
        return False
    content_digest = hashlib.sha256()
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="\n") as handle:
            for line in handle:
                content_digest.update(line.encode("ascii"))
                record = json.loads(line)
                if set(record) != {
                    "schema_version",
                    "scenario_id",
                    "family",
                    "seed",
                    "method",
                    "selector",
                    "round_index",
                    "agent_id",
                    "attempt",
                    "prompt_projection",
                    "messages",
                    "raw_completion",
                    "normalized_parse",
                    "response",
                    "hashes",
                }:
                    return False
                hashes = record["hashes"]
                projection = dict(record["prompt_projection"])
                own_private = projection.pop("own_private")
                scenario = generate_scenario(
                    ScenarioFamily(record["family"]),
                    int(record["seed"]),
                )
                profile = scenario.agents[int(record["agent_id"])]
                if own_private != {
                    "true_capabilities": list(profile.true_capabilities),
                    "task_costs": dict(sorted(profile.task_costs.items())),
                }:
                    return False
                public_projection = canonical_json(projection)
                if any(
                    forbidden in public_projection
                    for forbidden in (
                        '"true_capabilities"',
                        '"task_costs"',
                        '"pending_control"',
                        '"oracle',
                    )
                ):
                    return False
                if record["messages"][1]["content"] != (
                    f"AGENT_VIEW_JSON={canonical_json(record['prompt_projection'])}"
                ):
                    return False
                if int(record["attempt"]) > 0 and not any(
                    "Repair it once" in message["content"] for message in record["messages"]
                ):
                    return False
                if set(record["response"]) != {
                    "id",
                    "model_id",
                    "status",
                    "stop_reason",
                    "incomplete_reason",
                    "usage",
                    "transport_attempt_count",
                    "transport_error_count",
                    "transport_error_types",
                }:
                    return False
                if hashes["prompt_projection_sha256"] != sha256_text(
                    canonical_json(record["prompt_projection"])
                ):
                    return False
                if hashes["messages_sha256"] != sha256_text(canonical_json(record["messages"])):
                    return False
                completion = record["raw_completion"]
                expected_completion = None if completion is None else sha256_text(str(completion))
                if hashes["raw_completion_sha256"] != expected_completion:
                    return False
                count += 1
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return False
    return count == expected_count and content_digest.hexdigest() == expected_content_sha256


def persist_completed_episode(
    output: Path,
    episode: dict[str, Any],
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Persist the artifact first and its atomic completion marker second."""

    scenario = episode["scenario"]
    method = RecruitmentMethod(episode["config"]["method"])
    family = ScenarioFamily(scenario["family"])
    artifact, debug_path, marker = _artifact_paths(
        output,
        seed=int(scenario["seed"]),
        family=family,
        method=method,
    )
    debug_records = list(episode.get("_debug_call_records", ()))
    public_episode = {key: value for key, value in episode.items() if key != "_debug_call_records"}
    atomic_write_json(artifact, public_episode)
    debug_hashes = _write_debug_shard(debug_path, debug_records)
    artifact_hash = _sha256(artifact)
    debug_valid = _validate_debug_shard(
        debug_path,
        expected_count=debug_hashes["debug_record_count"],
        expected_content_sha256=debug_hashes["debug_content_sha256"],
        expected_gzip_sha256=debug_hashes["debug_gzip_sha256"],
    )
    if not debug_valid:
        raise RuntimeError("debug shard failed immediate replay/hash validation")
    calls = episode["call_ledger"]
    max_provider_attempts = 1 + int(episode["config"]["max_transport_retries"])
    max_output_tokens = int(episode["config"]["max_output_tokens"])
    budget_exhausted_decisions = sum(
        str(decision.get("abstain_code", "")).startswith("budget.")
        for round_record in episode["rounds"]
        for decision in round_record["decisions"]
    )
    marker_payload = {
        "schema_version": "alem-dice-e2b-complete-marker-v1",
        "status": "complete",
        "seed": int(scenario["seed"]),
        "family": family.value,
        "method": method.value,
        "artifact": str(artifact.relative_to(output)),
        "artifact_sha256": artifact_hash,
        "debug_artifact": str(debug_path.relative_to(output)),
        **debug_hashes,
        "debug_replay_valid": True,
        "deterministic_episode_hash": episode["deterministic_episode_hash"],
        "terminal_state_hash": episode["terminal_state_hash"],
        "terminal_audit_chain_hash": episode["terminal_audit_chain_hash"],
        "logical_calls": len(calls),
        "provider_attempts_actual": sum(int(call["transport_attempt_count"]) for call in calls),
        "provider_attempts_reserved": len(calls) * max_provider_attempts,
        "tokens_actual": sum(
            int(call["input_tokens"]) + int(call["output_tokens"]) for call in calls
        ),
        "tokens_reserved": sum(
            (int(call["prompt_bytes"]) + max_output_tokens) * max_provider_attempts
            for call in calls
        ),
        "budget_exhausted_decisions": budget_exhausted_decisions,
        "config_sha256": config_sha256,
        "source_hashes": source_hashes,
    }
    atomic_write_json(marker, marker_payload)
    return marker_payload


def load_completed_marker(
    output: Path,
    *,
    seed: int,
    family: ScenarioFamily,
    method: RecruitmentMethod,
    config_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, Any] | None:
    """Return a valid marker, otherwise leave the episode eligible for rerun."""

    artifact, debug_path, marker = _artifact_paths(output, seed=seed, family=family, method=method)
    if not marker.exists() or not artifact.exists() or not debug_path.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        debug_record_count = int(payload.get("debug_record_count", -1))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if (
        payload.get("status") != "complete"
        or payload.get("config_sha256") != config_sha256
        or payload.get("source_hashes") != source_hashes
        or payload.get("artifact_sha256") != _sha256(artifact)
        or not _validate_debug_shard(
            debug_path,
            expected_count=debug_record_count,
            expected_content_sha256=str(payload.get("debug_content_sha256", "")),
            expected_gzip_sha256=str(payload.get("debug_gzip_sha256", "")),
        )
    ):
        return None
    return payload


def _write_canary_gate(
    output: Path,
    marker: dict[str, Any],
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    artifact = output / marker["artifact"]
    episode = json.loads(artifact.read_text(encoding="utf-8"))
    calls = episode["call_ledger"]
    transport_attempts = sum(int(call["transport_attempt_count"]) for call in calls)
    transport_errors = sum(int(call["transport_error_count"]) for call in calls)
    invalid_calls = sum(not bool(call["valid"]) for call in calls)
    initial_calls = sum(int(call["attempt"]) == 0 for call in calls)
    repair_calls = sum(int(call["attempt"]) > 0 for call in calls)
    gates = {
        "replay_hash_match": bool(episode["replay_hash_match"]),
        "debug_replay_valid": bool(marker["debug_replay_valid"]),
        "formed_true_feasible_team": (
            int(episode["analysis_only"]["true_feasible_locked_tasks"]) >= 1
        ),
        "invalid_call_rate_at_most_25_percent": (
            invalid_calls / len(calls) <= 0.25 if calls else False
        ),
        "semantic_repair_rate_at_most_25_percent": (
            repair_calls / initial_calls <= 0.25 if initial_calls else False
        ),
        "transport_error_rate_at_most_5_percent": (
            transport_errors / transport_attempts <= 0.05 if transport_attempts else False
        ),
        "no_budget_exhaustion": int(marker["budget_exhausted_decisions"]) == 0,
    }
    payload = {
        "schema_version": "alem-dice-e2b-canary-gate-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "cell": {
            "seed": marker["seed"],
            "family": marker["family"],
            "method": marker["method"],
        },
        "gates": gates,
        "diagnostics": {
            "logical_calls": len(calls),
            "initial_calls": initial_calls,
            "semantic_repair_calls": repair_calls,
            "invalid_calls": invalid_calls,
            "transport_attempts": transport_attempts,
            "transport_errors": transport_errors,
            "executed_acting_rounds": episode["early_stop"]["executed_acting_rounds"],
            "early_stop_reason": episode["early_stop"]["reason"],
        },
        "marker_sha256": sha256_text(canonical_json(marker)),
        "config_sha256": config_sha256,
        "source_hashes": source_hashes,
    }
    atomic_write_json(output / "canary_gate.json", payload)
    return payload


def _load_passing_canary_gate(
    output: Path,
    marker: dict[str, Any],
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, Any] | None:
    path = output / "canary_gate.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("status") != "pass"
        or payload.get("config_sha256") != config_sha256
        or payload.get("source_hashes") != source_hashes
        or payload.get("marker_sha256") != sha256_text(canonical_json(marker))
    ):
        return None
    return payload


def _protocol_config(
    *,
    logical_call_cap: int,
    provider_attempt_cap: int,
    token_exposure_cap: int,
) -> dict[str, Any]:
    estimate = estimate_campaign(
        seed_count=len(FROZEN_SEEDS),
        family_count=len(FROZEN_FAMILIES),
        method_count=len(FROZEN_METHODS),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "seeds": list(FROZEN_SEEDS),
        "scenario_families": [family.value for family in FROZEN_FAMILIES],
        "methods": [method.value for method in FROZEN_METHODS],
        "policy": "public_sweep",
        "rounds": DEFAULT_ROUNDS,
        "agent_count": 6,
        "model_id": DEFAULT_MODEL,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "max_prompt_bytes": DEFAULT_MAX_PROMPT_BYTES,
        "max_semantic_repairs": DEFAULT_SEMANTIC_REPAIRS,
        "max_transport_retries": DEFAULT_TRANSPORT_RETRIES,
        "stall_rounds": DEFAULT_STALL_ROUNDS,
        "selectors": {
            RecruitmentMethod.OPEN_VOLUNTEER.value: "joint_exact_allocation",
            RecruitmentMethod.MUTUAL_NOMINATION.value: "native_mutual_reciprocal",
        },
        "hard_caps": {
            "logical_calls": logical_call_cap,
            "provider_attempts": provider_attempt_cap,
            "provider_attempt_token_exposure": token_exposure_cap,
        },
        "estimate": estimate,
    }


def _client_factory(config: ScreenConfig):
    client_config = SimpleNamespace(
        client_name="openai_responses",
        model_id=config.model_id,
        base_url=None,
        timeout=180,
        generate_kwargs={
            "max_output_tokens": config.max_output_tokens,
            "reasoning_effort": config.reasoning_effort,
            "prompt_cache_key": f"alem-e2b-v1-{config.method.value}",
            "prompt_cache_traffic_shards": 6,
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "prompt_cache_retention": "24h",
            "store": False,
        },
        max_retries=config.max_transport_retries,
        delay=1,
        alternate_roles=False,
        enable_thinking=False,
    )
    create = create_llm_client(client_config)
    return lambda agent_id: create()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-hosted",
        action="store_true",
        help="Make the preregistered hosted Responses calls; omitted means estimate-only.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episode-workers", type=int, default=3)
    parser.add_argument("--stage", choices=("canary", "full"), default="canary")
    parser.add_argument(
        "--max-logical-calls",
        type=int,
        default=DEFAULT_LOGICAL_CALL_CAP,
    )
    parser.add_argument(
        "--max-provider-attempts",
        type=int,
        default=DEFAULT_PROVIDER_ATTEMPT_CAP,
    )
    parser.add_argument(
        "--max-token-exposure",
        type=int,
        default=DEFAULT_TOKEN_EXPOSURE_CAP,
        help="Hard provider-attempt token reservation, using one token per prompt byte.",
    )
    return parser.parse_args()


def _launch_projection(episode_count: int) -> dict[str, int]:
    """Projection with the canary's maximum promotable 25% repair rate."""

    initial_calls = episode_count * DEFAULT_ROUNDS * 6
    repair_allowance = (initial_calls + 3) // 4
    logical_calls = initial_calls + repair_allowance
    provider_attempts = logical_calls * (1 + DEFAULT_TRANSPORT_RETRIES)
    token_exposure = provider_attempts * (DEFAULT_MAX_PROMPT_BYTES + DEFAULT_MAX_OUTPUT_TOKENS)
    return {
        "episodes": episode_count,
        "initial_logical_calls": initial_calls,
        "semantic_repair_allowance": repair_allowance,
        "logical_calls": logical_calls,
        "provider_attempts_reserved": provider_attempts,
        "provider_attempt_token_exposure": token_exposure,
    }


def _caps_conflicts(
    *,
    protocol: dict[str, Any],
    projection: dict[str, int],
    logical_used: int = 0,
    provider_attempts_reserved: int = 0,
    tokens_reserved: int = 0,
) -> list[str]:
    caps = protocol["hard_caps"]
    conflicts = []
    checks = (
        ("logical_calls", logical_used + projection["logical_calls"]),
        (
            "provider_attempts",
            provider_attempts_reserved + projection["provider_attempts_reserved"],
        ),
        (
            "provider_attempt_token_exposure",
            tokens_reserved + projection["provider_attempt_token_exposure"],
        ),
    )
    for name, projected_total in checks:
        if projected_total > int(caps[name]):
            conflicts.append(f"{name}: projected {projected_total} > hard cap {caps[name]}")
    return conflicts


def _print_estimate(
    protocol: dict[str, Any],
    *,
    stage: str,
    projection: dict[str, int],
    execute_hosted: bool,
) -> None:
    estimate = protocol["estimate"]
    print(
        "E2b PRELAUNCH ESTIMATE — hosted execution requested"
        if execute_hosted
        else "E2b DRY RUN — no provider calls"
    )
    print(
        f"{estimate['episodes']} episodes = {len(FROZEN_SEEDS)} seeds × "
        f"{len(FROZEN_FAMILIES)} families × {len(FROZEN_METHODS)} methods"
    )
    print(
        "Selectors: "
        + ", ".join(
            f"{method}={selector}" for method, selector in sorted(protocol["selectors"].items())
        )
    )
    print(
        f"Maximum logical calls: {estimate['max_logical_calls']} "
        f"({estimate['initial_logical_calls']} initial + "
        f"{estimate['semantic_repair_calls']} repairs)"
    )
    print(f"Maximum provider attempts: {estimate['max_provider_attempts']}")
    print(
        "Maximum recorded successful-call usage: "
        f"{estimate['max_recorded_input_tokens']} input + "
        f"{estimate['max_recorded_output_tokens']} output = "
        f"{estimate['max_recorded_total_tokens']} total"
    )
    print(
        "Maximum provider-attempt exposure if retries are also billable: "
        f"{estimate['max_provider_input_token_exposure']} input + "
        f"{estimate['max_provider_output_token_exposure']} output = "
        f"{estimate['max_provider_total_token_exposure']} total"
    )
    print(
        "The input ceiling charges one token per permitted prompt byte and is "
        "therefore deliberately conservative."
    )
    print(
        f"{stage.capitalize()} launch projection (25% repair allowance): "
        f"{projection['episodes']} episode(s), {projection['logical_calls']} logical "
        f"({projection['initial_logical_calls']} initial + "
        f"{projection['semantic_repair_allowance']} repair), "
        f"{projection['provider_attempts_reserved']} provider reservations, "
        f"{projection['provider_attempt_token_exposure']} token exposure"
    )
    caps = protocol["hard_caps"]
    print(
        "Hard campaign caps: "
        f"{caps['logical_calls']} logical, {caps['provider_attempts']} provider, "
        f"{caps['provider_attempt_token_exposure']} token exposure"
    )


def main() -> int:
    args = _parse_args()
    if args.episode_workers < 1:
        raise ValueError("--episode-workers must be positive")
    if (
        min(
            args.max_logical_calls,
            args.max_provider_attempts,
            args.max_token_exposure,
        )
        < 1
    ):
        raise ValueError("all hard campaign caps must be positive")
    all_jobs = tuple(
        (seed, family, method)
        for seed in FROZEN_SEEDS
        for family in FROZEN_FAMILIES
        for method in FROZEN_METHODS
    )
    jobs = (CANARY_CELL,) if args.stage == "canary" else all_jobs
    protocol = _protocol_config(
        logical_call_cap=args.max_logical_calls,
        provider_attempt_cap=args.max_provider_attempts,
        token_exposure_cap=args.max_token_exposure,
    )
    initial_projection = _launch_projection(len(jobs))
    _print_estimate(
        protocol,
        stage=args.stage,
        projection=initial_projection,
        execute_hosted=args.execute_hosted,
    )
    initial_conflicts = _caps_conflicts(
        protocol=protocol,
        projection=initial_projection,
    )
    if initial_conflicts:
        raise ValueError(
            "launch projection conflicts with hard caps before any provider call: "
            + "; ".join(initial_conflicts)
        )
    if not args.execute_hosted:
        return 0

    output = args.output.resolve()
    if args.stage == "full" and (not output.exists() or not args.resume):
        raise FileNotFoundError("the full stage requires the canary output and --resume")
    if args.stage == "canary" and output.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse {output}; pass --resume to verify atomic markers")
    source_status = _git("status", "--short")
    if source_status:
        raise RuntimeError(
            "hosted E2b requires a clean committed worktree; "
            "dry-run remains available without --execute-hosted"
        )
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes()
    config_sha256 = sha256_text(canonical_json(protocol))
    manifest_path = output / f"run_manifest_{args.stage}.json"
    if args.resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config_sha256:
            raise ValueError("resume config hash does not match the frozen protocol")
        if existing.get("source_hashes") != source_hashes:
            raise ValueError("resume source hashes do not match the original run")

    completed: list[dict[str, Any]] = []
    pending: list[tuple[int, ScenarioFamily, RecruitmentMethod]] = []
    for seed, family, method in jobs:
        marker = load_completed_marker(
            output,
            seed=seed,
            family=family,
            method=method,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
        )
        if marker is None:
            pending.append((seed, family, method))
        else:
            completed.append(marker)

    canary_marker = load_completed_marker(
        output,
        seed=CANARY_CELL[0],
        family=CANARY_CELL[1],
        method=CANARY_CELL[2],
        config_sha256=config_sha256,
        source_hashes=source_hashes,
    )
    if args.stage == "full" and (
        canary_marker is None
        or _load_passing_canary_gate(
            output,
            canary_marker,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
        )
        is None
    ):
        raise RuntimeError(
            "full E2b refused before any call: a matching passing canary gate is required"
        )

    logical_used = sum(int(marker["logical_calls"]) for marker in completed)
    provider_reserved = sum(int(marker["provider_attempts_reserved"]) for marker in completed)
    tokens_reserved = sum(int(marker["tokens_reserved"]) for marker in completed)
    pending_projection = _launch_projection(len(pending))
    resume_conflicts = _caps_conflicts(
        protocol=protocol,
        projection=pending_projection,
        logical_used=logical_used,
        provider_attempts_reserved=provider_reserved,
        tokens_reserved=tokens_reserved,
    )
    if resume_conflicts:
        raise ValueError(
            "remaining launch projection conflicts with hard caps before any "
            "provider call: " + "; ".join(resume_conflicts)
        )
    caps = protocol["hard_caps"]
    campaign_budget = CampaignBudget(
        logical_limit=int(caps["logical_calls"]),
        provider_attempt_limit=int(caps["provider_attempts"]),
        token_limit=int(caps["provider_attempt_token_exposure"]),
        logical_used=logical_used,
        provider_attempts_reserved=provider_reserved,
        tokens_reserved=tokens_reserved,
    )
    manifest = {
        "schema_version": "alem-dice-e2b-campaign-v1",
        "stage": args.stage,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "stage_launch_projection": pending_projection,
        "config_sha256": config_sha256,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_status": source_status,
        "source_hashes": source_hashes,
        "python": platform.python_version(),
        "episode_workers": min(args.episode_workers, max(1, len(pending))),
        "completed_at_launch": len(completed),
    }
    atomic_write_json(manifest_path, manifest)

    def run_job(job: tuple[int, ScenarioFamily, RecruitmentMethod]) -> dict[str, Any]:
        seed, family, method = job
        scenario = generate_scenario(family, seed)
        config = ScreenConfig(method=method)
        episode = run_llm_recruitment_episode(
            scenario,
            config,
            client_factory=_client_factory(config),
            campaign_budget=campaign_budget,
        )
        return persist_completed_episode(
            output,
            episode,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
        )

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(args.episode_workers, max(1, len(pending)))) as pool:
        futures = {pool.submit(run_job, job): job for job in pending}
        for future in as_completed(futures):
            seed, family, method = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "seed": str(seed),
                        "family": family.value,
                        "method": method.value,
                        "error_type": type(exc).__name__,
                        "error_sha256": sha256_text(str(exc)),
                    }
                )

    manifest.update(
        {
            "status": (
                "complete"
                if not failures
                and len(completed) == len(jobs)
                and all(int(marker["budget_exhausted_decisions"]) == 0 for marker in completed)
                else "failed"
            ),
            "finished_at": datetime.now(UTC).isoformat(),
            "completed_episodes": len(completed),
            "expected_episodes": len(jobs),
            "failed_episodes": failures,
            "campaign_budget": campaign_budget.snapshot(),
            "markers": sorted(
                completed,
                key=lambda value: (
                    value["seed"],
                    value["family"],
                    value["method"],
                ),
            ),
        }
    )
    if args.stage == "canary" and manifest["status"] == "complete":
        canary_marker = completed[0]
        canary_gate = _write_canary_gate(
            output,
            canary_marker,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
        )
        manifest["canary_gate"] = canary_gate
        if canary_gate["status"] != "pass":
            manifest["status"] = "canary_failed"
    atomic_write_json(manifest_path, manifest)
    if manifest["status"] != "complete":
        raise RuntimeError(
            f"E2b {args.stage} status={manifest['status']}: "
            f"{len(completed)}/{len(jobs)} complete, {len(failures)} failed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
