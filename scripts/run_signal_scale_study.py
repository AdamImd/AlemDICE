#!/usr/bin/env python3
"""Run the preregistered three-arm signal study with three parallel seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ARM_ORDER = ("free_concise", "cohesion_concise", "free_thinking")
ARMS = {
    "free_thinking": {
        "coordination": "free",
        "reasoning": "true",
        "use_cot": "true",
        "remember_cot": "true",
        "max_tokens": 2048,
    },
    "free_concise": {
        "coordination": "free",
        "reasoning": "false",
        "use_cot": "false",
        "remember_cot": "false",
        "max_tokens": 2048,
    },
    "cohesion_concise": {
        "coordination": "cohesion",
        "reasoning": "false",
        "use_cot": "false",
        "remember_cot": "false",
        "max_tokens": 2048,
    },
}
ENDPOINTS = (
    "http://127.0.0.1:11435",
    "http://127.0.0.1:11436",
    "http://127.0.0.1:11437",
)
MODEL = "gemma4:31b"
SEEDS = (9999, 10000, 10001)
STEPS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the fixed output directory and retry incomplete episodes",
    )
    parser.add_argument(
        "--max-arm-attempts",
        type=int,
        default=3,
        help="maximum evaluator invocations per incomplete arm in this launch",
    )
    return parser.parse_args()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def request_json(url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def model_record(endpoint: str) -> dict:
    payload = request_json(f"{endpoint}/api/tags")
    match = next(
        (model for model in payload.get("models", []) if model.get("name") == MODEL),
        None,
    )
    if match is None:
        raise RuntimeError(f"{endpoint} does not expose {MODEL}")
    return match


def warm_request(endpoint: str, slot: int) -> dict:
    started = time.monotonic()
    response = request_json(
        f"{endpoint}/api/chat",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply exactly OK."}],
            "stream": False,
            "keep_alive": "24h",
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_ctx": 12288,
                "num_predict": 8,
            },
        },
        timeout=420,
    )
    return {
        "endpoint": endpoint,
        "concurrency_slot": slot,
        "wall_seconds": time.monotonic() - started,
        "done": response.get("done"),
        "done_reason": response.get("done_reason"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_duration_ns": response.get("eval_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
    }


def concurrency_preflight() -> tuple[list[dict], list[dict]]:
    """Warm each server and verify that nine simultaneous requests complete."""

    requests = [(endpoint, slot) for endpoint in ENDPOINTS for slot in range(3)]
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        warmups = list(executor.map(lambda args: warm_request(*args), requests))
    process_records = [request_json(f"{endpoint}/api/ps") for endpoint in ENDPOINTS]
    return warmups, process_records


def episode_complete(path: Path, expected_seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(result, dict)
        and result.get("schema_version") == "alem-dice-episode-v1"
        and result.get("artifact_status") == "complete"
        and not result.get("error")
        and bool(result.get("termination_reason"))
        and result.get("seed") == expected_seed
        and result.get("early_stop_reason")
        != "consecutive_length_incomplete_responses"
    )


def arm_complete(output_root: Path, arm_name: str) -> bool:
    task_dir = output_root / arm_name / "alem" / "default"
    return all(
        episode_complete(task_dir / f"default_run_{index:02d}.json", seed)
        for index, seed in enumerate(SEEDS)
    )


def arm_command(root: Path, output_root: Path, arm_name: str) -> list[str]:
    arm = ARMS[arm_name]
    command = [
        "uv",
        "run",
        "--extra",
        "baselines-llm",
        "--python",
        "3.12",
        "python",
        "baselines/llm/eval_alem.py",
        "provider=ollama_gemma4_31b",
        "experiment=coordination_local_smoke",
        f"coordination={arm['coordination']}",
        f"eval.resume_from={output_root / arm_name}",
        "eval.num_episodes.alem=3",
        "eval.num_workers=3",
        f"eval.max_steps_per_episode={STEPS}",
        "experiment.seeds=[9999,10000,10001]",
        f"EVAL_SEED={SEEDS[0]}",
        "WANDB_MODE=disabled",
        f"agent.reasoning={arm['reasoning']}",
        f"agent.use_cot={arm['use_cot']}",
        f"agent.remember_cot={arm['remember_cot']}",
    ]
    for index, endpoint in enumerate(ENDPOINTS):
        command.extend(
            [
                f"clients.{index}.base_url={endpoint}",
                f"clients.{index}.generate_kwargs.temperature=0.0",
                f"clients.{index}.generate_kwargs.max_tokens={arm['max_tokens']}",
            ]
        )
    return command


def new_manifest(root: Path, models: list[dict]) -> dict:
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    return {
        "schema_version": "alem-signal-scale-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "branch": git_output(root, "branch", "--show-current"),
        "source_commit": git_output(root, "rev-parse", "HEAD"),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "models": models,
        "model": MODEL,
        "endpoints": ENDPOINTS,
        "seeds": SEEDS,
        "steps": STEPS,
        "parallel_workers": 3,
        "maximum_in_flight_requests": 9,
        "arm_order": ARM_ORDER,
        "arms": ARMS,
        "launches": [],
        "warmups": [],
        "commands": [],
    }


def validate_resume(root: Path, manifest: dict, models: list[dict]) -> None:
    expected = {
        "schema_version": "alem-signal-scale-v2",
        "source_commit": git_output(root, "rev-parse", "HEAD"),
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "model": MODEL,
        "endpoints": list(ENDPOINTS),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "arm_order": list(ARM_ORDER),
        "arms": ARMS,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    recorded_digests = {model["digest"] for model in manifest.get("models", [])}
    current_digests = {model["digest"] for model in models}
    if recorded_digests != current_digests:
        mismatches.append("model digests")
    if mismatches:
        raise RuntimeError("Resume manifest mismatch: " + ", ".join(mismatches))


def main() -> int:
    args = parse_args()
    if args.max_arm_attempts < 1:
        raise ValueError("--max-arm-attempts must be positive")

    root = Path(__file__).resolve().parents[1]
    output_root = root / "outputs" / "alem_eval" / "signal_scale_3seed_200"
    manifest_path = output_root / "manifest.json"
    events_path = output_root / "events.jsonl"
    models = [model_record(endpoint) for endpoint in ENDPOINTS]
    digests = {model["digest"] for model in models}
    if len(digests) != 1:
        raise RuntimeError(f"Ollama endpoints disagree on model digest: {digests}")

    if args.resume:
        if not manifest_path.is_file():
            raise RuntimeError(f"Cannot resume: missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume(root, manifest, models)
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        manifest = new_manifest(root, models)

    launch = {
        "started_at": datetime.now(UTC).isoformat(),
        "resume": args.resume,
        "max_arm_attempts": args.max_arm_attempts,
    }
    manifest["launches"].append(launch)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_jsonl(events_path, {"event": "preflight_started", **launch})
    warmups, process_records = concurrency_preflight()
    manifest["warmups"].append(
        {
            "at": datetime.now(UTC).isoformat(),
            "requests": warmups,
            "process_records": process_records,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_jsonl(
        events_path,
        {
            "event": "preflight_finished",
            "at": datetime.now(UTC).isoformat(),
            "requests": len(warmups),
        },
    )

    failed_arms: list[str] = []
    for arm_name in ARM_ORDER:
        if arm_complete(output_root, arm_name):
            append_jsonl(
                events_path,
                {
                    "event": "arm_skipped_complete",
                    "arm": arm_name,
                    "at": datetime.now(UTC).isoformat(),
                },
            )
            continue

        command = arm_command(root, output_root, arm_name)
        existing_attempts = [
            int(path.name.split(".attempt_")[1].split(".")[0])
            for path in output_root.glob(f"{arm_name}.attempt_*.console.log")
        ]
        previous_attempt = max(existing_attempts, default=0)
        for launch_attempt in range(1, args.max_arm_attempts + 1):
            attempt = previous_attempt + launch_attempt
            append_jsonl(
                events_path,
                {
                    "event": "arm_started",
                    "arm": arm_name,
                    "attempt": attempt,
                    "launch_attempt": launch_attempt,
                    "at": datetime.now(UTC).isoformat(),
                    "command": command,
                },
            )
            console_path = output_root / f"{arm_name}.attempt_{attempt:02d}.console.log"
            with console_path.open("w", encoding="utf-8") as console:
                result = subprocess.run(
                    command,
                    cwd=root,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                )
            complete = arm_complete(output_root, arm_name)
            entry = {
                "arm": arm_name,
                "attempt": attempt,
                "argv": command,
                "exit_code": result.returncode,
                "complete": complete,
                "console_log": str(console_path),
            }
            manifest["commands"].append(entry)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            append_jsonl(
                events_path,
                {
                    "event": "arm_finished",
                    "arm": arm_name,
                    "attempt": attempt,
                    "launch_attempt": launch_attempt,
                    "at": datetime.now(UTC).isoformat(),
                    "exit_code": result.returncode,
                    "complete": complete,
                },
            )
            if complete:
                break
        if not arm_complete(output_root, arm_name):
            failed_arms.append(arm_name)

    launch["finished_at"] = datetime.now(UTC).isoformat()
    launch["failed_arms"] = failed_arms
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_jsonl(
        events_path,
        {
            "event": "study_finished",
            "at": launch["finished_at"],
            "failed_arms": failed_arms,
        },
    )
    print(output_root)
    return int(bool(failed_arms))


if __name__ == "__main__":
    raise SystemExit(main())
