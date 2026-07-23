#!/usr/bin/env python3
"""Run the preregistered local DCP1 mechanism pilot with durable provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ARMS = ("free", "consensus", "roles", "cohesion", "integrated")


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ollama_model(host: str, name: str) -> dict:
    with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=10) as response:
        payload = json.load(response)
    model = next((item for item in payload.get("models", []) if item.get("name") == name), None)
    if model is None:
        raise RuntimeError(f"Ollama does not list required model {name!r}")
    return model


def append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--concise",
        action="store_true",
        help="disable internal/visible chain-of-thought and request tags only",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or f"coordination_pilot_{stamp}"
    output_root = root / "outputs" / "alem_eval" / label
    output_root.mkdir(parents=True, exist_ok=False)
    event_path = output_root / "events.jsonl"
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=root)
    manifest = {
        "schema_version": "alem-dcp1-pilot-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "branch": run_text(["git", "branch", "--show-current"]),
        "source_commit": run_text(["git", "rev-parse", "HEAD"]),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "uv_lock_sha256": sha256(root / "uv.lock"),
        "model": ollama_model(args.host, args.model),
        "host": args.host,
        "arms": args.arms,
        "max_tokens": args.max_tokens,
        "steps": args.steps,
        "concise": args.concise,
        "output_root": str(output_root),
        "commands": [],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    failures = 0
    for arm in args.arms:
        arm_dir = output_root / arm
        command = [
            "uv", "run", "--extra", "baselines-llm", "--python", "3.12",
            "python", "baselines/llm/eval_alem.py",
            "provider=ollama_gemma4_31b",
            "experiment=coordination_local_smoke",
            f"coordination={arm}",
            f"eval.resume_from={arm_dir}",
            f"clients.0.base_url={args.host}", f"clients.1.base_url={args.host}",
            f"clients.2.base_url={args.host}",
            f"clients.0.model_id={args.model}", f"clients.1.model_id={args.model}",
            f"clients.2.model_id={args.model}",
            "clients.0.generate_kwargs.temperature=0.0",
            "clients.1.generate_kwargs.temperature=0.0",
            "clients.2.generate_kwargs.temperature=0.0",
            f"clients.0.generate_kwargs.max_tokens={args.max_tokens}",
            f"clients.1.generate_kwargs.max_tokens={args.max_tokens}",
            f"clients.2.generate_kwargs.max_tokens={args.max_tokens}",
            "WANDB_MODE=disabled",
            f"eval.max_steps_per_episode={args.steps}",
        ]
        if args.concise:
            command.extend(
                [
                    "agent.reasoning=false",
                    "agent.use_cot=false",
                    "agent.remember_cot=false",
                ]
            )
        started = datetime.now(UTC).isoformat()
        append_event(event_path, {"event": "arm_started", "arm": arm, "at": started, "command": command})
        log_path = output_root / f"{arm}.console.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        append_event(event_path, {
            "event": "arm_finished", "arm": arm,
            "at": datetime.now(UTC).isoformat(), "exit_code": result.returncode,
            "console_log": str(log_path),
        })
        manifest["commands"].append({"arm": arm, "argv": command, "exit_code": result.returncode})
        failures += result.returncode != 0
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    append_event(event_path, {
        "event": "study_finished", "at": datetime.now(UTC).isoformat(),
        "failed_arms": failures,
    })
    print(output_root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
