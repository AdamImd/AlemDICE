#!/usr/bin/env python3
"""Launch a 32-agent, 1,000-step evaluation against vLLM or Ray Serve."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "baselines" / "llm" / "eval_alem.py"
EXPERIMENT = "gemma4_e4b_vllm_32x1000"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gemma-4-E4B-it"
DEFAULT_SEED = 14100
DEFAULT_STEPS = 1000


def _normalized_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("base URL must start with http:// or https://")
    if not base_url.endswith("/v1"):
        raise argparse.ArgumentTypeError("base URL must include the vLLM /v1 API prefix")
    return base_url


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _available_models(base_url: str, timeout: float = 15.0) -> list[str]:
    """Return model IDs from the read-only OpenAI-compatible discovery route."""

    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": "Bearer EMPTY"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not query {base_url}/models: {exc}") from exc

    records = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = sorted(
        {str(record["id"]) for record in records if isinstance(record, dict) and record.get("id")}
    )
    if not model_ids:
        raise RuntimeError(f"{base_url}/models returned no model IDs")
    return model_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        type=_normalized_base_url,
        default=os.environ.get("ALEM_VLLM_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible inference API root (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ALEM_VLLM_MODEL", DEFAULT_MODEL),
        help=f"served model ID reported by /v1/models (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=_positive_integer, default=DEFAULT_STEPS)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="parent directory for runs (defaults to a model-specific directory)",
    )
    parser.add_argument(
        "--run-name",
        help="optional stable run name; omit for a timestamped, collision-free run",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="resume an existing resolved run directory instead of creating a new run",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the read-only GET /v1/models endpoint check",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved evaluator command without launching the episode",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="additional Hydra overrides applied after the fixed experiment settings",
    )
    return parser


def _evaluator_command(args: argparse.Namespace) -> list[str]:
    """Build the exact subprocess argv without shell interpolation."""

    command = [
        sys.executable,
        str(EVALUATOR),
        f"experiment={EXPERIMENT}",
        f"EVAL_SEED={args.seed}",
        f"experiment.seeds=[{args.seed}]",
        "eval.num_episodes.alem=1",
        f"eval.max_steps_per_episode={args.steps}",
        f"alem.max_timesteps={args.steps}",
        f"eval.output_dir={args.output_dir}",
        f"WANDB_MODE={args.wandb_mode}",
    ]
    if args.resume_from is not None:
        command.append(f"eval.resume_from={args.resume_from.expanduser().resolve()}")
    elif args.run_name:
        command.append(f"eval.run_name={args.run_name}")
    command.extend(override for override in args.overrides if override != "--")
    return command


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.output_dir is None:
        if args.model == DEFAULT_MODEL:
            args.output_dir = "outputs/alem_eval/gemma4_e4b_vllm"
        else:
            model_slug = "".join(
                character.lower() if character.isalnum() else "_" for character in args.model
            ).strip("_")
            args.output_dir = f"outputs/alem_eval/gemma4_ray/{model_slug}"

    if not args.skip_preflight:
        try:
            models = _available_models(args.base_url)
        except RuntimeError as exc:
            parser.error(str(exc))
        print(f"Inference endpoint: {args.base_url}")
        print(f"Served models: {', '.join(models)}")
        if args.model not in models:
            parser.error(
                f"requested model {args.model!r} is not served; "
                f"choose one with --model ({', '.join(models)})"
            )

    environment = os.environ.copy()
    environment["ALEM_VLLM_BASE_URL"] = args.base_url
    environment["ALEM_VLLM_MODEL"] = args.model
    # Keep JAX simulation off the GPUs reserved for Ray Serve and vLLM. This is
    # deliberately setdefault so an operator may explicitly choose otherwise.
    environment.setdefault("JAX_PLATFORMS", "cpu")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("PYTHONUNBUFFERED", "1")

    command = _evaluator_command(args)
    print(f"Experiment: 32 agents, 1 episode, {args.steps} max steps, seed {args.seed}")
    print("Maximum model concurrency: 32 requests")
    print(f"Model: {args.model}")
    print(f"Command: {shlex.join(command)}")
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
