#!/usr/bin/env python3
"""Resolve and deploy a selectable Gemma 4 profile through Ray Serve LLM."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ray_serve"


@dataclass(frozen=True)
class ModelProfile:
    """Memory-safe defaults for one supported checkpoint family.

    ``minimum_bf16_l4_shards`` is a conservative admission guard, not a Ray
    scheduler hint. It prevents an accidental deployment that cannot fit on
    standard 24 GB L4 devices.
    """

    key: str
    model_id: str
    model_source: str
    default_tensor_parallel_size: int
    minimum_bf16_l4_shards: int
    max_num_seqs: int


PROFILES = {
    "e4b": ModelProfile(
        key="e4b",
        model_id="gemma-4-E4B-it",
        model_source="google/gemma-4-E4B-it",
        default_tensor_parallel_size=1,
        minimum_bf16_l4_shards=1,
        max_num_seqs=16,
    ),
    "26b-a4b": ModelProfile(
        key="26b-a4b",
        model_id="gemma-4-26B-A4B-it",
        model_source="google/gemma-4-26B-A4B-it",
        default_tensor_parallel_size=4,
        minimum_bf16_l4_shards=4,
        max_num_seqs=16,
    ),
}


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=sorted(PROFILES), required=True)
    result.add_argument(
        "--gpus",
        type=positive_integer,
        required=True,
        help="total approved Ray GPU resources available for this deployment",
    )
    result.add_argument(
        "--tensor-parallel-size",
        type=positive_integer,
        help="GPUs per model replica; defaults to 1 for E4B and 4 for 26B/A4B",
    )
    result.add_argument(
        "--replicas",
        type=positive_integer,
        help="model replicas; defaults to floor(gpus / tensor-parallel-size)",
    )
    result.add_argument(
        "--model-source",
        help="override the Hugging Face ID or use an absolute/shared model path",
    )
    result.add_argument(
        "--served-model-name",
        help="override the model ID exposed by Ray Serve's /v1 endpoint",
    )
    result.add_argument("--accelerator-type", default="L4")
    result.add_argument("--max-model-len", type=positive_integer, default=32768)
    result.add_argument("--max-num-seqs", type=positive_integer)
    result.add_argument("--gpu-memory-utilization", type=probability, default=0.90)
    result.add_argument(
        "--allow-unsafe-bf16-fit",
        action="store_true",
        help="allow fewer L4 shards than the selected BF16 profile requires",
    )
    result.add_argument(
        "--quantized-checkpoint",
        action="store_true",
        help="declare that --model-source is quantized and may use fewer shards",
    )
    result.add_argument(
        "--output",
        type=Path,
        help="resolved Serve YAML path (default: outputs/ray_serve/<profile>.yaml)",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="write and print the resolved config without calling `serve deploy`",
    )
    return result


def resolve(args: argparse.Namespace) -> tuple[dict[str, object], Path, str]:
    """Validate resource arithmetic and produce a deterministic Serve config."""

    profile = PROFILES[args.model]
    tensor_parallel_size = args.tensor_parallel_size or profile.default_tensor_parallel_size
    replicas = args.replicas or args.gpus // tensor_parallel_size
    if replicas < 1:
        raise ValueError(f"{args.gpus} GPU(s) cannot host a TP={tensor_parallel_size} replica")

    required_gpus = replicas * tensor_parallel_size
    if required_gpus > args.gpus:
        raise ValueError(
            f"{replicas} replica(s) at TP={tensor_parallel_size} require "
            f"{required_gpus} GPUs, but --gpus={args.gpus}"
        )

    if args.quantized_checkpoint and not args.model_source:
        raise ValueError("--quantized-checkpoint requires an explicit --model-source")

    model_source = args.model_source or profile.model_source
    if (
        tensor_parallel_size < profile.minimum_bf16_l4_shards
        and not args.quantized_checkpoint
        and not args.allow_unsafe_bf16_fit
    ):
        raise ValueError(
            f"{profile.model_id} BF16 requires at least "
            f"TP={profile.minimum_bf16_l4_shards} on standard 24 GB L4 GPUs; "
            "use more shards, declare a quantized local checkpoint with "
            "--quantized-checkpoint, or explicitly pass --allow-unsafe-bf16-fit"
        )

    served_model_name = args.served_model_name or profile.model_id
    max_num_seqs = args.max_num_seqs or profile.max_num_seqs
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"gemma4_{profile.key}.yaml"
    output_path = output_path.expanduser().resolve()

    config: dict[str, object] = {
        "proxy_location": "HeadOnly",
        "http_options": {
            "host": "127.0.0.1",
            "port": 8000,
            "request_timeout_s": 900,
            "keep_alive_timeout_s": 30,
        },
        "logging_config": {
            "log_level": "INFO",
            "encoding": "TEXT",
            "enable_access_log": False,
        },
        "applications": [
            {
                "name": "gemma4_selected",
                "route_prefix": "/",
                "import_path": "ray.serve.llm:build_openai_app",
                "args": {
                    "llm_configs": [
                        {
                            "model_loading_config": {
                                "model_id": served_model_name,
                                "model_source": model_source,
                            },
                            "accelerator_type": args.accelerator_type,
                            "engine_kwargs": {
                                "tensor_parallel_size": tensor_parallel_size,
                                "dtype": "bfloat16",
                                "max_model_len": args.max_model_len,
                                "max_num_seqs": max_num_seqs,
                                "gpu_memory_utilization": args.gpu_memory_utilization,
                                "enable_prefix_caching": True,
                                "limit_mm_per_prompt": {},
                            },
                            "deployment_config": {
                                # Ray schedules each replica as one placement
                                # group containing ``tensor_parallel_size`` GPU
                                # workers; independent replicas provide request
                                # concurrency.
                                "num_replicas": replicas,
                                "max_ongoing_requests": max_num_seqs,
                                "health_check_period_s": 10,
                                "health_check_timeout_s": 30,
                                "graceful_shutdown_timeout_s": 900,
                            },
                        }
                    ]
                },
            }
        ],
    }
    summary = (
        f"model={served_model_name} source={model_source} replicas={replicas} "
        f"TP={tensor_parallel_size} GPUs={required_gpus}/{args.gpus}"
    )
    return config, output_path, summary


def main() -> int:
    args = parser().parse_args()
    try:
        config, output_path, summary = resolve(args)
    except ValueError as exc:
        parser().error(str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Resolved Ray Serve profile: {summary}")
    print(f"Config: {output_path}")

    model_id = config["applications"][0]["args"]["llm_configs"][0]["model_loading_config"][
        "model_id"
    ]
    print(
        "Evaluation model selector: "
        f"ALEM_VLLM_MODEL={model_id} "
        "ALEM_VLLM_BASE_URL=http://127.0.0.1:8000/v1"
    )

    if args.dry_run:
        print(output_path.read_text(encoding="utf-8"), end="")
        return 0

    serve_bin = shutil.which("serve")
    if serve_bin is None:
        candidate = Path(
            os.environ.get(
                "RAY_SERVE_BIN",
                str(Path.home() / ".venvs" / "ray-llm" / "bin" / "serve"),
            )
        )
        if not candidate.is_file():
            print(
                "ERROR: Ray Serve CLI not found; activate ~/.venvs/ray-llm or set RAY_SERVE_BIN",
                file=sys.stderr,
            )
            return 2
        serve_bin = str(candidate)

    completed = subprocess.run([serve_bin, "deploy", str(output_path)], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
