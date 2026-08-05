#!/usr/bin/env python3
"""Safely manage and evaluate against a four-GPU rootless Ollama fleet.

The script deliberately manages only processes whose PID and Linux start time
it recorded itself. Existing Ollama services (including ports 11434--11437) are
never inspected as ownership candidates and are never stopped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "outputs" / "kingpin_ollama"
STATE_PATH = RUNTIME_DIR / "fleet.json"
DEFAULT_PORTS = (11534, 11535, 11536, 11537)
PROFILE = "gemma4_31b_ollama_32x25"


class FleetError(RuntimeError):
    """An unsafe or invalid fleet state."""


@dataclass(frozen=True)
class ComputeProcess:
    pid: int
    gpu_uuid: str
    name: str
    memory_mib: int


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def _nvidia_csv(query: str) -> list[list[str]]:
    result = _run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"])
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def gpu_inventory() -> list[dict[str, Any]]:
    rows = _nvidia_csv("gpu=index,uuid,name,memory.total")
    inventory = [
        {
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(memory),
        }
        for index, uuid, name, memory in rows
    ]
    return sorted(inventory, key=lambda item: item["index"])


def compute_processes() -> list[ComputeProcess]:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise FleetError(f"cannot query GPU compute processes: {result.stderr.strip()}")
    processes: list[ComputeProcess] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or fields[0] in {"", "[Not Found]"}:
            continue
        try:
            processes.append(ComputeProcess(int(fields[0]), fields[1], fields[2], int(fields[3])))
        except ValueError as exc:
            raise FleetError(f"unexpected nvidia-smi compute row: {line!r}") from exc
    return processes


def process_start_time(pid: int) -> int | None:
    try:
        # The comm field may contain spaces and parentheses, so split after its
        # final ')' before selecting /proc stat field 22 (starttime).
        suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(suffix[19])
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return None


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def descendants(root_pids: set[int]) -> set[int]:
    children: dict[int, set[int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            pid = int(stat_path.parent.name)
            suffix = stat_path.read_text().rsplit(")", 1)[1].split()
            parent = int(suffix[1])
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
        children.setdefault(parent, set()).add(pid)
    found = set(root_pids)
    frontier = list(root_pids)
    while frontier:
        for child in children.get(frontier.pop(), ()):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def _load_state(*, required: bool = True) -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        if required:
            raise FleetError(f"no managed fleet state at {STATE_PATH}; run start first")
        return None
    try:
        state = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetError(f"invalid fleet state: {STATE_PATH}") from exc
    if state.get("schema_version") != 1 or not isinstance(state.get("servers"), list):
        raise FleetError(f"unsupported fleet state: {STATE_PATH}")
    return state


def _write_state(state: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE_PATH)


def _server_owned(server: dict[str, Any]) -> bool:
    pid = int(server["pid"])
    return process_start_time(pid) == int(
        server["start_time"]
    ) and "ollama serve" in process_cmdline(pid)


def _managed_roots(state: dict[str, Any]) -> set[int]:
    roots = set()
    for server in state["servers"]:
        if not _server_owned(server):
            raise FleetError(
                f"managed server PID identity changed for port {server['port']}; "
                "refusing to operate"
            )
        roots.add(int(server["pid"]))
    return roots


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _request_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 30
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise FleetError(f"request failed for {url}: {exc}") from exc
    if not isinstance(body, dict):
        raise FleetError(f"expected JSON object from {url}")
    return body


def _wait_ready(port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _request_json(f"http://127.0.0.1:{port}/api/version", timeout=2)
            return
        except FleetError:
            time.sleep(0.5)
    raise FleetError(f"Ollama on port {port} did not become ready within {timeout}s")


def _cleanup_started(servers: list[dict[str, Any]]) -> None:
    for server in servers:
        if _server_owned(server):
            try:
                os.killpg(int(server["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass


def start_fleet(args: argparse.Namespace) -> None:
    existing = _load_state(required=False)
    if existing:
        live = [server for server in existing["servers"] if _server_owned(server)]
        if live:
            raise FleetError("a managed fleet is already running; use status or stop")
        STATE_PATH.unlink(missing_ok=True)

    inventory = gpu_inventory()
    if len(inventory) < len(args.ports):
        raise FleetError(f"need {len(args.ports)} GPUs, found {len(inventory)}")
    busy = compute_processes()
    if busy:
        details = ", ".join(
            f"pid={item.pid} gpu={item.gpu_uuid} {item.name} {item.memory_mib}MiB" for item in busy
        )
        raise FleetError(f"GPU compute processes already exist: {details}")
    occupied = [port for port in args.ports if not _port_available(port)]
    if occupied:
        raise FleetError(f"ports already occupied: {occupied}")

    ollama = shutil.which(args.ollama_bin)
    if not ollama:
        raise FleetError(f"Ollama executable not found: {args.ollama_bin}")
    model_dir = Path(args.models).resolve()
    if not model_dir.is_dir() or not os.access(model_dir, os.R_OK | os.X_OK):
        raise FleetError(f"model directory is not readable: {model_dir}")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    servers: list[dict[str, Any]] = []
    try:
        for gpu, port in zip(inventory, args.ports, strict=True):
            log_path = RUNTIME_DIR / f"ollama-{port}.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                    "OLLAMA_HOST": f"127.0.0.1:{port}",
                    "OLLAMA_MODELS": str(model_dir),
                    "OLLAMA_KEEP_ALIVE": "-1",
                    "OLLAMA_MAX_LOADED_MODELS": "1",
                    "OLLAMA_NUM_PARALLEL": str(args.parallel),
                    "OLLAMA_CONTEXT_LENGTH": str(args.context_length),
                    "OLLAMA_LOAD_TIMEOUT": "10m",
                    "OLLAMA_NOPRUNE": "1",
                }
            )
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    [ollama, "serve"],
                    cwd=REPO_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            start_time = None
            for _ in range(20):
                start_time = process_start_time(process.pid)
                if start_time is not None:
                    break
                time.sleep(0.05)
            if start_time is None:
                raise FleetError(f"could not identify new Ollama PID {process.pid}")
            servers.append(
                {
                    "pid": process.pid,
                    "start_time": start_time,
                    "port": port,
                    "gpu_index": gpu["index"],
                    "gpu_uuid": gpu["uuid"],
                    "gpu_name": gpu["name"],
                    "log": str(log_path),
                }
            )
        for server in servers:
            _wait_ready(int(server["port"]))
    except Exception:
        _cleanup_started(servers)
        raise

    state = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "models_dir": str(model_dir),
        "parallel": args.parallel,
        "context_length": args.context_length,
        "servers": servers,
    }
    _write_state(state)
    print(f"Started {len(servers)} managed Ollama servers")
    status_fleet(argparse.Namespace())


def status_fleet(_: argparse.Namespace) -> None:
    state = _load_state()
    for server in state["servers"]:
        owned = _server_owned(server)
        print(
            f"port={server['port']} gpu={server['gpu_index']} uuid={server['gpu_uuid']} "
            f"pid={server['pid']} owned_live={str(owned).lower()} log={server['log']}"
        )
    managed = descendants(_managed_roots(state))
    processes = compute_processes()
    for item in processes:
        ownership = "managed" if item.pid in managed else "FOREIGN"
        print(
            f"compute pid={item.pid} gpu={item.gpu_uuid} memory={item.memory_mib}MiB "
            f"owner={ownership} name={item.name}"
        )


def stop_fleet(_: argparse.Namespace) -> None:
    state = _load_state()
    roots = _managed_roots(state)
    for pid in roots:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and any(process_start_time(pid) is not None for pid in roots):
        time.sleep(0.25)
    remaining = [pid for pid in roots if process_start_time(pid) is not None]
    if remaining:
        # Identity was validated immediately before TERM. Kill only those same
        # process groups if the graceful timeout elapsed.
        for server in state["servers"]:
            if int(server["pid"]) in remaining and _server_owned(server):
                os.killpg(int(server["pid"]), signal.SIGKILL)
    STATE_PATH.unlink(missing_ok=True)
    print("Stopped the managed Ollama fleet; unrelated services were untouched")


def _model_record(tags: dict[str, Any], model: str) -> dict[str, Any] | None:
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return item
    return None


def _preflight_once(state: dict[str, Any]) -> None:
    roots = _managed_roots(state)
    if compute_processes():
        raise FleetError("preflight must begin before any GPU model runner is loaded")

    digests = set()
    for server in state["servers"]:
        port = int(server["port"])
        _request_json(f"http://127.0.0.1:{port}/api/version")
        model = _model_record(_request_json(f"http://127.0.0.1:{port}/api/tags"), state["model"])
        if model is None:
            raise FleetError(f"{state['model']} is unavailable on port {port}")
        digests.add(model.get("digest"))
    if len(digests) != 1 or None in digests:
        raise FleetError(f"model digests differ across servers: {digests}")

    def warm(server: dict[str, Any]) -> tuple[int, float]:
        started = time.monotonic()
        _request_json(
            f"http://127.0.0.1:{server['port']}/api/chat",
            {
                "model": state["model"],
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {
                    "num_predict": 4,
                    "temperature": 0,
                    # Exercise the same KV-cache allocation as the evaluation,
                    # not merely a tiny connectivity prompt.
                    "num_ctx": int(state["context_length"]),
                },
            },
            timeout=600,
        )
        return int(server["port"]), time.monotonic() - started

    jobs = [server for server in state["servers"] for _ in range(8)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(warm, jobs))

    managed = descendants(roots)
    processes = compute_processes()
    foreign = [item for item in processes if item.pid not in managed]
    if foreign:
        raise FleetError(f"foreign GPU compute processes appeared: {foreign}")
    for server in state["servers"]:
        assigned = [item for item in processes if item.pid in descendants({int(server["pid"])})]
        if not assigned:
            raise FleetError(f"no GPU runner found below server on port {server['port']}")
        wrong = [item for item in assigned if item.gpu_uuid != server["gpu_uuid"]]
        if wrong:
            raise FleetError(f"server on port {server['port']} used an unassigned GPU: {wrong}")
        loaded = _request_json(f"http://127.0.0.1:{server['port']}/api/ps")
        if _model_record(loaded, state["model"]) is None:
            raise FleetError(f"model is not resident on port {server['port']}")
    slowest = max(duration for _, duration in results)
    print(
        f"Preflight passed: 32 concurrent requests, digest={next(iter(digests))}, "
        f"slowest={slowest:.1f}s"
    )


def preflight_fleet(args: argparse.Namespace) -> None:
    state = _load_state()
    try:
        _preflight_once(state)
    except FleetError:
        if not args.retry_parallel or int(state["parallel"]) <= args.retry_parallel:
            raise
        print(
            f"Preflight failed at OLLAMA_NUM_PARALLEL={state['parallel']}; "
            f"retrying once at {args.retry_parallel}",
            file=sys.stderr,
        )
        settings = argparse.Namespace(
            ports=tuple(int(item["port"]) for item in state["servers"]),
            ollama_bin="ollama",
            models=state["models_dir"],
            model=state["model"],
            parallel=args.retry_parallel,
            context_length=state["context_length"],
        )
        stop_fleet(argparse.Namespace())
        start_fleet(settings)
        _preflight_once(_load_state())


def _assert_no_foreign_compute(state: dict[str, Any]) -> None:
    managed = descendants(_managed_roots(state))
    foreign = [item for item in compute_processes() if item.pid not in managed]
    if foreign:
        raise FleetError(f"foreign GPU compute processes detected: {foreign}")


def _uv_executable() -> str:
    """Find uv in interactive PATH or its standard rootless install location."""

    configured = os.environ.get("ALEM_UV_BIN")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(found) if (found := shutil.which("uv")) else None,
        Path.home() / ".local" / "bin" / "uv",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise FleetError("uv was not found in PATH or ~/.local/bin; install uv or set ALEM_UV_BIN")


def evaluator_command(args: argparse.Namespace, *, smoke: bool) -> list[str]:
    steps = 1 if smoke else args.steps
    run_name = args.run_name or (
        f"smoke_n32_seed{args.seed}" if smoke else f"kingpin_n32_seed{args.seed}_{steps}"
    )
    command = [
        _uv_executable(),
        "run",
        "--extra",
        "baselines-llm",
        "--python",
        "3.12",
        "python",
        "baselines/llm/eval_alem.py",
        f"experiment={PROFILE}",
        f"EVAL_SEED={args.seed}",
        f"experiment.seeds=[{args.seed}]",
        f"eval.max_steps_per_episode={steps}",
        f"alem.max_timesteps={steps}",
        f"eval.run_name={run_name}",
        f"WANDB_MODE={args.wandb_mode}",
    ]
    if args.resume_from:
        command.append(f"eval.resume_from={Path(args.resume_from).resolve()}")
    if smoke:
        command.extend(f"clients.{index}.generate_kwargs.max_tokens=128" for index in range(32))
    return command


def run_evaluation(args: argparse.Namespace, *, smoke: bool) -> None:
    state = _load_state()
    _assert_no_foreign_compute(state)
    command = evaluator_command(args, smoke=smoke)
    print(" ".join(command))
    if args.dry_run:
        return
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "ALEM_OLLAMA_MODEL": state["model"],
        }
    )
    for index, server in enumerate(state["servers"]):
        environment[f"ALEM_OLLAMA_URL_{index}"] = f"http://127.0.0.1:{server['port']}"
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
    try:
        while process.poll() is None:
            time.sleep(5)
            _assert_no_foreign_compute(state)
    except BaseException:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
        raise
    if process.returncode:
        raise FleetError(f"evaluation exited with status {process.returncode}")
    print("Smoke evaluation completed" if smoke else "Evaluation completed")


def _add_eval_arguments(parser: argparse.ArgumentParser, *, default_steps: int) -> None:
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--seed", type=int, default=14100)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--dry-run", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="start four owned GPU-pinned servers")
    start.add_argument("--ports", type=int, nargs=4, default=DEFAULT_PORTS)
    start.add_argument("--ollama-bin", default="ollama")
    start.add_argument("--models", default="/home/ollama/models")
    start.add_argument("--model", default="gemma4:31b")
    start.add_argument("--parallel", type=int, default=8)
    start.add_argument("--context-length", type=int, default=12288)
    start.set_defaults(handler=start_fleet)

    status = subparsers.add_parser("status", help="show owned servers and GPU runners")
    status.set_defaults(handler=status_fleet)
    stop = subparsers.add_parser("stop", help="stop only the recorded owned servers")
    stop.set_defaults(handler=stop_fleet)
    preflight = subparsers.add_parser("preflight", help="load and stress all replicas")
    preflight.add_argument("--retry-parallel", type=int, default=4)
    preflight.set_defaults(handler=preflight_fleet)
    smoke = subparsers.add_parser("smoke", help="run one 32-agent simulator step")
    _add_eval_arguments(smoke, default_steps=1)
    smoke.set_defaults(handler=lambda args: run_evaluation(args, smoke=True))
    run = subparsers.add_parser("run", help="run the 32-agent baseline")
    _add_eval_arguments(run, default_steps=25)
    run.set_defaults(handler=lambda args: run_evaluation(args, smoke=False))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.handler(args)
    except (FleetError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
