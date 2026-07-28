#!/usr/bin/env bash
set -euo pipefail

RAY_BIN="${RAY_BIN:-$HOME/.venvs/ray-llm/bin/ray}"
GCS_PORT="${RAY_GCS_PORT:-6379}"
NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-6700}"
OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-6701}"
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-10001}"
MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-20000}"
MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-20999}"

usage() {
  cat <<'EOF'
Usage:
  ray_cluster_node.sh head HEAD_PRIVATE_IP
  ray_cluster_node.sh worker HEAD_PRIVATE_IP WORKER_PRIVATE_IP
  ray_cluster_node.sh status
  ray_cluster_node.sh stop

Environment:
  RAY_BIN                 Ray executable (default: ~/.venvs/ray-llm/bin/ray)
  CUDA_VISIBLE_DEVICES    Optional comma-separated GPU indices or UUIDs to offer Ray
  RAY_GCS_PORT            Head GCS port (default: 6379)
  RAY_MIN_WORKER_PORT     First Ray worker port (default: 20000)
  RAY_MAX_WORKER_PORT     Last Ray worker port (default: 20999)

The head may be CPU-only. A worker must expose at least one GPU. Before joining,
the script refuses any visible GPU with an existing NVIDIA compute process.
Graphics-only Xorg processes are not returned by the compute-process query.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_ray() {
  [[ -x "$RAY_BIN" ]] || die "Ray executable not found: $RAY_BIN"
}

validate_ip() {
  local value="$1"
  python3 - "$value" <<'PY'
import ipaddress
import sys

try:
    ipaddress.ip_address(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"invalid IP address {sys.argv[1]!r}: {exc}")
PY
}

visible_gpus() {
  local -n result_ref=$1
  result_ref=()

  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    [[ -n "$CUDA_VISIBLE_DEVICES" ]] || return 0
    IFS=',' read -r -a result_ref <<<"$CUDA_VISIBLE_DEVICES"
    return 0
  fi

  command -v nvidia-smi >/dev/null 2>&1 || return 0
  mapfile -t result_ref < <(
    nvidia-smi --query-gpu=uuid --format=csv,noheader |
      sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
  )
}

gpu_preflight() {
  local role="$1"
  local -a devices
  visible_gpus devices

  if [[ "$role" == "worker" && "${#devices[@]}" -eq 0 ]]; then
    die "worker exposes no GPUs; set CUDA_VISIBLE_DEVICES or fix NVIDIA access"
  fi

  local device busy
  for device in "${devices[@]}"; do
    [[ "$device" =~ ^[0-9]+$ || "$device" =~ ^GPU-[0-9A-Fa-f-]+$ ]] ||
      die "unsafe CUDA_VISIBLE_DEVICES entry: $device"
    busy="$(
      nvidia-smi -i "$device" \
        --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
    )"
    if [[ -n "$busy" ]]; then
      echo "GPU $device already has compute work:" >&2
      echo "$busy" >&2
      die "refusing to register an occupied GPU with Ray"
    fi
  done

  printf '%s\n' "${#devices[@]}"
}

start_head() {
  [[ "$#" -eq 1 ]] || { usage >&2; exit 2; }
  local head_ip="$1"
  validate_ip "$head_ip"
  require_ray

  local gpu_count
  gpu_count="$(gpu_preflight head)"

  "$RAY_BIN" start \
    --head \
    --node-ip-address="$head_ip" \
    --port="$GCS_PORT" \
    --node-manager-port="$NODE_MANAGER_PORT" \
    --object-manager-port="$OBJECT_MANAGER_PORT" \
    --ray-client-server-port="$RAY_CLIENT_PORT" \
    --min-worker-port="$MIN_WORKER_PORT" \
    --max-worker-port="$MAX_WORKER_PORT" \
    --dashboard-host=127.0.0.1 \
    --dashboard-port=8265 \
    --num-gpus="$gpu_count"

  echo "Ray head ready at $head_ip:$GCS_PORT with $gpu_count visible GPU(s)."
}

start_worker() {
  [[ "$#" -eq 2 ]] || { usage >&2; exit 2; }
  local head_ip="$1"
  local worker_ip="$2"
  validate_ip "$head_ip"
  validate_ip "$worker_ip"
  require_ray

  local gpu_count
  gpu_count="$(gpu_preflight worker)"

  "$RAY_BIN" start \
    --address="$head_ip:$GCS_PORT" \
    --node-ip-address="$worker_ip" \
    --node-manager-port="$NODE_MANAGER_PORT" \
    --object-manager-port="$OBJECT_MANAGER_PORT" \
    --min-worker-port="$MIN_WORKER_PORT" \
    --max-worker-port="$MAX_WORKER_PORT" \
    --num-gpus="$gpu_count"

  echo "Ray worker joined $head_ip:$GCS_PORT with $gpu_count visible GPU(s)."
}

command_name="${1:-}"
case "$command_name" in
  head)
    shift
    start_head "$@"
    ;;
  worker)
    shift
    start_worker "$@"
    ;;
  status)
    require_ray
    "$RAY_BIN" status
    ;;
  stop)
    require_ray
    "$RAY_BIN" stop --force
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
