#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="alem-dice"
PYTHON="$ROOT/.venv/bin/python"

usage() {
  echo "Usage: ./commands.sh {setup|test|smoke|openai-reduced|openai-full|resume|summarize|visualize} [args]"
}

require_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON. Run ./commands.sh setup first." >&2
    exit 2
  fi
}

command_name="${1:-}"
if [[ -z "$command_name" ]]; then
  usage
  exit 2
fi
shift

cd "$ROOT"

case "$command_name" in
  setup)
    if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      mamba env update --yes --name "$ENV_NAME" --file environment.yml
    else
      mamba env create --yes --name "$ENV_NAME" --file environment.yml
    fi
    env_python="$(mamba run --name "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
    mamba run --name "$ENV_NAME" uv sync \
      --python "$env_python" \
      --extra alem-dice \
      --extra dev \
      --locked
    echo "Ready: $ROOT/.venv"
    ;;
  test)
    require_python
    JAX_PLATFORM_NAME=cpu WANDB_MODE=disabled PYTHONPATH="$ROOT" \
      "$PYTHON" -m pytest -q "$@"
    ;;
  smoke)
    require_python
    JAX_PLATFORM_NAME=cpu WANDB_MODE=disabled \
      "$PYTHON" scripts/run_openai_matrix.py --profile fake_smoke "$@"
    ;;
  openai-reduced)
    require_python
    "$PYTHON" scripts/run_openai_matrix.py --profile openai_reduced "$@"
    ;;
  openai-full)
    require_python
    "$PYTHON" scripts/run_openai_matrix.py --profile upstream_main_full "$@"
    ;;
  resume)
    require_python
    run_dir="${1:-}"
    if [[ -z "$run_dir" ]]; then
      echo "resume requires the matrix RUN_DIR printed by the launcher" >&2
      exit 2
    fi
    if [[ ! -f "$run_dir/matrix_manifest.json" ]]; then
      echo "resume requires a matrix root containing matrix_manifest.json: $run_dir" >&2
      exit 2
    fi
    shift
    readarray -t manifest_values < <(
      "$PYTHON" - "$run_dir" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1]).expanduser().resolve() / "matrix_manifest.json"
payload = json.loads(manifest.read_text(encoding="utf-8"))
print(payload["profile"])
print(payload.get("ablation") or "")
PY
    )
    resume_args=(--profile "${manifest_values[0]}" --resume "$run_dir")
    if [[ -n "${manifest_values[1]}" ]]; then
      resume_args+=(--ablation "${manifest_values[1]}")
    fi
    "$PYTHON" scripts/run_openai_matrix.py "${resume_args[@]}" "$@"
    ;;
  summarize)
    require_python
    run_dir="${1:-}"
    if [[ -z "$run_dir" ]]; then
      echo "summarize requires RUN_DIR" >&2
      exit 2
    fi
    "$PYTHON" scripts/summarize_run.py "$run_dir"
    ;;
  visualize)
    run_dir="${1:-}"
    if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
      echo "visualize requires an existing RUN_DIR" >&2
      exit 2
    fi
    mapfile -t viewers < <(find "$run_dir" -type f -name '*_debug.html' | sort)
    if [[ "${#viewers[@]}" -eq 0 ]]; then
      echo "No debug HTML found under $run_dir" >&2
      exit 1
    fi
    printf '%s\n' "${viewers[@]}"
    if command -v xdg-open >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
      xdg-open "${viewers[0]}" >/dev/null 2>&1 &
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac
