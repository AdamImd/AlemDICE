# Native Windows setup without Administrator access

The core Alem environment and the LLM evaluation client can run in native
64-bit Windows with only a user-installed `uv`. GPU-backed simulation and local
model serving have separate platform requirements described below.

## Supported path

Use Windows 10 or 11, PowerShell, and Python 3.12. From a PowerShell window:

```powershell
# Installs uv into the current user account.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart PowerShell if the installer changed PATH, then enter the checkout.
cd C:\path\to\AlemDICE

# uv installs Python, creates .venv, and installs the locked dependencies.
uv python install 3.12
uv sync --frozen --python 3.12 --extra baselines-llm

# Native Windows uses the CPU JAX backend.
$env:JAX_PLATFORMS = "cpu"
uv run --frozen python scripts/check_ntfs_paths.py
uv run --frozen python -c "import alem; print('alem', alem.__version__)"
uv run --frozen python examples/random_rl_agent.py --coord easy --steps 3 --log-every 0
uv run --frozen python examples/llm_text_smoke.py --coord easy --show-affordances
```

No activation of `.venv` is necessary. `uv run` selects it automatically.
The source can be obtained with a user-level Git installation or as a ZIP;
Git is not required merely to run an already downloaded checkout. The NTFS
path check itself uses Git and is intended for development checkouts.

JAX documents native Windows x86-64 CPU wheels, but labels them experimental.
Some machines may already have the Microsoft Visual C++ 2019 Redistributable;
if the JAX import reports a missing runtime DLL, an administrator may need to
install that Microsoft runtime.

## LLM evaluation

Hosted APIs and remote Ollama, vLLM, or SGLang servers work from native Windows
because this repository is only their HTTP client. For example, a three-agent
hosted OpenAI run can be launched directly from PowerShell:

```powershell
$env:OPENAI_API_KEY = "..."
$env:JAX_PLATFORMS = "cpu"
uv run --frozen python baselines/llm/eval_alem.py `
  clients.0.client_name=openai `
  clients.1.client_name=openai `
  clients.2.client_name=openai `
  clients.0.model_id=gpt-5.4-nano `
  clients.1.model_id=gpt-5.4-nano `
  clients.2.model_id=gpt-5.4-nano `
  eval.num_episodes.alem=1 `
  eval.max_steps_per_episode=10 `
  WANDB_MODE=disabled
```

Set each client's `base_url` when using a remote OpenAI-compatible server.
The native Ollama client can target an Ollama API URL instead; see
[`OLLAMA_GEMMA4_31B.md`](OLLAMA_GEMMA4_31B.md) for its provider configuration.

## What `uv` does and does not provide

`uv` alone is sufficient to install Python 3.12 and this repository's Python
dependencies in the current user's profile. It is therefore the recommended
native-Windows setup for CPU simulation and hosted or remote LLM evaluation.
It is not a replacement for non-Python system software:

| Capability | Native Windows without admin | Additional requirement |
| --- | --- | --- |
| Core symbolic/text environment on CPU | Yes | `uv` only, subject to the JAX runtime note above |
| Hosted or remote LLM evaluation | Yes | API credentials and network access |
| Human pygame client | Usually | `uv sync --extra play` and a desktop session |
| Local Ollama inference | Yes | Ollama's separate per-user Windows installer and model storage |
| Native JAX NVIDIA/AMD acceleration | No | Use Linux or an existing WSL2 installation |
| Local vLLM inference | No | vLLM requires Linux; use a remote server or WSL2 |
| Bash launch wrappers and research campaign orchestrators | No | Use WSL2/Linux; direct Python evaluators work natively |
| Replay MP4 encoding | No | A separately installed `ffmpeg` executable on `PATH` |
| LaTeX report compilation | No | A separately installed TeX distribution |
| Docker workflows | Environment-dependent | Docker Desktop/WSL setup, normally managed by an administrator |

Ollama's official Windows installer is per-user and does not require
Administrator privileges. vLLM explicitly does not support native Windows.
Likewise, JAX supports CPU execution natively but lists NVIDIA and AMD GPU
support only through Linux or experimental WSL2 paths. Enabling WSL2, installing
host GPU drivers, or installing missing Microsoft runtimes may require help
from the machine administrator even though no `root` access is needed inside
an already provisioned WSL distribution.

## Repository limitations on native Windows

The principal evaluator, environment examples, provider clients, and compact
test suite use portable Python. The following research operations remain
POSIX-specific:

- `commands.sh` and the shell scripts under `scripts/`;
- `scripts/run_source_scaling_study.py`, which uses POSIX advisory locks and
  process groups;
- `scripts/run_recruitment_llm_screen.py`, which uses POSIX advisory locks.

Run those exact campaign launchers in Linux or WSL2. On native Windows, invoke
`baselines/llm/eval_alem.py` directly with PowerShell arguments as above.

## Platform references

- [uv installation on Windows](https://docs.astral.sh/uv/getting-started/installation/)
- [uv-managed Python installations](https://docs.astral.sh/uv/guides/install-python/)
- [JAX supported platforms](https://docs.jax.dev/en/latest/installation.html)
- [Ollama for Windows](https://docs.ollama.com/windows)
- [vLLM GPU installation requirements](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
