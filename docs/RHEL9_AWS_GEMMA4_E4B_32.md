# RHEL 9 on AWS: 32-agent Gemma 4 E4B evaluation

This guide installs AlemDICE on a RHEL 9 EC2 instance and runs one Easy
episode with 32 physical agents, seed 14100, and a 1,000-step cap. All agents
send requests to one vLLM server at `http://127.0.0.1:8000/v1`.

The launcher keeps JAX on CPU so the single GPU remains dedicated to vLLM.
Each active tick submits the 32 agent calls concurrently; vLLM performs
continuous batching on the shared GPU. With one episode, `eval.num_workers=1`
is intentional and does not serialize the agents.

## 1. RHEL prerequisites

Confirm the operating system, architecture, disk space, and GPU:

```bash
cat /etc/redhat-release
uname -m
df -h .
nvidia-smi
```

Use an x86-64 RHEL 9 instance. If `nvidia-smi` is unavailable, provision an
AWS NVIDIA GPU AMI or have the machine administrator install a driver before
installing vLLM. The host driver and vLLM are outside AlemDICE's Python
environment.

Git and `curl` are commonly already installed. With `sudo`, missing packages
can be installed through RHEL's package manager:

```bash
sudo dnf install -y git curl
```

Administrator access is not otherwise required. On a managed instance without
`sudo`, ask for Git and `curl`, or transfer an existing repository checkout and
the standalone `uv` binary into your user account.

## 2. Install AlemDICE as the current user

Install `uv` under the current account:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Enter an existing checkout, or clone it first:

```bash
git clone <ALEMDICE_REPOSITORY_URL> "$HOME/AlemDICE"
cd "$HOME/AlemDICE"
```

Let `uv` install Python 3.12 and the locked CPU/LLM dependencies:

```bash
uv python install 3.12
uv sync --frozen --python 3.12 --extra baselines-llm

JAX_PLATFORMS=cpu uv run --frozen python -c \
  "import alem, openai; print('Alem', alem.__version__)"
```

Do not install the repository's `gpu` extra on this single-GPU host: the GPU
belongs to vLLM, while the comparatively light environment simulation runs on
CPU.

## 3. Verify the existing vLLM server

The configured model ID is case-sensitive and must appear in `/v1/models`:

```bash
curl -sS http://127.0.0.1:8000/v1/models |
  uv run --frozen python -m json.tool
```

The default expected ID is `gemma-4-E4B-it`. If the server reports a different
ID, pass that exact value through `--model`.

The endpoint should remain bound to `127.0.0.1` when the evaluator runs on the
same host. There is no reason to expose port 8000 through the EC2 security
group.

## 4. Optional: install and start vLLM

Skip this section when the server is already running. Keep vLLM in a separate
environment because it pins PyTorch/CUDA packages that should not be mixed
with AlemDICE's JAX environment:

```bash
uv venv --python 3.12 --seed "$HOME/.venvs/vllm"
source "$HOME/.venvs/vllm/bin/activate"
uv pip install vllm --torch-backend=auto
```

Set `HF_MODEL` to the actual Hugging Face model repository or local checkpoint
path. `--served-model-name` gives the API the stable ID expected by the
experiment:

```bash
export HF_MODEL="<HUGGING_FACE_OR_LOCAL_GEMMA_4_E4B_MODEL>"

CUDA_VISIBLE_DEVICES=0 vllm serve "$HF_MODEL" \
  --served-model-name gemma-4-E4B-it \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90
```

If startup runs out of memory, lower `--max-model-len` or
`--max-num-seqs`. Lowering `--max-num-seqs` does not change the evaluator's
32-agent semantics; vLLM queues excess requests, reducing throughput.

The default experiment uses explicit tagged reasoning and sends
`enable_thinking=false`, so `--reasoning-parser` is not required. To test
Gemma's native thinking later, launch the server with its supported Gemma
reasoning parser and add `agent.reasoning=true` to the evaluator overrides.

## 5. Run the evaluation

Inspect the fully resolved command without starting the episode:

```bash
uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py --dry-run
```

Start the default run:

```bash
uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py
```

If `/v1/models` reports another served name:

```bash
uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py \
  --model "<EXACT_ID_FROM_V1_MODELS>"
```

The run is timestamped under:

```text
outputs/alem_eval/gemma4_e4b_vllm/
```

The episode can terminate before 1,000 steps if the environment reaches a
terminal state. The upper bounds are 1,000 environment ticks and 32,000
decision calls; inactive-agent handling can reduce the actual request count.

## 6. Run in the background and resume

For a long SSH session, use a terminal multiplexer if one is available.
Otherwise, launch with `nohup` and record only the evaluator PID:

```bash
mkdir -p outputs/launcher_logs

nohup uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py \
  > outputs/launcher_logs/gemma4_e4b_n32_1000.log 2>&1 &

echo "$!" > outputs/launcher_logs/gemma4_e4b_n32_1000.pid
tail -f outputs/launcher_logs/gemma4_e4b_n32_1000.log
```

Stop the evaluator without stopping vLLM:

```bash
kill "$(cat outputs/launcher_logs/gemma4_e4b_n32_1000.pid)"
```

The log prints the resolved run directory. Resume that directory with:

```bash
uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py \
  --resume-from outputs/alem_eval/gemma4_e4b_vllm/<RESOLVED_RUN_NAME>
```

## 7. Ray cluster handoff

For multiple local or networked GPUs, use Ray Serve instead of manually
managing vLLM endpoints. See `docs/RAY_CLUSTER_INFERENCE.md` for the complete
head, worker, safety-preflight, and deployment sequence. The environment and
policy profile do not change; point the launcher at Ray Serve:

```bash
uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py \
  --base-url http://127.0.0.1:8000/v1
```

This preserves the agent count, seed, horizon, prompts, and model ID, making
single-server versus Ray Serve throughput directly comparable.

## References

- [uv Linux installation](https://docs.astral.sh/uv/getting-started/installation/)
- [uv-managed Python](https://docs.astral.sh/uv/guides/install-python/)
- [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [RHEL 9 DNF package management](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_software_with_the_dnf_tool/)
