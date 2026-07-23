# Launch Gemma 4 31B with vLLM on CUDA 12.5

This guide serves the instruction-tuned `google/gemma-4-31B-it` checkpoint
through vLLM's OpenAI-compatible API, without Docker or root access. Google
reports 30.7B parameters for this model, so it is the approximately-30B dense
Gemma 4 model listed as **Gemma 4 31B** in the Alem results.

The target system is one node with 4 NVIDIA A100 80 GB GPUs and a driver that
supports CUDA 12.5. The examples use BF16, which is native on A100. Do not use
FP8- or NVFP4-only checkpoints on A100.

## 1. Obtain a GPU allocation

Do installation and serving on a compute node with visible GPUs, rather than on
a login node. On a Slurm cluster, an interactive allocation might look like:

```bash
salloc --nodes=1 --gres=gpu:a100:4 --cpus-per-task=32 --mem=256G --time=08:00:00
srun --pty bash
```

Cluster resource names differ. Use the local cluster documentation if
`gpu:a100:4` is not a valid GRES request.

Check the driver, GPUs, compiler, and available modules:

```bash
nvidia-smi
module avail cuda
module avail gcc
module avail python
```

Load user-accessible modules. These commands do not require `sudo`:

```bash
module load cuda/12.5
module load gcc/11
module load python/3.12

nvcc --version
gcc --version
python3 --version
```

vLLM source builds require GCC/G++ 11.3 or newer. The version printed by
`nvidia-smi` describes driver compatibility; the version printed by `nvcc`
describes the toolkit used to compile vLLM.

## 2. Authenticate with Hugging Face

Open <https://huggingface.co/google/gemma-4-31B-it>, accept any access terms,
and create a read token. Keep the token out of shell history when possible:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN
```

Choose a cache location with at least 100 GB free. A project or scratch
filesystem is preferable to a small home quota:

```bash
export HF_HOME="${SCRATCH:-$HOME}/huggingface"
mkdir -p "$HF_HOME"
```

## 3. Create a separate vLLM environment

Do not install vLLM into AlemDICE's environment: vLLM pins PyTorch and CUDA
components that can conflict with the repository's JAX dependencies.

Install `uv` under the current user account:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv --python 3.12 --seed "$HOME/.venvs/vllm-gemma4"
source "$HOME/.venvs/vllm-gemma4/bin/activate"
```

### Preferred CUDA 12.5 installation: build from source

Current prebuilt vLLM wheels use a newer CUDA runtime. Building against the
cluster toolkit avoids requiring a driver upgrade. Install the CUDA 12.4
PyTorch wheel first; it is compatible with a CUDA 12.5-capable driver:

```bash
uv pip install \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124
```

Point the build at the loaded CUDA toolkit and compile only for A100's compute
capability 8.0:

```bash
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=16

git clone --branch v0.19.1 --depth 1 \
  https://github.com/vllm-project/vllm.git "$HOME/src/vllm"
cd "$HOME/src/vllm"

python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
uv pip install --no-build-isolation -e .
```

vLLM 0.19.0 introduced Gemma 4 support; 0.19.1 includes additional Gemma 4
fixes. Use a newer stable tag if the cluster is being configured later, but
retain the CUDA 12.5 source-build procedure.

If the cluster has no CUDA or GCC modules, install them into a user-owned Conda
environment instead:

```bash
conda create -y -p "$HOME/.conda/envs/vllm-build" \
  python=3.12 pip cmake ninja gcc_linux-64=11 gxx_linux-64=11
conda activate "$HOME/.conda/envs/vllm-build"
conda install -y -c nvidia/label/cuda-12.5.0 cuda-toolkit

export CUDA_HOME="$CONDA_PREFIX"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
```

Then run the PyTorch and vLLM source-build commands above. Only the NVIDIA
kernel driver must be installed by the cluster administrator.

Verify the installation:

```bash
python - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

## 4. Launch Gemma 4 31B

For the simplest deployment using all four allocated GPUs:

```bash
source "$HOME/.venvs/vllm-gemma4/bin/activate"
export HF_HOME="${SCRATCH:-$HOME}/huggingface"
export HF_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --reasoning-parser gemma4
```

The 12,288-token limit matches the repository's documented local-model setup
and leaves substantially more KV-cache space than Gemma 4's maximum 256K
context. The Alem agents are text-only, so multimodal profiling is disabled.

The official vLLM Gemma 4 recipe uses TP2 for the 31B model on A100. TP2 is a
good alternative when reserving only two GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1

vllm serve google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --reasoning-parser gemma4
```

Keep the server in the foreground initially so startup and out-of-memory errors
are visible. Model download and first startup can take several minutes.

## 5. Slurm batch script

Save the following as `serve-gemma4-31b.sbatch` outside the repository and
adjust the partition, account, module names, and time limit for the cluster:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=gemma4-vllm
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=gemma4-vllm-%j.log

set -euo pipefail

module load cuda/12.5
module load gcc/11
module load python/3.12

source "$HOME/.venvs/vllm-gemma4/bin/activate"
export HF_HOME="${SCRATCH:-$HOME}/huggingface"
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --reasoning-parser gemma4
```

Submit and inspect the log:

```bash
sbatch serve-gemma4-31b.sbatch
squeue -u "$USER"
tail -f gemma4-vllm-JOB_ID.log
```

If AlemDICE runs on another node, use the compute node's hostname rather than
`localhost`, subject to the cluster's networking policy:

```bash
squeue -j JOB_ID -o '%N'
```

## 6. Test the OpenAI-compatible API

Check server readiness:

```bash
curl http://localhost:8000/v1/models
```

Test a non-thinking response:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-31B-it",
    "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Test Gemma 4 reasoning parsing:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-31B-it",
    "messages": [{"role": "user", "content": "What is 17 times 23?"}],
    "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

The response should contain final text in `choices[0].message.content` and, when
thinking is enabled, parsed reasoning in the model's reasoning field.

## 7. Connect AlemDICE

Run AlemDICE from its own environment, not the vLLM environment. Start with a
five-step smoke test:

```bash
cd /path/to/AlemDICE_dev
source .venv/bin/activate

scripts/smoke_llm.sh google/gemma-4-31B-it \
  --base-url http://localhost:8000/v1 \
  --steps 5 \
  --coord easy
```

For a direct three-agent evaluation with Gemma 4 thinking enabled:

```bash
python baselines/llm/eval_alem.py \
  agent.reasoning=true \
  clients.0.client_name=vllm \
  clients.1.client_name=vllm \
  clients.2.client_name=vllm \
  clients.0.model_id=google/gemma-4-31B-it \
  clients.1.model_id=google/gemma-4-31B-it \
  clients.2.model_id=google/gemma-4-31B-it \
  clients.0.base_url=http://localhost:8000/v1 \
  clients.1.base_url=http://localhost:8000/v1 \
  clients.2.base_url=http://localhost:8000/v1 \
  eval.num_episodes.alem=1 \
  eval.max_steps_per_episode=200 \
  alem.coordination_difficulty=easy
```

All three agents share one loaded model. Their concurrent requests consume KV
cache but do not load three copies of the weights.

## Troubleshooting

### CUDA driver version is insufficient

Confirm that the source build is active rather than a prebuilt CUDA 12.9 wheel:

```bash
which vllm
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
nvcc --version
```

Rebuild from the source checkout with the CUDA 12.5 module loaded.

### Out of memory during startup

Try these changes in order:

1. Keep TP4 and reduce `--gpu-memory-utilization` to `0.85`.
2. Reduce `--max-model-len` to `8192`.
3. Reduce `--max-num-seqs` to `4`.
4. Confirm no unrelated process is occupying the GPUs with `nvidia-smi`.

### NCCL or tensor-parallel failure

All GPUs must be on the same allocated node. Inspect topology and shared-memory
space:

```bash
nvidia-smi topo -m
df -h /dev/shm
```

For diagnosis only, try TP2 on GPUs 0 and 1. Do not set NCCL transport-disabling
variables permanently without guidance from the cluster administrator.

### Hugging Face 401 or 403 response

Accept the model's Hugging Face terms and verify authentication:

```bash
hf auth login --token "$HF_TOKEN"
hf auth whoami
```

## References

- [Gemma 4 31B IT model card](https://huggingface.co/google/gemma-4-31B-it)
- [Official vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)
- [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [NVIDIA user-local Conda toolkit installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation)

