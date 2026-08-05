# Kingpin: rootless four-GPU Ollama evaluation

This deployment runs one independently managed Ollama server per A100 and maps
eight of the 32 Alem agents to each server. It does not use sudo, a custom
router, or the existing services on ports `11434` through `11437`.

## One-time installation

From a machine with access to the repository:

```bash
ssh kingpin
mkdir -p /data/aimdieke
cd /data/aimdieke
git clone --branch deploy/kingpin-ollama \
  https://github.com/AdamImd/AlemDICE.git
cd AlemDICE
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra baselines-llm --python 3.12
```

Kingpin already provides `/usr/local/bin/ollama`, four A100 80 GB GPUs, and the
read-only shared model store `/home/ollama/models`. The launcher checks all of
these assumptions before it starts anything.

## Start and validate the fleet

Run these commands from `/data/aimdieke/AlemDICE`:

```bash
export PATH="$HOME/.local/bin:$PATH"
python scripts/run_kingpin_ollama.py start
python scripts/run_kingpin_ollama.py preflight
python scripts/run_kingpin_ollama.py status
```

`start` refuses to proceed if any GPU has a compute process or if a requested
port is occupied. Xorg graphics processes are not compute processes. The four
servers use ports `11534`--`11537`, GPU UUID pinning, `gemma4:31b`, a 12,288
token context, and eight request slots each. `preflight` verifies the model
digest, sends 32 simultaneous requests, and proves that each model runner is a
descendant of the expected server on the expected GPU. If eight-way concurrency
fails, it restarts only its own servers once with four slots per GPU.

## Smoke and evaluation

```bash
python scripts/run_kingpin_ollama.py smoke
python scripts/run_kingpin_ollama.py run --steps 25 --seed 14100
```

The run uses the unmodified baseline team topology, 32 agents, one episode,
native Gemma thinking, and disabled W&B. JAX is forced to CPU so simulation
rendering cannot reserve inference GPU memory. Results are written beneath
`outputs/alem_eval/gemma4_31b_ollama/`.

For a detached run that survives an SSH disconnect:

```bash
tmux new-session -d -s alem-ollama \
  'cd /data/aimdieke/AlemDICE && export PATH="$HOME/.local/bin:$PATH" && python scripts/run_kingpin_ollama.py run --steps 25 --seed 14100 2>&1 | tee outputs/kingpin_ollama/eval-25.log'
tmux attach -t alem-ollama
```

Resume a preserved failed run by passing its exact output directory:

```bash
python scripts/run_kingpin_ollama.py run --steps 25 --seed 14100 \
  --resume-from outputs/alem_eval/gemma4_31b_ollama/<run-directory>
```

## Stop and diagnose

```bash
python scripts/run_kingpin_ollama.py status
python scripts/run_kingpin_ollama.py stop
```

The state file records both PID and Linux process start time. `stop` refuses to
signal a reused or changed PID and only terminates process groups created by the
launcher. It never uses `pkill`, never targets an unresolved variable, and never
stops the pre-existing Ollama services. Per-server logs and the state file are
under `outputs/kingpin_ollama/`.

Useful read-only checks are:

```bash
nvidia-smi
curl -s http://127.0.0.1:11534/api/ps
tail -f outputs/kingpin_ollama/ollama-11534.log
```
