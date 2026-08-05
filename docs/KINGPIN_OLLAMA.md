# Kingpin: rootless four-GPU Ollama evaluation

This deployment runs one independently managed Ollama server per A100 and maps
eight of the 32 Alem agents to each server. It does not use sudo, a custom
router, or the existing services on ports `11434` through `11437`.

## Copy/paste runbook

Run the following commands from your own workstation. After `ssh kingpin`, all
remaining commands run on Kingpin.

### 1. Connect and enter the checkout

```bash
ssh kingpin
cd /data/aimdieke/AlemDICE
```

### 2. Check the current evaluation

```bash
if test -f outputs/kingpin_ollama/eval25.exit; then
  printf 'Evaluation exit code: '
  cat outputs/kingpin_ollama/eval25.exit
else
  echo 'Evaluation is still running, or has not been started.'
fi

tail -100 outputs/kingpin_ollama/eval25.log
tmux ls
```

An exit code of `0` means success. Any other value means the evaluator failed;
inspect the end of `eval25.log`. To watch a running log, use:

```bash
tail -f outputs/kingpin_ollama/eval25.log
```

Press `Ctrl-C` to stop following the log; that does not stop the evaluation.

### 3. Check the managed Ollama fleet

```bash
python3 scripts/run_kingpin_ollama.py status
nvidia-smi
```

The expected fleet has one owned server on each port `11534`--`11537` and one
managed model runner on each GPU. Do not run `start` if this command reports a
healthy live fleet.

### 4. Start the fleet after a reboot

Only use these commands when `status` says there is no managed fleet:

```bash
python3 scripts/run_kingpin_ollama.py start
python3 scripts/run_kingpin_ollama.py preflight
python3 scripts/run_kingpin_ollama.py status
```

Run `preflight` immediately after a fresh `start`. It intentionally refuses to
begin when a model runner or any unrelated compute process is already using a
GPU. It loads the four replicas, checks their model digest and GPU assignment,
and sends 32 concurrent requests.

### 5. Run a one-step smoke evaluation

```bash
python3 scripts/run_kingpin_ollama.py smoke
```

### 6. Run the 32-agent, 25-step evaluation

For a foreground run that stops if the SSH connection closes:

```bash
python3 scripts/run_kingpin_ollama.py run \
  --steps 25 \
  --seed 14100
```

For the recommended detached run that survives an SSH or VPN disconnect:

```bash
if tmux has-session -t kingpin-eval25 2>/dev/null; then
  echo 'kingpin-eval25 already exists; inspect it before starting another run.'
else
  mv outputs/kingpin_ollama/eval25.exit \
    outputs/kingpin_ollama/eval25.exit.previous 2>/dev/null || true
  tmux new-session -d -s kingpin-eval25 \
    'cd /data/aimdieke/AlemDICE && .venv/bin/python scripts/run_kingpin_ollama.py run --steps 25 --seed 14100 > outputs/kingpin_ollama/eval25.log 2>&1; echo $? > outputs/kingpin_ollama/eval25.exit'
  echo 'Started kingpin-eval25.'
fi
```

Attach to its terminal with:

```bash
tmux attach -t kingpin-eval25
```

To detach without stopping it, press `Ctrl-B`, release both keys, then press
`D`.

### 7. Find the results

```bash
ls -lah outputs/alem_eval/gemma4_31b_ollama/
cat outputs/alem_eval/gemma4_31b_ollama/kingpin_n32_seed14100_25_easy/summary_stats.json
```

The summary file exists only after the evaluator finishes its reporting phase.

### 8. Stop only this deployment's Ollama servers

Stop or finish the evaluation first, then run:

```bash
python3 scripts/run_kingpin_ollama.py stop
```

This stops only the PID-verified processes created by this launcher. It does
not stop the pre-existing services on ports `11434`--`11437`.

### 9. Update Kingpin to the latest deployment branch

Do this when no evaluation is running:

```bash
cd /data/aimdieke/AlemDICE
git switch deploy/kingpin-ollama
git pull --ff-only
git status --short
```

`git status --short` should print nothing. Runtime files under `outputs/` are
ignored by Git and remain on the server.

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

## Fleet behavior and safety checks

`start` refuses to proceed if any GPU has a compute process or if a requested
port is occupied. Xorg graphics processes are not compute processes. The four
servers use ports `11534`--`11537`, GPU UUID pinning, `gemma4:31b`, a 12,288
token context, and eight request slots each. `preflight` verifies the model
digest, sends 32 simultaneous requests, and proves that each model runner is a
descendant of the expected server on the expected GPU. If eight-way concurrency
fails, it restarts only its own servers once with four slots per GPU.

## Evaluation configuration

The run uses the unmodified baseline team topology, 32 agents, one episode,
native Gemma thinking, and disabled W&B. JAX is forced to CPU so simulation
work cannot reserve inference GPU memory. Debug pixel rendering is disabled for
this text-only evaluation: enabling it would render a full frame separately for
every agent on every tick, while leaving the models' text observations
unchanged. Results are written beneath `outputs/alem_eval/gemma4_31b_ollama/`.

Resume a preserved failed run by passing its exact output directory:

```bash
python scripts/run_kingpin_ollama.py run --steps 25 --seed 14100 \
  --resume-from outputs/alem_eval/gemma4_31b_ollama/<run-directory>
```

## Process ownership

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
