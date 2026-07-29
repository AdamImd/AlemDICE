# Ray Serve model selection

Use `scripts/deploy_ray_gemma4.py` to generate and deploy exactly one Gemma 4
model profile. The generated Serve YAML is written beneath
`outputs/ray_serve/`; generated files are run artifacts, not source files.

On eight approved L4 GPUs:

```bash
# Eight independent E4B replicas (TP=1).
python scripts/deploy_ray_gemma4.py --model e4b --gpus 8

# Two independent 26B/A4B replicas, each sharded over four GPUs (TP=4).
python scripts/deploy_ray_gemma4.py --model 26b-a4b --gpus 8
```

Inspect a profile without deploying it:

```bash
python scripts/deploy_ray_gemma4.py \
  --model 26b-a4b \
  --gpus 8 \
  --dry-run
```

Use a local checkpoint by passing the same absolute path on every Ray node:

```bash
python scripts/deploy_ray_gemma4.py \
  --model e4b \
  --gpus 8 \
  --model-source /mnt/models/gemma-4-E4B-it
```

Ray does not copy a head-local model directory to workers. Mount or stage the
checkpoint at an identical path across all eligible nodes.

The default checkpoints are BF16. A standard NVIDIA L4 has 24 GB of memory.
The 26B/A4B BF16 profile therefore refuses TP values below four unless the
caller selects a quantized checkpoint or explicitly acknowledges an unsafe
fit. Always confirm actual hardware with `nvidia-smi`.

Default layouts:

| L4 GPUs | E4B | 26B/A4B |
| ---: | --- | --- |
| 1 | 1 replica at TP=1 | Unsafe BF16 fit |
| 4 | 4 replicas at TP=1 | 1 replica at TP=4 |
| 8 | 8 replicas at TP=1 | 2 replicas at TP=4 |

Keep each A4B TP=4 group on one four-GPU host when possible. E4B TP=1 is
better suited to collections of networked one-GPU workers because inference
does not require cross-node tensor-parallel collectives.
