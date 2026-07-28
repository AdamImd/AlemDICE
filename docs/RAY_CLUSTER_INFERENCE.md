# Ray cluster inference for AlemDICE

Ray Serve LLM is the inference gateway. The removed custom fleet router is not
part of this deployment. Ray owns replica placement, health checking, request
queuing, and OpenAI-compatible ingress.

The recommended order is:

1. Install the same Ray environment on every node.
2. Start the head.
3. Join each approved GPU worker.
4. Verify cluster resources.
5. Deploy Ray Serve from the head.
6. Start AlemDICE from the head.

Do not deploy the model before all intended workers have joined.

## Network and security

Use private IP addresses. On AWS, place every cluster node in a security group
that permits traffic from itself. The simplest reliable rule is all TCP from
that same security group, with no Ray ports open to the internet. The provided
scripts use:

- `6379`: Ray GCS on the head
- `6700`: node manager on each node
- `6701`: object manager on each node
- `10001`: Ray Client on the head
- `20000-20999`: Ray worker processes
- `8265`: head dashboard, bound to loopback
- `8000`: Ray Serve, bound to loopback

Because ports 8000 and 8265 bind to `127.0.0.1`, run the evaluator and Serve
deployment command on the head. Use an SSH forward for administrative access
instead of exposing them publicly.

## Install on every node

Use the identical Python and Ray versions on the head and workers:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv python install 3.12
uv venv --python 3.12 --seed "$HOME/.venvs/ray-llm"
uv pip install --python "$HOME/.venvs/ray-llm/bin/python" \
  "ray[serve,llm]==2.55.1"
```

Each worker must be able to obtain the same pinned model. Prefer an S3 model
mirror or a pre-populated local model volume for a multi-node AWS cluster.
Do not store Hugging Face, W&B, or AWS credentials in the Serve YAML.

## Select GPUs

With no override, a node offers all locally visible GPUs to Ray. To offer only
specific devices, set `CUDA_VISIBLE_DEVICES` before invoking the node script:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

The startup script checks every offered GPU with NVIDIA's compute-process
query. It exits without joining the cluster if another compute process is
using one. Xorg graphics processes do not appear in that query.

Exclude undersized or unsupported devices simply by omitting them from
`CUDA_VISIBLE_DEVICES`.

## Start the head

Choose one stable private address for the head. The head can also contribute
GPUs:

```bash
cd "$HOME/AlemDICE"
source "$HOME/.venvs/ray-llm/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
scripts/ray_cluster_node.sh head 10.0.1.10
```

For a CPU-only head:

```bash
export CUDA_VISIBLE_DEVICES=
scripts/ray_cluster_node.sh head 10.0.1.10
```

Do not use `127.0.0.1` as the head address when network workers must join.

## Join each networked GPU node

On every worker, using that worker's own private address:

```bash
cd "$HOME/AlemDICE"
source "$HOME/.venvs/ray-llm/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3
scripts/ray_cluster_node.sh worker 10.0.1.10 10.0.1.21
```

Run the equivalent command on each additional host, changing the worker IP and
visible device list. Ray schedules by logical GPU resources, so no SSH tunnels
or one-port-per-vLLM-server configuration is needed.

## Verify the cluster

On the head:

```bash
scripts/ray_cluster_node.sh status

python - <<'PY'
import ray

ray.init(address="auto")
print(ray.cluster_resources())
PY
```

Confirm that the reported `GPU` total equals the number of devices intentionally
offered by all nodes. If it does not, do not deploy Serve.

## Select and deploy a Gemma 4 model

The model selector generates a resolved Ray Serve configuration from the
chosen checkpoint, approved GPU count, and shard size. A standard NVIDIA L4
has 24 GB of memory; use `nvidia-smi` as the authority for the actual device.

Gemma 4 E4B BF16 needs approximately 17.9 GB and defaults to TP=1. On eight
L4s, this produces eight independent replicas:

```bash
python scripts/deploy_ray_gemma4.py \
  --model e4b \
  --gpus 8
```

Gemma 4 26B/A4B BF16 needs approximately 57.7 GB including Google's estimated
loading overhead. The L4 profile conservatively defaults to TP=4. On eight
L4s, this produces two independent replicas:

```bash
python scripts/deploy_ray_gemma4.py \
  --model 26b-a4b \
  --gpus 8
```

The default BF16 layouts are:

| Approved L4 GPUs | Gemma 4 E4B | Gemma 4 26B/A4B |
| ---: | --- | --- |
| 1 | 1 replica at TP=1 | Does not fit safely |
| 4 | 4 replicas at TP=1 | 1 replica at TP=4 |
| 8 | 8 replicas at TP=1 | 2 replicas at TP=4 |

For 26B/A4B, keep each four-GPU tensor-parallel group on one physical host
whenever possible. Tensor-parallel ranks exchange data during inference, so
spanning ordinary network links can be substantially slower. Ray's packed
placement favors co-location when a node has enough GPUs. E4B TP=1 replicas
do not perform cross-node tensor-parallel communication and are therefore the
simpler choice for one-GPU network workers.

Preview without deploying:

```bash
python scripts/deploy_ray_gemma4.py \
  --model 26b-a4b \
  --gpus 8 \
  --dry-run
```

The default served IDs are `gemma-4-E4B-it` and
`gemma-4-26B-A4B-it`. Use the exact selected ID for the evaluation. Advanced
overrides include `--tensor-parallel-size`, `--replicas`,
`--max-model-len`, and `--model-source`. The selector prevents a deployment
from requesting more GPUs than the verified cluster exposes.

For example, one TP=4 A4B replica on an eight-GPU cluster intentionally leaves
four GPUs unused:

```bash
python scripts/deploy_ray_gemma4.py \
  --model 26b-a4b \
  --gpus 8 \
  --tensor-parallel-size 4 \
  --replicas 1
```

Watch deployment status after either selection:

```bash
serve status
curl -sS http://127.0.0.1:8000/v1/models |
  "$HOME/.venvs/ray-llm/bin/python" -m json.tool
```

Model initialization can take several minutes. Start the evaluation only after
all configured replicas are healthy. Switching models replaces the existing
`gemma4_selected` Serve application.

## Run AlemDICE

The evaluation launcher targets Ray Serve's endpoint. Select the model ID that
matches the deployed profile:

```bash
uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model gemma-4-E4B-it
```

For 26B/A4B:

```bash
uv run --frozen --extra baselines-llm --python 3.12 \
  python scripts/run_gemma4_e4b_32x1000.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model gemma-4-26B-A4B-it
```

No agent, environment, seed, or prompt change is required.

## Stop or restart

Delete only the Serve application while leaving the Ray cluster alive:

```bash
serve shutdown -y
```

Stop workers first:

```bash
scripts/ray_cluster_node.sh stop
```

Then stop the head with the same command. To change cluster membership, shut
down Serve before removing GPU nodes; once the intended nodes have rejoined,
verify resources and rerun the model selector.

## Choosing manual startup versus Ray's AWS launcher

Use the commands above for a fixed group of existing instances. If Ray should
create and terminate EC2 nodes itself, use Ray's AWS cluster launcher and an
autoscaler configuration instead. Do not mix manual workers and autoscaler
ownership until the fixed cluster is validated.

## References

- [Google Gemma 4 model and memory overview](https://ai.google.dev/gemma/docs/core)
- [NVIDIA L4 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- [Ray manual cluster startup](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html)
- [Ray Serve configuration files](https://docs.ray.io/en/latest/serve/production-guide/config.html)
