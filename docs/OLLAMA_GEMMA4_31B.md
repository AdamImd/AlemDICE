# Run Gemma 4 31B through Ollama

This guide connects AlemDICE to the native Ollama chat API for the
`gemma4:31b` model. The verified server is `kingpin` (`10.40.232.232`), where
Ollama 0.23.1 listens only on `127.0.0.1:11434`. AlemDICE therefore connects
through an SSH local port forward.

The installed model is the 31.3B-parameter Q4_K_M build. The repository preset
uses text-only observations by default, although the native client also supports
image attachments.

## 1. Install the LLM dependencies

From the AlemDICE repository:

```bash
uv sync --extra baselines-llm
```

The `baselines-llm` extra includes the official Ollama Python client. Ollama
itself remains installed and managed on `kingpin`.

## 2. Open the SSH tunnel

Keep this command running in a dedicated terminal:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 127.0.0.1:11434:127.0.0.1:11434 \
  kingpin
```

If the SSH alias is unavailable, use the verified user and address directly:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 127.0.0.1:11434:127.0.0.1:11434 \
  aimdieke@10.40.232.232
```

If local port 11434 is occupied, replace the first `11434` after `-L` with
`11435` and use `http://127.0.0.1:11435` as each client's `base_url`.

## 3. Verify Ollama and the model

In another terminal:

```bash
curl http://127.0.0.1:11434/api/version
curl http://127.0.0.1:11434/api/tags
```

The tags response must include the exact model ID `gemma4:31b`.

## 4. Run a smoke evaluation

The provider preset configures all three agents, the native host URL, Gemma's
recommended sampling values, a 12,288-token context, and internal thinking:

```bash
uv run --extra baselines-llm --python 3.12 \
  python baselines/llm/eval_alem.py \
  provider=ollama_gemma4_31b \
  WANDB_MODE=disabled \
  eval.num_episodes.alem=1 \
  eval.max_steps_per_episode=5 \
  eval.num_workers=1 \
  eval.save_images=false \
  eval.debug=false \
  eval.generate_debriefs=false
```

`agent.reasoning=true` makes the native request use `think=true` and records
Ollama's separate `message.thinking` value as the agent reasoning trace. Disable
internal thinking while retaining the harness's visible chain-of-thought format
with:

```bash
agent.reasoning=false
```

For a generic launcher invocation instead of the preset:

```bash
scripts/run_llm_eval.sh gemma4:31b \
  --client ollama \
  --thinking true \
  --episodes 1 \
  --steps 5 \
  --difficulty easy \
  --smoke
```

The native Ollama URL is the server root (`http://127.0.0.1:11434`), not the
OpenAI-compatible `/v1` path.

## 5. Run a longer evaluation

After the smoke test succeeds, increase episodes and steps explicitly:

```bash
uv run --extra baselines-llm --python 3.12 \
  python baselines/llm/eval_alem.py -m \
  provider=ollama_gemma4_31b \
  WANDB_MODE=disabled \
  alem.coordination_difficulty=easy,medium,hard \
  eval.num_episodes.alem=20 \
  eval.max_steps_per_episode=10000
```

Provider routing is explicit. `client_name=ollama` selects the native adapter;
the existing `openai`, `openai_responses`, and `vllm` clients remain available
for models served through those interfaces.

## Troubleshooting

### Connection refused

Confirm the tunnel process is still running and that the local listener exists:

```bash
ss -ltn 'sport = :11434'
ssh kingpin 'curl -sS http://127.0.0.1:11434/api/version'
```

Direct HTTP to `10.40.232.232:11434` will fail while Ollama remains bound to
remote loopback; this is expected.

### Wrong base URL

The native client rejects URLs containing `/v1`. Use the root URL for
`client_name=ollama`. OpenAI-compatible clients continue to use URLs ending in
`/v1`.

### Slow first request

The first request loads roughly 20 GB of model weights. Keep the 420-second
preset timeout for cold starts. Subsequent requests reuse the loaded model while
Ollama's keep-alive window remains active.

## References

- [Ollama native chat API](https://docs.ollama.com/api/chat)
- [Official Ollama Python client](https://github.com/ollama/ollama-python)
- [Gemma 4 model page](https://ollama.com/library/gemma4)
