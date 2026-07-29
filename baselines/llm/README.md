# Alem language-agent evaluator

This directory contains the code paths retained for the AAAI artifact:

- `eval_alem.py`: Hydra evaluation entry point;
- `eval_utils/agents/`: source language-agent policies;
- `eval_utils/team_formation.py`: recruitment state machine and communication
  restrictions;
- `eval_utils/recruitment_selection.py`: allocation and oracle comparators;
- `recruitment_arena.py`: deterministic six-agent recruitment environment;
- `recruitment_llm_screen.py`: bounded hosted-model screen; and
- `config/`: named baseline, scaling, ablation, and local-vLLM profiles.

The evaluator preserves the Alem text observations and physical action space.
Team-formation records affect who may exchange ordinary team messages; they do
not add privileged world observations or physical actions.

Install and test:

```bash
uv sync --frozen --extra baselines-llm
uv run --frozen --extra baselines-llm \
  python -m pytest baselines/llm -q
```

Run a named profile:

```bash
uv run --frozen --extra baselines-llm \
  python baselines/llm/eval_alem.py experiment=fake_smoke
```

See [`../../docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for the full
experiment order and [`../../EVALUATION.md`](../../EVALUATION.md) for reporting
requirements.
