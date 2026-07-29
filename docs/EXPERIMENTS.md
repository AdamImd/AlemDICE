# AAAI experiment sequence

The submission follows staged gates. Each stage answers one question and must
pass before evidence from the next stage is interpreted.

## E0 — evaluator compatibility

**Question:** Does the unmodified evaluator support variable team sizes without
shape, action-index, replay, accounting, or communication errors?

E0 uses deterministic scripted agents and makes no provider calls:

```bash
uv run --frozen --extra baselines-llm \
  python scripts/run_e0_agent_compatibility.py \
  --counts 1,2,3,4,6 \
  --seeds 13000,13001 \
  --steps 10 \
  --output outputs/e0_artifact
```

The committed summaries are `Results/e0_agent_compatibility_v1.*` and
`Results/e0_1_agent_compatibility_200_v1.*`. These are engineering results, not
behavioral evidence.

## E1 — source-baseline scaling

**Question:** How does the source language-agent baseline change as the number
of physical agents increases?

The frozen study uses populations 1, 2, 3, 4, and 6; seeds 13100–13102; Easy
difficulty; and 200 steps.

```bash
export OPENAI_API_KEY=...

uv run --frozen --extra baselines-llm \
  python scripts/run_source_scaling_study.py --stage all --dry-run

uv run --frozen --extra baselines-llm \
  python scripts/run_source_scaling_study.py --stage all
```

The dry run prints the exact commands and call ceiling. Curated episode data and
paired contrasts are in `Results/e1_source_scaling/`.

## E2a — deterministic recruitment arena

**Question:** Under controlled, bounded communication, which decentralized
recruitment rule forms feasible teams efficiently?

The provider-free six-agent arena compares recruitment and control arms over
scenario families with known full-information oracles:

```bash
uv run --frozen --extra baselines-llm \
  python scripts/run_recruitment_arena.py --help

uv run --frozen --extra baselines-llm \
  python scripts/run_recruitment_arena.py \
  --output outputs/recruitment_arena/e2a
```

Primary outcomes are normalized reward, regret, feasible formation rate,
communication bytes, roster agreement, and replay-hash agreement.

## E2b — hosted-model recruitment screen

**Question:** Can language agents produce valid decentralized recruitment
records under the same protocol and information boundary?

This stage is a bounded integration screen. Inspect its frozen budget, run a
canary, and only then authorize the complete screen:

```bash
uv run --frozen --extra baselines-llm \
  python scripts/run_recruitment_llm_screen.py --help
```

The protocol is
`reports/agent_scaling_recruitment/e2b_protocol.md`. Curated outcomes are in
`Results/e2b_v4_hosted_results_v1.*`. Failed attempts remain part of the audit
record.

## E3 — matched Alem transfer and scaling

**Question:** Does the selected recruitment protocol improve absolute
coordination task completions in Alem at matched seeds, horizons, team sizes,
and task requirements?

The matched analysis covers the source baseline and treatment at larger team
sizes. The comparison contract is in `EVALUATION.md`; committed scaling data are
in `Results/e3b1_scaling_matched/`. Regenerate all derived tables and figures:

```bash
uv run --frozen --extra baselines-llm \
  python scripts/analyze_scaling_results.py
```

Absolute task completions are primary. Normalized percentages and per-agent
returns diagnose efficiency. Single-seed cells remain descriptive until
repeated over matched seeds.

## Local and distributed inference

For a 32-agent, 1,000-step local vLLM run:

```bash
export ALEM_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export ALEM_VLLM_MODEL=/absolute/path/to/model

uv run --frozen --extra baselines-llm \
  python scripts/run_gemma4_e4b_32x1000.py --dry-run
```

Ray cluster startup and sharded model-serving commands are in
`RAY_CLUSTER_INFERENCE.md`. Changing inference infrastructure must not change
the model snapshot or experiment configuration in a matched comparison.

## Evidence rules

- Never combine failed attempts with zero-valued completed episodes.
- Never infer uplift from unmatched seeds, horizons, models, or task rules.
- Record absolute and normalized outcomes.
- Treat one-seed comparisons as descriptive.
- Preserve resolved configs, attempt ledgers, source hashes, and raw summaries.
