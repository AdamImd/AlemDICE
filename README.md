# Anonymous AAAI Artifact: Scalable Team Formation in Alem

This branch is the minimal research artifact for an AAAI submission on
decentralized team formation and language-agent coordination. It builds on the
[Alem](https://github.com/alem-world/alem-env) environment and retains only the
code, configurations, data, and reports needed to inspect or reproduce the
submission.

The repository is intentionally anonymous and does not contain author
identities, development videos, historical prototypes, reinforcement-learning
trainers, leaderboard tooling, or obsolete deployment paths.

## What is included

| Path | Purpose |
| --- | --- |
| `alem/` | Alem environment and assets |
| `baselines/llm/` | Language-agent evaluator and team-formation mechanisms |
| `scripts/` | Reproducible experiment, analysis, and deployment entry points |
| `Results/` | Curated machine-readable results used by the reports |
| `reports/` | Detailed methods, protocols, and scaling analysis |
| `Figures/` | Submission-ready figures |
| `AAAI/` | AAAI manuscript, bibliography, style, and checklist sources |
| `docs/` | Experiment and infrastructure documentation |

The original Alem behavior is retained as the source baseline. Experimental
team formation is layered above that interface; it does not alter the physical
action space or observation semantics.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are the supported path. No
administrator access is required:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <anonymous-artifact-url>
cd <artifact-directory>
uv sync --frozen --extra baselines-llm
```

For a CUDA host, add `--extra gpu`. CPU execution is sufficient for unit tests,
provider-free experiments, and analysis.

## Fast verification

Run the focused test suite:

```bash
uv run --frozen --extra baselines-llm \
  python -m pytest alem/tests baselines/llm -q
```

Validate the variable-population evaluator without making model calls:

```bash
uv run --frozen --extra baselines-llm \
  python scripts/run_e0_agent_compatibility.py \
  --counts 1,2,3,4,6 \
  --seeds 13000,13001 \
  --steps 10 \
  --output outputs/e0_artifact
```

E0 uses deterministic scripted `Noop` agents. It is an engineering gate, not a
behavioral result.

## Main experiments

The experiment sequence and claim boundary are defined in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). The principal entry points are:

```bash
# Provider-free six-agent recruitment arena
uv run --frozen --extra baselines-llm \
  python scripts/run_recruitment_arena.py --help

# Hosted-model recruitment screen (inspect before any paid run)
uv run --frozen --extra baselines-llm \
  python scripts/run_recruitment_llm_screen.py --help

# Source-baseline population scaling
uv run --frozen --extra baselines-llm \
  python scripts/run_source_scaling_study.py --dry-run --stage all

# 32-agent, 1,000-step local vLLM evaluation
uv run --frozen --extra baselines-llm \
  python scripts/run_gemma4_e4b_32x1000.py --dry-run
```

Paid experiments are never launched by installation, tests, or manuscript
builds. Review each launcher's dry-run output and call budget first.

## Results and analysis

Curated raw summaries are in `Results/`. Regenerate the cross-scale tables and
figures with:

```bash
uv run --frozen --extra baselines-llm \
  python scripts/analyze_scaling_results.py
```

The resulting report is
[`reports/scaling_analysis/scaling_analysis.pdf`](reports/scaling_analysis/scaling_analysis.pdf).
See [`EVALUATION.md`](EVALUATION.md) for metrics, matched-comparison rules, and
artifact requirements.

## Build the manuscript

With a TeX installation containing `latexmk`:

```bash
cd AAAI
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex
```

Generated TeX files are ignored. The committed manuscript sources are the
submission record.

## Distributed inference

The evaluator can target a local or Ray-managed OpenAI-compatible vLLM
endpoint. Cluster startup, local model paths, sharding, and model selection are
documented in [`docs/RAY_CLUSTER_INFERENCE.md`](docs/RAY_CLUSTER_INFERENCE.md).
The inference layer is operational infrastructure and does not change the
experimental protocol.

## Scope and provenance

This artifact is derived from Alem. The retained upstream boundary and local
changes are described in [`docs/UPSTREAM.md`](docs/UPSTREAM.md). Results should
be attributed only to configurations whose source revision, model, seed,
horizon, and environment settings match their recorded manifest.

The software is released under the MIT License; see [`LICENSE`](LICENSE).
