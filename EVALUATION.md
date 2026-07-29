# Evaluation protocol

This document defines the minimum evidence required for results reported from
the AAAI artifact.

## Comparison contract

A treatment and baseline are directly comparable only when they share:

- the same Git revision and dependency lock;
- the same Alem task, difficulty, world seed, and episode horizon;
- the same number of physical agents and maximum task-team size;
- the same model identifier, inference settings, prompt mode, and action parser;
- the same observation and physical action interfaces; and
- the same retry, failure, and early-termination accounting.

Any mismatch must be labeled as an integration check or cross-run context, not
as treatment uplift. Use paired seeds whenever reporting a treatment effect.

## Primary outcomes

Report both normalized and absolute outcomes:

- mean episode return;
- total task completions (unique team achievement unlocks);
- normal task-completion percentage;
- coordination task-completion percentage;
- total task-completion percentage;
- normal and coordination reward percentages;
- per-agent reward distribution; and
- valid episodes, failed attempts, and achieved horizon.

For scaling studies, absolute task completions are the primary measure of team
capacity. Per-agent or percentage measures characterize efficiency and must not
replace the absolute count.

## Reliability and cost

Every run must also record:

- model calls and provider attempts;
- transport failures and semantic/action-parse failures;
- input, output, reasoning, and cached tokens when exposed;
- parse success rate;
- mean model latency and episode wall time;
- communication payload and delivered bytes when applicable; and
- resolved configuration, seed ledger, source commit, and dependency-lock hash.

Failed episodes remain in the attempt ledger. They must not be silently removed
from the denominator or merged with valid episodes.

## Statistical reporting

Use at least three matched seeds for exploratory comparisons and more for
confirmatory claims. Report the paired effect for every seed, the mean effect,
uncertainty interval, and all attempted runs. A single seed is descriptive
evidence only. Avoid significance claims when the study was adapted after
observing outcomes.

## Reproduction order

1. Run the focused unit tests.
2. Run E0, the provider-free variable-agent compatibility gate.
3. Reproduce the deterministic recruitment arena.
4. Run the hosted-model recruitment screen only after its dry run and canary.
5. Run matched Alem baseline/treatment comparisons.
6. Regenerate tables and figures from committed result summaries.

The concrete commands and experiment identifiers are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
