# E2d2 replicated joint-allocation results

Status: complete (canonical provider-free campaign, 2026-07-23).

The source, tests, runner, hypotheses, outcomes, integrity gates, and canonical
seeds were frozen before launch in
`reports/agent_scaling_recruitment/e2d2_joint_allocation_preregistration.md`.
The preregistration/source commit was
`4c21420ee3958b7e9fe68073b345a2dc40f56857`, and the worktree was clean at
launch.

## Frozen comparison

The campaign held the six-agent Contract Net protocol and `public_sweep` bid
barrier fixed and compared:

1. `first_valid`, the E2 reference;
2. task-local `exact_utility`; and
3. `joint_exact_allocation`, a deterministic replicated calculation over
   public task cards, delivered claimed bids, active public awards, and public
   team membership.

The joint objective maximizes public completed reward first, then summed
claimed-information roster utility, then minimizes summed claimed raw cost,
then selects the lexicographically smallest agent assignment. Each agent has
one task-or-idle label, so cross-task rosters are exclusive by construction.
Task sponsors actuate their own canonical awards; there is no privileged
planner or hidden input.

The canonical run used seeds 23200--24199, four frozen families, twelve acting
rounds, and eight workers: 12,000 episodes. It completed in 24.29 seconds with
zero model calls or provider requests.

## Aggregate results

| Method | Episodes | Normalized reward | Oracle allocation coverage | Utility regret at equal reward | Achieved raw cost | Raw-cost delta | Delivered control bytes/episode | Mean lock round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_valid` | 4,000 | 1.0000 | 1.0000 | 0.045707 | 136.294 | 41.086 | 3,776.0 | 5.875 |
| task-local `exact_utility` | 4,000 | 1.0000 | 1.0000 | 0.008581 | 102.073 | 6.865 | 3,721.2 | 5.690 |
| `joint_exact_allocation` | 4,000 | 1.0000 | 1.0000 | 0.000000 | 95.208 | 0.000 | 3,558.5 | 5.500 |

Raw-cost delta is achieved minus oracle raw cost and is descriptive: the
oracle maximizes utility before minimizing raw cost.

### Preregistered joint-versus-task-local contrast

Joint allocation preserved normalized reward, completed-task count, and
oracle-allocation coverage in all 4,000 paired episodes. Every paired
top-line difference was exactly zero, satisfying the reward-safety condition.

Joint allocation reduced mean utility regret by 0.008581, eliminating 100% of
the task-local residual. The seed-clustered 95% normal interval for
joint-minus-task-local regret was [-0.009209, -0.007954]. Joint was better in
748 paired seed-family episodes, tied in 3,252, and was never worse.

Because E2d2 reports are truthful and its public objective matches the scoring
oracle's reward/utility/cost order, joint allocation matched oracle utility and
raw cost in all 4,000 episodes. This is an integration result under truthful
claims, not evidence that public selection remains oracle-optimal under noisy
or strategic reports.

Secondary effects also favored joint allocation:

- achieved raw cost fell by 6.865 per episode, or 6.73%
  (clustered 95% interval [-7.367, -6.363]);
- delivered control traffic fell by 162.654 bytes/episode, or 4.37%
  (interval [-166.103, -159.205]);
- bounded control submissions fell by 0.7535/episode, or 4.57%; and
- mean lock time improved by 0.190 rounds, or 3.34%
  (interval [-0.202, -0.178]).

Traffic was lower in 1,507 pairs and tied in 2,493, with no regressions. Lock
time was earlier in 507 pairs and tied in 3,493, with no regressions. The
traffic reduction has a protocol explanation: joint allocation does not emit
an award for the mutually impossible second scarce task, and it avoids the
overlapping first awards that force recovery in some two-disjoint episodes.
The enumeration itself is local computation and does not add communication.

Relative to `first_valid`, joint allocation reduced utility regret by 0.045707
(clustered 95% interval [-0.047027, -0.044386]), reduced achieved raw cost by
41.086, reduced delivered bytes by 217.5/episode, and locked 0.375 rounds
earlier. Reward and allocation coverage remained identical.

## Family localization

| Family | Task-local regret | Joint regret | Improved/tied/worse | Task-local bytes | Joint bytes | Task-local lock | Joint lock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single complementary | 0.000000 | 0.000000 | 0/1,000/0 | 2,452.4 | 2,452.4 | 5.000 | 5.000 |
| Oversubscribed | 0.000000 | 0.000000 | 0/1,000/0 | 2,597.2 | 2,597.2 | 5.000 | 5.000 |
| Scarce capability | 0.026083 | 0.000000 | 509/491/0 | 4,484.6 | 4,059.6 | 6.000 | 6.000 |
| Two disjoint | 0.008243 | 0.000000 | 239/761/0 | 5,350.5 | 5,124.9 | 6.761 | 6.000 |

This localization matches the preregistered mechanism diagnosis. Joint scope
does nothing when one task is active or task-local exact selection already
matches the global assignment. It removes the residual precisely when tasks
compete for scarce agents or independently attractive rosters overlap.

## Scientific interpretation

E2d2 is a positive mechanism result. The E2d residual was caused by allocation
scope, not a weakness in the task-local utility formula. A reward-first joint
calculation over the shared delayed ledger removes that residual without a
leader, reward loss, extra messages, or slower formation. In these frozen
families it also reduces communication and latency by preventing awards that
would later conflict.

The strongest supported claim is:

> With truthful public bids and six agents, a bounded replicated joint
> allocation rule can realize the same reward/utility/cost assignment as the
> analysis oracle while preserving decentralized task-sponsored actuation.

The result does not establish robustness to false or noisy capability claims,
more than two simultaneous tasks, changing membership during enumeration, LLM
record-generation failures, or multi-step Alem execution. The next decisive
tests are report-noise/calibration ablations and the LLM/Alem execution bridge.

## Integrity and independent audit

Every preregistered gate passed:

- 12,000 expected seed-family-arm rows, all unique;
- zero invalid records and zero rejected transitions;
- zero unauthorized ordinary deliveries;
- exact requested roster sizes and no cross-task member overlap;
- claimed-information-only selector audits;
- full roster agreement;
- zero model calls, provider requests, and transport errors; and
- matching terminal state and audit-chain hashes for every replay.

An independent post-run audit:

- recomputed the three artifact hashes;
- recomputed compressed and uncompressed hashes for all eight event shards;
- recomputed and matched all 12,000 per-episode canonical hashes to the CSV;
- independently replayed all 12,000 directory histories;
- matched every replay terminal state and audit-chain hash;
- reconstructed the public ledger at every joint decision and matched all
  12,000 public-input digests;
- checked all 12,000 joint allocation decisions for exact roster sizes,
  exclusivity, reward-first objective order, and claimed-only content; and
- found zero errors.

## Provenance

Output root:
`outputs/recruitment_arena/e2d2_joint_allocation_v1`

Artifact SHA-256:

- `run_manifest.json`:
  `a7bd1769d75062069e0ce18c456bea3438bc841a1e76036e02239d8e218861e3`
- `episodes.csv`:
  `79980695502646f1c550a43e0d5f6f7670053b1128d40a3cf5031692246165b2`
- `summary.json`:
  `871b39b528b7e8fa1a5e71d4b902b92caebd4631149516e258e6258a5666bbf4`
- `summary.md`:
  `731666847d780e37c1918eff41569d2c1012be9835d43749f758b366a4981536`

Canonical command:

```bash
uv run --extra baselines-llm --python 3.12 \
  python scripts/run_recruitment_selector_ablation.py \
  --seed-start 23200 \
  --num-seeds 1000 \
  --rounds 12 \
  --workers 8 \
  --selectors first_valid exact_utility joint_exact_allocation \
  --output /home/adam/Desktop/AlemDICE/outputs/recruitment_arena/e2d2_joint_allocation_v1
```

Validation command:

```bash
uv run --extra dev --extra baselines-llm --python 3.12 \
  pytest -q \
  baselines/llm/test_recruitment_selection.py \
  baselines/llm/test_recruitment_arena.py
```

Result: 37 passed. Ruff, Ruff format-check, and Python byte-compilation also
passed for all changed source/test files before launch.
