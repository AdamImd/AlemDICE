# E2d2 preregistration: replicated joint public-ledger allocation

Status: frozen before canonical launch on 2026-07-23.

At the time this protocol was written,
`outputs/recruitment_arena/e2d2_joint_allocation_v1` did not exist. The source,
tests, runner, and this note will be committed with a clean worktree before
the first canonical episode is launched.

## Motivation

E2d held Contract Net with public task sweeping fixed and found that task-local
`exact_utility` preserved complete reward while reducing utility regret by
81.7% relative to `first_valid`. Residual utility regret was confined to the
scarce and two-disjoint-task families. This is the predicted failure mode of a
greedy task-local selector: it can choose the best roster for one task without
choosing the best exclusive allocation across simultaneously active tasks.

E2d2 changes only the scope of the exact public calculation. It does not add a
leader, privileged state, extra model call, private message, or true-information
input.

## Replicated public-ledger mechanism

All task cards, CFPs, bids, awards, and accepted transitions are delivered by
the same delayed public TFP1 ledger. After the bid barrier for every active,
unawarded task, each agent can independently run the same bounded pure
function over:

- public task IDs, rewards, demands, and exact requested roster sizes;
- delivered bidder IDs, claimed capabilities, and claimed costs;
- current public awards; and
- public current team membership.

The function enumerates one label per currently eligible agent: an active task
ID or idle. With at most six agents and at most two frozen tasks, the bound is
at most \((T+1)^6\) assignments. A nonempty task assignment is valid only when
it has the exact requested size, every member has a delivered bid for that
task, and summed claimed capability meets every positive demand dimension.
Because each agent has exactly one label, simultaneous rosters are exclusive.

The objective is lexicographic and exact:

1. maximize summed public reward of assigned tasks;
2. maximize summed task-local claimed-information utility

   \[
   U(S,t)=
   \min_{j:d_{tj}>0}
   \frac{\sum_{i\in S}q_{ij}}{d_{tj}}
   -
   0.25\frac{\sum_{i\in S}c_{it}}{100|S|};
   \]

3. minimize summed claimed raw cost; and
4. minimize the canonical agent-assignment tuple, with stable task IDs before
   idle.

The public-reward objective is first, so utility cannot be purchased by
dropping a publicly feasible task. True capabilities, true costs, terminal
feasibility, and the analysis oracle are unavailable to this computation.

The plan is not stored by a hidden coordinator. Each authorized task sponsor
emits its own `AWARD` for the roster implied by the replicated result. If one
sponsor owns multiple tasks and can emit only one bounded record in a round,
stable task order selects the first emission. Delivered active awards are
public fixed commitments; every replica then recomputes the exact remaining
subproblem while reserving their members. This produces the same continuation
without private memory. Any overlapping active awards, private-field access,
or replica-order dependence is an integrity failure rather than a recoverable
experimental outcome.

Under this public-ledger/state-machine assumption the method is decentralized:
there is no distinguished planning process or hidden input, only deterministic
replicated computation and protocol-authorized per-task actuation. The study
must stop before launch if that property cannot be maintained.

## Frozen comparison

All three arms use six agents, Contract Net, `public_sweep`, truthful scripted
bids, the same delayed ledger, and twelve acting rounds:

1. `first_valid`, retained as the E2 reference;
2. task-local `exact_utility`; and
3. `joint_exact_allocation`.

Frozen canonical configuration:

- seeds: 23200--24199 inclusive;
- scenario families: single complementary, two disjoint, scarce capability,
  and oversubscribed;
- episodes: 1,000 seeds × 4 families × 3 arms = 12,000;
- acting rounds: 12, followed by the common drain/scoring round;
- process workers: 8;
- model calls and provider requests: 0; and
- output root:
  `outputs/recruitment_arena/e2d2_joint_allocation_v1`.

No seed, family, metric, or selector will be added or removed after inspecting
canonical outcomes. A failed integrity gate invalidates the campaign and
requires a new versioned output root after the fault is documented.

## Outcomes and hypotheses

The reward-safety outcomes are paired normalized task reward and
oracle-allocation coverage. `joint_exact_allocation` must not reduce either
relative to task-local `exact_utility`.

The primary residual outcome is paired true-information utility regret,
conditioned on matching oracle task reward. The directional hypothesis is that
joint allocation reduces this regret relative to task-local exact selection,
with the effect concentrated in the scarce and two-disjoint families.

Secondary outcomes are:

- achieved raw cost and achieved-minus-oracle raw-cost delta at equal reward;
- delivered control bytes and bounded-record count;
- mean lock round;
- paired better/tie/worse counts; and
- family-localized effects.

`first_valid` is a reference rather than the main residual contrast. Raw-cost
delta is descriptive because the oracle maximizes utility before minimizing
raw cost.

For each paired contrast, report the mean difference and a two-sided 95%
normal interval whose standard error clusters the four family observations by
seed. Also report the four family means and exact paired
better/tie/worse counts. No claim of LLM-agent or Alem execution efficacy will
be made from this provider-free mechanism study.

## Integrity gates

The canonical run must satisfy all of the following:

- all 12,000 expected seed-family-arm rows are present and unique;
- every selector audit names claimed information only;
- no serialized selector audit contains a true-capability or true-cost field;
- every selected roster has the exact task-requested size;
- no selected or locked rosters share an agent;
- all records are valid and all submitted transitions are accepted;
- roster agreement is complete;
- ordinary messages never cross team boundaries;
- every exported replay reproduces its terminal state and audit-chain hash;
- all shard and per-episode content hashes independently verify; and
- model calls, provider requests, and transport errors are all zero.

## Pre-launch validation

Essential tests must demonstrate:

1. exact cross-task exclusivity and lexicographic ties;
2. invariance to private truth fields and input iteration order;
3. deterministic episode and replay hashes; and
4. a constructed public-bid case where task-local greedy exact selection
   completes one task but joint reward-first allocation completes two.

The constructed case validates the mechanism's distinguishing behavior; it is
not part of the canonical outcome sample.

Canonical command, to be run only after this note and clean source are
committed:

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
