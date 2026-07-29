# E2b v4 completed hosted Open Volunteer screen

> **Status: complete, integrity-valid, and promotable to a bounded E3 Alem mechanism transfer.** This is a six-agent structured recruitment screen, not an Alem efficacy, scalability, long-horizon role-alignment, or Open-versus-Mutual result.

## Bound source

- Preserved root: `outputs/recruitment_llm/e2b_luna_screen_v4`
- Source commit: `2092ebcb3e059f57acaed8e2b56460ec2b69ab8d`
- Root tree SHA-256: `36ba9a6445293896fec943f0145b075520a6f524136d8324e8a8e162957c221c`
- Canary manifest SHA-256: `0560637ee7e75aa3992450dfe82eac9d0fbb44649a917590c4f71f3d84a46cc4`
- Canary gate SHA-256: `d1ae8b56f9032d93043f102efa5566f47dab5c446442cbebe43d8bce645015c1`
- Full manifest SHA-256: `d263c7aecbc032cc117b1084c9eac246e3b04ac614592befa949d4664b50ad02`
- Reservation ledger SHA-256: `6096f58d2496d0b9fa0062dce9e4aa31adc9971179e7f5e13366e26f9759cb4f`
- Provider binding: requested/resolved `gpt-5.6-luna`; `high` reasoning; 4,096-token output allowance.

## Canary and promoted full stage

| Stage partition | Cells | Full reward | Full coverage | Calls | Repairs | Invalid | Input tokens | Output tokens | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Canary (separate) | 2 | 2/2 | 2/2 | 36 | 0 | 0 | 29,874 | 7,504 | 24.412s |
| Promoted invocation: new cells only | 10 | 10/10 | 10/10 | 257 | 2 | 2 | 232,091 | 70,285 | 89.432s |
| Combined evidence | 12 | 12/12 | 12/12 | 293 | 2 | 2 | 261,965 | 77,789 | — |

The full manifest is cumulative: it began with the two completed canary cells and dispatched only the remaining ten cells with three cell workers.

## Efficacy and allocation quality

- All 15/15 locks were truly feasible; true-infeasible locks: 0.
- Exact whole-allocation oracle matches: 10/12; exact task rosters: 12/15.
- Mean true-utility regret: 0.005660; mean true raw-cost delta: 4.916667.
- Mean lock round: 3.467; mean executed acting rounds: 5.167.

| Family | Cells | Reward/coverage | Calls | Mean rounds | Exact allocation | Utility regret | Cost delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oversubscribed` | 3 | 3/3 | 54 | 4.000 | 2/3 | 0.003889 | 4.666667 |
| `scarce_capability` | 3 | 3/3 | 99 | 6.667 | 3/3 | 0.000000 | 0.000000 |
| `single_complementary` | 3 | 3/3 | 54 | 4.000 | 3/3 | 0.000000 | 0.000000 |
| `two_disjoint` | 3 | 3/3 | 86 | 6.000 | 2/3 | 0.018750 | 15.000000 |

The two non-oracle allocations retained full reward and feasibility but committed before every useful bid became public. `oversubscribed/22001` incurred +14 raw cost; `two_disjoint/22002` incurred +45 and did not lock its second team until round 8. This is an observed association in two cells, not a causal estimate.

## Reliability, usage, and traffic

- 293 logical calls = 291 initial decisions + 2 semantic repairs; 293 actual provider attempts.
- Transport errors: 0; incomplete calls: 0; max-output truncations: 0.
- Invalid initial decisions: 2/291 (0.687%); both repairs returned valid abstentions.
- Usage: 261,965 input + 77,789 output = 339,754 actual tokens; 72,168 reasoning tokens (92.77% of output).
- Mean/max call latency: 3.281s/12.004s; summed per-call latency: 961.447s; summed round decision wall: 281.622s.
- Structured control: 133 submissions, 127 accepted and 6 rejected transitions; 6,340 payload bytes and 30,410 delivered bytes (2534.2/episode).
- Bid fidelity: 75/76 exact capability vectors and 76/76 exact costs. One claim used `[0,80,20]` instead of `[20,80,20]`; it did not change feasibility or reward in that cell.

The two rejected semantic decisions were:

- `scarce_capability/22001`, round 3, agent 4: `TFP1|TYPE=ACCEPT|TASK=scarce.t1|MEMBERS=0,4` failed `transition_preflight:state.roster_not_ready`; repair outcome `abstain`.
- `two_disjoint/22002`, round 4, agent 1: `TFP1|TYPE=APPLY|TASK=disjoint.t1|CAP=80,20,20|COST=45` failed `transition_preflight:state.agent_already_teamed`; repair outcome `abstain`.

## Context against deterministic E2d2

The provider-free E2d2 joint-exact arm achieved 1.0 reward and coverage with zero utility regret and zero cost delta across 4,000 episodes, using 3558.5 delivered control bytes per episode. Hosted v4 matched reward and coverage but not perfect oracle allocation. This is contextual only: E2d2 used scripted truthful Contract Net behavior, different seeds, and different phase timing.

## Decision

Promote Open Volunteer/Public Sweep with truth-free joint exact allocation to a **bounded, paired E3 Alem transfer** while preserving Source action behavior. Record reward/events, feasible locks, role violations, control bytes, provider tokens, latency, repairs, bid fidelity, and roster regret. Add a preregistered bid-closure/quorum or provisional-lock-grace ablation because the only allocation misses followed commitment on incomplete public bids.

Do not interpret this screen as evidence of Open superiority over a redesigned Mutual mechanism, population scalability, embodied task efficacy, or long-horizon role alignment.
