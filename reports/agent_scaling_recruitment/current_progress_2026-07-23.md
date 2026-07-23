# Current Research Progress — 2026-07-23

## Bottom line

The staged program now has two substantive signals:

1. The unchanged Source baseline has a non-monotonic population curve in the
   tested environment. Performance improves through three agents, remains
   similar at four, and falls at six while communication, token use, and tick
   time continue to rise.
2. Open Volunteer with truth-free joint exact allocation can form bounded teams
   reliably in the synthetic six-agent RecruitmentArena screen.

Neither result yet validates embodied dynamic-team efficacy. The six-agent
Source decline is materially confounded by a native task-generation defect:
hard synchronous mining and construction can require all six agents even though
only four non-overlapping cardinal positions can target one tile. The E3
action-loop transfer is specified but remains unimplemented and not runnable.

## Stage status

| Stage | Status | What it establishes |
| --- | --- | --- |
| E0 | Complete: pass | Variable-agent evaluator, accounting, replay, and rendering work at `N={1,2,3,4,6}`. |
| E0.1 | Structural pass; literal horizon fail | Ten 200-requested-tick infrastructure episodes produced complete artifacts; three passive-policy episodes ended naturally before tick 200. |
| E1 | Complete: descriptive | All 15 Source population cells completed and audited. Native `N=6` is a stress characterization, not an unconfounded population effect. |
| E2b v2/v3 | Complete: failed diagnostics | Output truncation was removed in v3, but frozen Mutual Nomination still self-omitted and formed a true-infeasible roster. |
| E2b v4 | Complete: mechanism pass | Open Volunteer/joint exact completed 12/12 cells and 15/15 true-feasible locks. |
| E3a | Pending | Provider-free task-provenance, feasibility, routing, replay, and Source-invariance implementation and gates. |
| E3b0 | Protocol frozen; not runnable | Three-seed controlled Source-versus-recruitment action-loop screen; blocked on E3a. |

## E1 completed Source population screen

### Method and integrity

- Physical populations: `N={1,2,3,4,6}`.
- Paired seeds: `13100`, `13101`, `13102`.
- Requested horizon: 200 ticks.
- Model: GPT-5.4 nano, high reasoning.
- Unchanged Source action path: `robust_all`,
  `specific_collaborative`, scratchpad, existing parser, one call per physical
  agent per tick, and one-tick-delayed full peer broadcast.
- Complete cells: 15/15.
- Actual environment ticks: 2,991.
- Episode logical responses: 9,573.
- Episode provider attempts: 9,574.
- Campaign totals including preflight: 9,574 logical responses and 9,575
  provider attempts.
- Provider faults: one recovered `APIConnectionError`; zero lost logical calls,
  incomplete responses, or unrecovered failures.
- Terminations: 14 environment truncations at tick 200 and one natural
  termination at `N=3`, seed 13102, tick 191.

The final audit reconstructed all 91 episode fields, all 300 population
estimates, all 600 paired estimates, 4,219 ordinary communication routes,
token/latency sums, response identities, retry accounting, attempt ledgers, and
the exact Noop partition from raw artifacts.

### Primary outcomes

Values are seed means. The intervals in the machine summary are descriptive
paired bootstraps over only three seeds.

| N | Base% | Coord.% | Total% | Return/agent | Unique team unlocks | Summed agent unlocks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.379 | — | 3.379 | 7.133 | 7.33 | 7.33 |
| 2 | 5.376 | 3.774 | 4.699 | 14.283 | 10.33 | 16.00 |
| 3 | 12.596 | 5.870 | 9.752 | 21.678 | 19.33 | 36.33 |
| 4 | 8.602 | 10.901 | 9.574 | 20.375 | 19.33 | 42.33 |
| 6 | 8.141 | 4.822 | 6.738 | 11.883 | 15.00 | 42.33 |

Key paired Total-score directions:

| Contrast | Difference (points) | Descriptive interval |
| --- | ---: | ---: |
| `N3 - N1` | +6.372 | `[+4.008, +9.540]` |
| `N4 - N1` | +6.195 | `[+3.176, +9.327]` |
| `N4 - N3` | -0.177 | `[-3.457, +5.319]` |
| `N6 - N3` | -3.014 | `[-3.191, -2.926]` |
| `N6 - N4` | -2.837 | `[-8.511, +0.532]` |

All three `N6 - N3` paired differences were negative. However, the structural
feasibility defect below prevents interpreting this as degradation caused only
by adding agents.

### Unnormalized coordination events

These event families use heterogeneous units and must not be summed.

| N | Coordination attempts | Successes | Aggregate rate | Successful transfers | Requests | Revives | Deaths |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 65 | 11 | 16.9% | 12/12 | 20 | 0 | 0 |
| 3 | 60 | 5 | 8.3% | 20/20 | 33 | 0 | 4 |
| 4 | 85 | 11 | 12.9% | 71/71 | 70 | 4 | 1 |
| 6 | 153 | 9 | 5.9% | 78/78 | 79 | 2 | 2 |

`N=6` generated 80% more attempts than `N=4` but fewer successes. Across the
three `N=6` episodes, only one of 69 recorded synchronous attempts succeeded.
Simple Request/Give transfer was mechanically reliable, while simultaneous
coordination was not.

### Communication and compute cost

| N | Calls/episode | Total tokens/episode | Delivered bytes/episode | Tick wall | Actionable fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 0.863M | 0 | 4.546 s | 1.000 |
| 2 | 400 | 2.593M | 19,671 | 6.899 s | 0.958 |
| 3 | 591 | 3.877M | 66,736 | 8.661 s | 0.869 |
| 4 | 800 | 5.591M | 148,177 | 10.329 s | 0.941 |
| 6 | 1,200 | 9.126M | 421,603 | 14.648 s | 0.901 |

From `N=4` to `N=6`, calls rose 50%, total tokens 63.2%, delivered bytes 184.5%,
and tick wall time 41.8%. `N=4` and `N=6` had the same summed agent-unlock mean,
42.33, but `N=6` had fewer unique team unlocks. More agents duplicated
achievements rather than broadening the team repertoire.

### Noop accounting

Mean `N=6` effective Noops per episode:

- 119.33 inactive turns (70.5%);
- 34.67 intentional actionable waits (20.5%);
- 15.33 parser fallbacks (9.1%); and
- zero validation or unexplained residual fallback.

Inactive means dead, sleeping, or resting; the environment masks those turns to
`Noop`. Inactivity does not fully explain the result because `N=6` had a higher
actionable fraction than `N=3`.

One concrete parser mismatch was nested action markup:
`<action><action>Do</action></action>`. The agent announced that it was acting,
but the frozen parser correctly failed closed to `Noop`. This was real, but not
large enough to explain the full six-agent result.

### Native six-agent feasibility defect

Source world generation can assign a hard synchronous mining or construction
requirement equal to `player_count`. At six agents:

- a hard target can require six simultaneous agents;
- `Do` targets only the cardinal tile an agent faces; and
- player collision rules prevent position overlap.

Only four non-overlapping cardinal neighbors exist, so a six-agent target can
be impossible.

In seed 13102, agents mentioned the impossible tree at `(22,26)` in 182
messages across 80 ticks and made 75 explicit “all six” or “six-agent”
announcements. The seed delivered 514,150 communication bytes but obtained zero
coordination successes and Total score 2.660. This shows both structural
infeasibility and a gap between communicated plans and synchronized actions.

Scientific classification: E1b is valid as a **Source-as-implemented stress
characterization**. It does not establish an intrinsic scalability limit.

## E2 completed formation signal

The hosted E2b v4 Open-only screen completed 12/12 cells across three seeds and
four scenario families:

- normalized reward: 1.0 in 12/12;
- oracle-allocation coverage: 1.0 in 12/12;
- true-feasible locks: 15/15;
- calls: 293 (291 initial plus two semantic repairs);
- provider transport/incomplete/truncation failures: zero;
- exact whole allocations: 10/12;
- exact task rosters: 12/15;
- mean true-utility regret: 0.005660; and
- mean raw-cost delta: +4.9167.

This supports bounded transfer of Open Volunteer plus truth-free joint exact
allocation. It does not show that Open is superior to a repaired Mutual method,
that teams can execute in Alem, or that role alignment persists over long
missions.

## Current claim ledger

| Abstract-level claim | Current status |
| --- | --- |
| Variable-population infrastructure | Supported by E0/E0.1. |
| Source population behavior | Described by E1, with a native `N=6` feasibility confound. |
| Bounded decentralized team formation | Positive synthetic signal from E2b v4. |
| Dynamic teams improve embodied execution | Untested; requires E3. |
| Communication is bounded at scale | Not true for Source full broadcast; team-scoped reduction remains untested. |
| Distributed task planning | Not established; E2 tests formation/allocation, not embodied planning. |
| Context management preserves role coherence | Untested. |
| Long-term role alignment | Untested. |

The present abstract therefore overstates completed evidence if written in the
past tense. “We develop” can describe implementation, but claims that mechanisms
support scaling or preserve long-term role alignment must remain hypotheses
until the corresponding stages complete.

## Next work

1. Implement provider-free E3a:
   - opt-in hard-mining requirement cap, disabled by default;
   - locally visible task-instance provenance;
   - strict raw-message TFP1 classification;
   - replicated truth-free public allocation;
   - task-bound leases and deterministic replay;
   - team-only ordinary message routing and exact byte accounting; and
   - baseline prompt/action/route invariance.
2. Run the complete E3a adversarial gate matrix with zero provider calls.
3. If every gate passes, run the bounded three-seed E3b0 Source-versus-dynamic
   screen under the identical feasibility-controlled world distribution.
4. Separately run native-versus-feasibility-fixed Source at `N=6`, with `N=4`
   as a negative control, before explaining the E1 regression.
5. Update the combined LaTeX/PDF and this progress ledger from canonical
   artifacts after each stage.

## Audit references

- Raw E1 root:
  `outputs/alem_eval/e1_source_scaling_3seed_200_v2`
- Trusted measured Source commit:
  `49bc152e2b70609aa9a4518b86b1f8f1fced5a14`
- E1 evidence commit:
  `af1417d2029772ba0d5bc351d6b25ddc6b84c7dd`
- E1 study-manifest SHA-256:
  `fc1ae646345f644fc645ce8f09aa005c259cef23d76844f9bbdf9c6dfe642b7e`
- E1 summary JSON SHA-256:
  `3818cf6ecdde3be7ce9c8daf16a04ad5ad759cac402ff53212a4051b68fc0e10`
- E2b v4 result:
  `Results/e2b_v4_hosted_results_v1.{json,md}`
- Frozen E3 protocol:
  `reports/agent_scaling_recruitment/e3b0_protocol.md`
