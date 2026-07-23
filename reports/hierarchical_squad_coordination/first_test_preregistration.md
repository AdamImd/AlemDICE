# First test preregistration: 30-step embodied commander gate

Status: **DRAFT FOR REVIEW — NOT RUN**

Protocol version: `alem-dice-embodied-commander-study-v1`

Prepared: 2026-07-22

Implementation branch: `research/embodied-squad-commander`

Parent source revision: `af5c16f91cd8a978e78d13ca270bea9bebf35c36`

No Luna preflight or experimental call has been made for this protocol. This
document freezes the intended first test for review before any paid run.

## 1. Purpose and permitted conclusion

The first test asks whether the embodied commander mechanism is wired correctly
enough to justify a longer experiment. It compares the unchanged Source baseline
against one commander treatment for a single matched 30-step Easy seed.

This test can support only the conclusion that the protocol is operational and
semantically behaving as intended. With one seed and one episode per arm, it
cannot establish that hierarchy improves reward, achievements, coordination, or
generalization. Outcome differences will be recorded as diagnostics, not treated
as efficacy evidence.

## 2. Research question

Can Agent 0 create valid, bounded, leased assignments for all three embodied
agents, have the agents acknowledge and execute those assignments without
assuming planning authority, and preserve the Source action path's parse
reliability and legal information boundary for 30 environment steps?

The mechanism passes only if both its machine-checkable invariants and a manual
semantic trace review pass.

## 3. Fixed experimental design

| Item | Frozen value |
| --- | --- |
| Environment/task | `alem/default` |
| Difficulty | Easy |
| Seed | `12000` in both arms |
| Episodes | One per arm; two total |
| Step cap | 30 per episode |
| Physical/logical agents | Three / three |
| Worker roles | Existing Source warrior, forager, and miner roles |
| Model/API | `gpt-5.6-luna` through `openai_responses` |
| Reasoning | `none` |
| Temperature | `1.0` |
| Maximum output | 8,192 tokens per logical call |
| Provider timeout/retries | 420 seconds; five configured retries; 2-second delay |
| Debriefs | Disabled |
| W&B | Disabled |
| Episode concurrency | One episode at a time (`num_workers=1`) |
| Arm order | Source baseline, then commander broadcast |
| Optional star ablation | Excluded from this test |

The two arms use the same seed, environment, three worker model configurations,
roles, action parser, observation construction, memory settings, and simulator
step cap. Arm-specific prompt-cache keys prevent cache reuse across conditions.

### Arm A — Source baseline

`team.topology=baseline`

This is the existing Source behavior: each of the three role agents chooses its
own action using the normal `RobustAllAgent` path and one-tick peer communication.
It receives no squad contract, leased plan, structured status requirement, or
extra planner call.

### Arm B — embodied commander with broadcast statuses

`team.topology=embodied_commander_broadcast`

Agent 0 remains one of the three embodied agents. A separate serial model call,
owned by Agent 0, creates or reviews the squad plan before the action phase. This
call is not a fourth agent and cannot act in the environment. After planning, the
same three worker calls choose their actions concurrently. The full accepted plan
is shown to all three agents, with each agent's own assignment highlighted. The
normal Source peer-message route remains present; valid structured status records
are peer-broadcast and also made available to the next commander review.

The commander treatment is intended to change only team planning and role
authority. It does not replace the Source action policy, simulator, observations,
role prompts, action parser, or peer-communication behavior.

## 4. Commander protocol frozen for this test

### 4.1 Authority and information boundary

Only the serial Agent 0 planning phase may create or revise the objective and
assignments. In its later action call, Agent 0 is a commander-executor and may not
revise the plan. Agents 1 and 2 are wingman-executors: they retain tactical
autonomy for movement, prerequisites, immediate survival, and local execution,
but may not create competing plans or reassign work.

The planner receives only:

- the public game rules;
- Agent 0's legal long- and short-term text observations;
- the currently accepted plan, if any;
- middleware-validated agent status reports since the prior review; and
- a private planner scratchpad capped at 1,000 characters.

It does not receive other agents' private observations, raw simulator state, or
future state. The full accepted plan is subsequently delivered to every agent so
that dependencies and synchronization requests are common knowledge.

### 4.2 Plan schedule and leases

- Scheduled reviews occur at ticks 0, 5, 10, 15, 20, and 25.
- Each accepted plan has a 10-tick lease.
- A transition to `BLOCKED`, `COMPLETE`, or `EMERGENCY`, or an invalid plan,
  requests a review on the next tick.
- Unscheduled repair/event reviews share a cap of three in this 30-step episode.
- A scheduled review may return `KEEP` only while a valid plan is active.
- Event and invalid-repair reviews must return `REPLACE`.
- A malformed proposal is rejected atomically. An existing unexpired plan remains
  active without having its lease extended; with no valid plan, agents are told
  to use only immediate survival action or `Noop` and report `WAITING`.

### 4.3 Accepted plan contract

The planner must return exactly one `<squad_plan>` JSON payload. A replacement
must contain one unique assignment for every member (0, 1, and 2), with a unique
task ID, bounded directive, target, completion condition, dependencies, and an
optional synchronization window. Dependencies must reference tasks in that plan
and form an acyclic graph. Synchronization actions must be legal Alem actions and
their tick window must fall inside the lease.

### 4.4 Status contract

An executor may emit one bounded status in this form:

```text
SCP1|TEAM=squad-0|VERSION=<n>|TASK=<task-id>|STATE=<state>|EVIDENCE=<observed fact>|REQUEST=<optional request>
```

Allowed states are `ACK`, `ACTIVE`, `BLOCKED`, `COMPLETE`, `EMERGENCY`, and
`WAITING`. Middleware rejects malformed records, wrong-team records, stale plan
versions, and task IDs that do not match the sender's assignment. A wingman
attempt to place `<squad_plan>` in communication is counted but never accepted as
a plan.

## 5. Timing and call budget

The baseline has at most 90 logical action calls: 30 ticks times three agents.
The treatment has the same 90 action calls plus at most nine planning calls: six
scheduled reviews and up to three unscheduled repair/event reviews. The study cap
is therefore **189 logical calls**. The optional preflight is separate and adds
exactly two bounded logical calls, for a maximum of **191 calls before review of
the first result**. Provider-level transport retries are not included in these
logical-call caps and are recorded separately.

Within a treatment tick, ordering is:

1. If review is due, make one serial Agent 0 planning call and validate it.
2. Inject the accepted plan (or no-plan safety contract) into all three prompts.
3. Make the three worker action calls concurrently.
4. Step the environment, validate statuses, and schedule any next-tick event
   review.

The serial planning latency is measured separately from worker-round latency.

## 6. Preflight gate (two calls, no episode)

The preflight will be run only after this document and implementation are
approved. It makes exactly:

1. one Luna worker request that must parse as `<action>Noop</action>`; and
2. one Luna planning request that must parse as a complete `REPLACE` plan for
   agents 0, 1, and 2.

Failure stops the study. Preflight responses are connectivity/schema checks and
are not experimental observations.

## 7. Preregistered automated pass gate

The 30-step test passes its automated gate only if every condition below is true:

- both episode artifacts are complete;
- both arms record zero transport errors;
- the treatment accepts at least one valid plan;
- treatment active-plan coverage is at least 0.80;
- treatment action-parse rate is at least 0.95;
- treatment status-parse rate is at least 0.90;
- zero unauthorized plan attempts are accepted;
- zero stale status records are accepted; and
- the structural hidden-state guard records zero violations.

These thresholds are wiring thresholds, not evidence that the treatment is
better than baseline. In particular, `accepted_unauthorized_plans` and
`accepted_stale_statuses` are parser/runtime invariants, while the current
hidden-state counter reflects construction-level isolation rather than semantic
inspection. They do not replace the manual review below.

## 8. Required manual semantic trace review

The automated result is provisional until a reviewer examines every planning
response (at most nine) and the relevant per-tick prompts, communications, and
actions. Record pass/fail and a trace reference for each item.

| Review item | Required observation | Result / trace reference |
| --- | --- | --- |
| Planner input boundary | No private wingman observation or raw simulator state appears in a planner prompt | Pending |
| Plan timing | A plan review, when due, finishes before that tick's worker calls | Pending |
| Complete delivery | Each worker receives the same full accepted plan and its own highlighted assignment | Pending |
| Commander authority | Agent 0 plans only in the serial phase and does not revise assignments in its action output | Pending |
| Wingman authority | Agents 1 and 2 execute or report on assigned work and do not originate objectives or reassign agents | Pending |
| Tactical autonomy | Safety/prerequisite deviations remain local and do not become competing team plans | Pending |
| Grounded evidence | Status evidence is consistent with information visible to its sender | Pending |
| Version/task checks | Accepted statuses use the active version and sender's assigned task | Pending |
| Event handling | New blocked/complete/emergency reports cause a next-tick review, subject to the cap | Pending |
| Invalid output handling | Invalid plans do not partially apply or extend the prior lease | Pending |
| Behavioral compliance | Actions are plausibly directed toward assignments rather than merely producing parseable statuses | Pending |
| Baseline isolation | No commander directive/status requirement appears in Source baseline prompts | Pending |

Any clear hidden-information leak, unauthorized plan revision, cross-arm prompt
contamination, or fabricated status evidence is a semantic failure regardless of
the automated gate.

## 9. Recorded diagnostics (not first-test promotion criteria)

The report will show, for each arm, steps, termination reason, team return,
achievement percentages, reward percentage, action-parse rate, tokens, provider
requests, transport errors, wall time, and wall time per step. The treatment also
records plan calls and validity, plan coverage and age, status validity, assignment
acknowledgment and switching, event-to-review latency, plan expiries, and rejected
stale/wrong-assignment/unauthorized records.

Paired deltas for seed 12000 will be displayed for transparency. Their signs will
not be used to change the gate or claim success after seeing the result.

## 10. Audit artifacts

Before the episodes begin, the launcher writes an immutable run manifest and
resolved base configuration. The manifest records the source commit and branch,
Git status, `uv.lock` hash, resolved base and arm hashes, prompt-contract hash,
exact commands, cache keys, seeds, arm order, and call caps. A paid run normally
requires a clean committed worktree.

The run must preserve:

- `commander_study_manifest.json` and `resolved_base_config.yaml`;
- canonical complete episode JSON for both arms;
- append-only attempt ledgers, including failed-attempt token and transport cost;
- per-step debug records containing planning and action-call audit fields;
- `commander_episodes.csv`;
- `commander_study_summary.json`; and
- `commander_study_report.md` plus the copies published under `Results/`.

Incomplete or failed attempts remain auditable but are not substituted for the
canonical matched episodes. Resume must validate the frozen manifest before
retrying.

## 11. Current implementation validation (no model calls)

The test suite was deliberately reduced to 27 safety-critical contracts. It now
covers one environment reset/step smoke, action parsing and provider transport,
attempt accounting, frozen experiment profiles, commander authority/lease/privacy
rules, one paired Source-vs-commander evaluator path, and launcher/summary gates.
The complete retained suite passed in 52.82 seconds. Ruff also passed on the
touched core test files.

Exhaustive simulator dynamics, rendering permutations, superseded team-leader
studies, and obsolete experiment-report tests are intentionally outside this
branch's regression gate. None of the retained local checks contacted Luna or
produced experimental data.

## 12. Decision rule after the first test

- **Proceed to planning the 200-step study:** preflight passes, every automated
  gate passes, every manual review item passes, and the reviewer explicitly
  approves continuation.
- **Revise and repeat the 30-step test:** failures appear reparable without
  changing the research question. Document the failure, issue a new protocol
  version, and do not overwrite the failed run.
- **Stop the commander approach:** the legal information boundary cannot be
  maintained, authority separation is not behaviorally credible, or the mechanism
  repeatedly fails to produce/execute valid plans.

Even a full pass does not authorize an efficacy claim. It authorizes only a
separately reviewed three-seed, 200-step exploratory comparison.

## 13. Commands reserved for after approval

From the repository root:

```bash
# Inspect the exact arms, overrides, cap, and output path; no files or API calls.
./commands.sh commander-study --stage 30 --dry-run

# Exactly two logical Luna schema/connectivity calls, then exit.
./commands.sh commander-study --stage 30 --preflight

# One 30-step Source episode followed by one matched commander episode.
./commands.sh commander-study --stage 30
```

Do not add `--include-star` to this first test. Do not use `--allow-dirty` for the
reviewed run unless the exception and exact dirty paths are documented in advance.

## 14. Reviewer sign-off

- [ ] The Source arm is an acceptable unchanged baseline.
- [ ] The treatment changes only the intended team-planning/authority mechanism.
- [ ] The planner's legal information boundary is acceptable.
- [ ] The six scheduled reviews, ten-tick lease, and three unscheduled-call cap
      are acceptable.
- [ ] The automated thresholds are acceptable as wiring gates.
- [ ] The manual semantic checklist is sufficient.
- [ ] The 189-call study cap (191 including preflight) is acceptable.
- [ ] Approve the two-call Luna preflight.
- [ ] Approve the 30-step paired test after a successful preflight.

Requested changes / approval notes:

> Pending reviewer input.
