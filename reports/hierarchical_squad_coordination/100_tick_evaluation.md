# 100-tick Source vs. embodied-commander evaluation

Status: **COMPLETE — DESCRIPTIVE ONE-SEED RESULT; PREREGISTERED GATE FAILED**

Protocol: `alem-dice-embodied-commander-study-v1`, stage `100`

Date authorized: 2026-07-22

Branch: `research/embodied-squad-commander`

## Question and scope

This evaluation compares two coordination conditions using the same
`gpt-5.6-luna` worker model:

1. `baseline`: unchanged Source decentralized behavior and peer broadcast.
2. `embodied_commander_broadcast`: a serial Agent 0 planning phase creates
   leased assignments before the same three parallel worker action calls.

Calling these the “two models” refers to two agent-system conditions, not two
different provider model IDs. Using one model in both arms isolates the
coordination intervention.

This is one matched Easy seed (`12000`) with a 100-tick cap per arm. It can
provide mechanism and descriptive outcome evidence, but cannot establish an
average treatment effect or statistical significance.

## Frozen configuration

| Field | Value |
| --- | --- |
| Profile | `embodied_commander_100` |
| Task/difficulty | `alem/default`, Easy |
| Seed | `12000` in both arms |
| Tick cap | 100 per arm |
| Agents | Three physical and three logical participants |
| Model/API | `gpt-5.6-luna`, OpenAI Responses |
| Reasoning/temperature | `none` / `1.0` |
| Maximum output | 8,192 tokens per logical call |
| Episode workers | One; arms run sequentially |
| Within-tick actions | Three concurrent worker calls |
| Commander review/lease | Every 5 ticks / 10 ticks |
| Unscheduled review cap | 10 |
| Debriefs/W&B | Disabled / disabled |

The baseline permits at most 300 worker calls. The commander arm permits the
same 300 worker calls, 20 scheduled planning calls, and at most 10 unscheduled
repair/event planning calls. The study ceiling is 630 logical calls, excluding
transport retries. Each preflight attempt adds exactly two logical calls; the
extra troubleshooting attempts are audited separately from the study below.

## Frozen structural gate

The automated mechanism gate passes only if:

- both canonical episodes complete with zero transport errors;
- the commander produces at least one valid plan;
- plan validity is at least 0.90;
- active-plan coverage is at least 0.90;
- both action-parse rates are at least 0.95;
- conditional status validity is at least 0.90;
- valid-status coverage is at least 0.80 of all eligible agent-turns; and
- no unauthorized plan, stale status, or structural hidden-state violation is
  accepted.

Valid-status coverage over all eligible agent-turns is reported separately from
conditional status validity. The commander executor contract requires a status
on each active turn so the feedback loop is meaningfully exercised; the lower
0.80 gate tolerates occasional malformed or omitted outputs without hiding them.

The last three zero counters are construction invariants, not independent
semantic detectors. They must not be used as substitutes for raw-trace review.

## Required trace review

The final report will inspect every commander planning call; all invalid,
replacement, event, death, reward, and parse-failure ticks; and systematic
10-tick action samples. It will check:

- baseline prompts contain no commander intervention;
- planner prompts contain only Agent 0's legal observations and previously
  validated status reports;
- every accepted plan covers exactly Agents 0, 1, and 2;
- all three same-tick worker prompts receive the accepted full plan and the
  correct highlighted assignment;
- Agent 0 does not revise assignments during its action phase;
- wingmen do not invent objectives or reassign work;
- status versions/tasks match the sender's active assignment and evidence is
  grounded in its visible observation; and
- actions materially pursue assignments rather than only formatting valid
  statuses.

Raw `*_debug.jsonl` is the semantic source of truth. The existing HTML viewer
does not render commander planning records. Serial plan-before-action ordering
is supported by the same-tick prompt dependency and frozen code path; the debug
schema does not contain independent per-call start/end timestamps.

## Results

### Executive finding

The commander mechanism worked mechanically, and this seed contains a real
positive coordination signal, but the experiment does **not** establish that the
hierarchy is better overall. Relative to Source, the commander arm obtained one
successful handover, two coordination achievements, a 32.2% higher mean team
return, and twice the normalized total reward. It simultaneously covered fewer
achievement types, made less normal-task progress, rejected 11 of 30 planner
outputs, idled the whole team for five ticks after a lease expired, and cost
24.6% more tokens and 31.6% more wall time per tick.

The preregistered structural gate therefore **failed** on planner validity:
19/30 calls were accepted (63.33%), below the 90% threshold. All other automated
checks passed. Because this is one stochastic matched seed, every outcome
contrast below is descriptive rather than a treatment-effect estimate.

### Provenance and completion

| Field | Recorded value |
| --- | --- |
| Run ID | `20260723T021700Z_embodied_commander_100` |
| Frozen source | `4d9bb8b4d640e96cfcf04a38a75a70e90347e003` |
| Branch | `research/embodied-squad-commander` |
| Seed | `12000` in both arms |
| Canonical attempts | One per arm; both complete |
| End condition | 100 ticks, `environment_truncated`, both arms |
| Provider transport | 0 errors in 630 study requests |
| Model | `gpt-5.6-luna`, reasoning effort `none` |
| W&B | Disabled |

The two arms ran sequentially. Within each tick, the three worker calls ran in
parallel. In the treatment, the optional Agent 0 planning call ran serially
before those worker calls.

### Outcome comparison

Percentages below use the canonical `Team/*` metrics published in the study
summary. “Achievement coverage” is the fraction of the relevant achievement
catalog reached, not a success probability.

| Metric | Source baseline | Commander | Commander minus Source |
| --- | ---: | ---: | ---: |
| Ticks | 100 | 100 | 0 |
| Mean team return | 4.033 | 5.333 | +1.300 (+32.2%) |
| Total reward / maximum | 1.862% | 3.723% | +1.862 pp (+100.0%) |
| Team achievement coverage | 7.527% | 6.452% | -1.075 pp (-14.3%) |
| Unique achievement types | 7 | 6 | -1 |
| Normal achievement coverage | 10.606% | 6.061% | -4.545 pp |
| Normal achievements | 7 | 4 | -3 |
| Coordination achievement coverage | 0.000% | 7.407% | +7.407 pp |
| Coordination achievements | 0 | 2 | +2 |
| Normal reward / maximum | 3.226% | 1.843% | -1.383 pp |
| Coordination reward / maximum | 0.000% | 6.289% | +6.289 pp |
| Action parse rate | 100.0% (269/269 active turns) | 100.0% (300/300) | 0 pp |
| Inactive skips / deaths | 31 / 1 mob death | 0 / 0 | -31 / -1 |
| Coordination successes / resolved attempts | 0 / 10 | 1 / 5 | 0% to 20% |
| Handover successes / resolved attempts | 0 / 2 | 1 / 3 | 0% to 33.3% |
| Logical calls / requests / errors | 300 / 300 / 0 | 330 / 330 / 0 | +30 / +30 / 0 |
| Total tokens | 1,541,646 | 1,920,993 | +379,347 (+24.6%) |
| Episode wall time | 278.288 s | 366.090 s | +87.802 s (+31.6%) |
| Wall time per tick | 2.783 s | 3.661 s | +0.878 s (+31.6%) |
| Worker-round wall time | 198.009 s | 201.196 s | +3.187 s (+1.6%) |

The return gain was highly concentrated: Source agent returns were
`[2.1, 6.0, 4.0]`, whereas commander returns were `[2.0, 12.0, 2.0]`. Agent 1
received the treatment's coordination payoff; the commander and Agent 2 did not
improve. This is a positive event-level signal, not evidence of broadly better
team productivity.

### What worked

1. **The hierarchy was actually in the action loop.** All 19 accepted planner
   responses covered exactly Agents 0, 1, and 2. At every accepted review, each
   same-tick worker prompt contained the same accepted version and highlighted
   that worker's correct task. The treatment injected a command contract into
   all 300 worker turns. Source had zero commander records, command-contract
   prompt hits, or commander-plan routes.

2. **The initial resource pipeline was coherent and executed.** At tick 0 the
   miner reported that it lacked wood. Agent 0 gathered wood, the miner issued a
   request at tick 10, Agent 0 used `Give to Agent 2` at tick 11, Agent 0 placed a
   table at tick 16, and Agent 2 made a wood pickaxe at tick 17. These actions
   match the successive assignments rather than merely producing valid status
   syntax.

3. **A later handover succeeded.** Version 15 brought the separated agents
   toward a tree; version 16 assigned Agent 0 to complete the active handover and
   Agent 1 to support it. At tick 88 all three agents selected `Do`, Agent 1
   received reward `+10`, and the environment reported one mining-handover
   success, `HANDOVER_COMPLETE`, and `COORD_MINE_HANDOVER`. Source had three
   handover setups, no successes, and zero coordination reward. This is the
   strongest positive signal in the run.

4. **All treatment agents survived.** Source lost Agent 0 to a mob and recorded
   31 inactive action turns; the commander arm had no death or inactive turn.
   With one seed, this may reflect different trajectories rather than a robust
   safety benefit, but it is another favorable event-level observation.

5. **The structured feedback channel was reliable syntactically.** All 300
   eligible turns emitted a valid, current-version SCP1 status: 267 `ACTIVE`, 1
   `BLOCKED`, 17 `COMPLETE`, and 15 `WAITING`. There were no accepted stale or
   wrong-assignment statuses and no worker attempted to send a `<squad_plan>`.

6. **Most latency overhead was isolated to planning.** Worker-round wall time
   rose only 1.6%. The treatment's planning phase consumed 86.934 seconds, which
   explains almost all of the 87.802-second episode-time increase. This makes
   batching, lower-frequency review, or a smaller planner plausible optimization
   targets.

   The 30 planner calls used 123,544 tokens, only 6.43% of treatment tokens and
   32.57% of the incremental tokens. The treatment worker calls themselves still
   used 16.59% more tokens than Source because every action prompt carried the
   full plan and status contract. Optimizing only the planner will therefore not
   remove the token overhead.

### What did not work

1. **Planner format reliability was poor.** Eleven of 30 calls were rejected,
   failing the only automated gate. The failures were schema/envelope mistakes,
   not transport failures or parser-detectable assignment-graph mistakes; the
   complete audit is below.

2. **The repair policy can halt the squad.** The shared ten-call event/invalid
   repair budget was exhausted by tick 22. A bad scheduled response at tick 65
   retained version 14 only through tick 69; another bad response at tick 70
   arrived after that lease expired. For ticks 70--74, all three agents received
   the no-plan safety contract, selected `Noop`, and reported `WAITING`. Version
   15 restored activity at tick 75. This exactly accounts for 95% plan coverage.

3. **Syntactic status validity overstated semantic reliability.** For example,
   Agent 0 reported version 16's handover task as `COMPLETE` at ticks 85--87
   merely because it had used `Do`; the environment did not award the handover
   until Agent 1's action at tick 88. At tick 47, Agent 2 similarly reported a
   synchronized tree task `COMPLETE` while explicitly saying it was “awaiting
   observed chop result.” The parser correctly validates identity, version,
   task, state, and length, but it does not establish that evidence proves the
   declared state.

4. **Schema-valid plans could still be physically inconsistent.** Version 12 at
   tick 45 assigned Agents 0 and 2 to synchronized `Do` actions on the tree at
   `(20,15)`. Agent 0 was adjacent to that tree, but Agent 2's validated reports
   placed it around `(12,20)` beside a different tree at `(12,19)`. The planner
   nevertheless called Agent 2's position “confirmed adjacent.” Both agents
   obeyed by using `Do` on their respective local trees, their statuses described
   synchronized execution, and no coordination reward occurred. By tick 49 they
   reported no wood. Exact schema validation cannot replace geometric or
   evidence-grounding checks.

5. **The commander over-replanned and sometimes overcommitted.** It installed 17
   replacement plans, kept only 2, and changed 37 member assignments. From tick
   19 onward it repeatedly described a “confirmed” or “usable” ladder even while
   status evidence said no ladder was visible. No agent descended within the
   horizon. This suggests weak belief calibration and insufficient hysteresis.

6. **Role allocation was unbalanced.** Agent 1 selected `Noop` 48 times. The
   water-maintenance assignments alone produced 41 `Noop` actions between ticks
   22 and 69. The treatment obtained fewer normal achievements and less normal
   reward despite its one high-value coordination success. The commander
   preserved role obedience at the cost of underusing a wingman.

7. **Prompt authority did not fully constrain action semantics.** At tick 89,
   after declaring its local-tree task complete, Agent 2 unilaterally chose
   `Place Table` before receiving a new plan. It did not create a competing plan,
   so the structural unauthorized-plan counter remained zero, but the action was
   outside the bounded tree-gather assignment. Semantic scope drift needs an
   action-to-assignment audit rather than an envelope counter.

8. **Explicit acknowledgements were unused.** Assignment acknowledgement was
   0/51 even though every status parsed. Workers moved directly to `ACTIVE`, so
   the protocol cannot distinguish “received and accepted this new assignment”
   from routine execution. Either first-version `ACTIVE` should count as an
   implicit acknowledgement or the prompt/state machine must require one
   explicit `ACK` before `ACTIVE`.

### Failed planner-call audit

The parser's contract is intentionally strict:

- the response must contain one `<squad_plan>...</squad_plan>` envelope;
- `REPLACE` must contain exactly `operation`, `objective`, and `assignments` at
  the top level; and
- the middleware, not the model, owns `issued_tick`, `review_tick`, and
  `expiry_tick`.

There were only two root causes. Eight responses returned a semantically valid
JSON object but omitted the required envelope. For example, tick 5 began:

```text
{"operation":"REPLACE","objective":"Transfer wood to the miner, establish basic tools, and secure water while preparing to descend.", ...}
```

The required shape was:

```text
<squad_plan>{"operation":"REPLACE", ...}</squad_plan>
```

Three responses used the envelope but copied middleware-owned fields from the
displayed active plan. Tick 12 ended:

```text
... ],"issued_tick":12,"review_tick":17,"expiry_tick":22}</squad_plan>
```

Those fields would let a model influence lease timing if accepted, so the whole
proposal was atomically rejected. Replaying all 11 raw outputs through the exact
parser showed that adding only the envelope repaired all eight untagged outputs,
and removing only those three fields repaired the other three. No rejected call
had a secondary parser-detectable JSON, assignment-coverage, dependency,
identifier, length, or action error. This replay does not certify that the plans
were physically sensible.

| Tick | Trigger | Exact failure | Runtime consequence and recovery |
| ---: | --- | --- | --- |
| 5 | scheduled | Missing envelope | Kept v2; valid v3 installed at tick 6. |
| 10 | scheduled | Missing envelope | Kept v3; valid v4 installed at tick 11. |
| 12 | `event:COMPLETE` | Added `issued_tick`, `review_tick`, `expiry_tick` | Kept v4; valid v5 installed at tick 13. |
| 18 | `event:COMPLETE` | Added the three timing fields | Kept v6; valid v7 installed at tick 19. |
| 21 | `event:COMPLETE` | Added the three timing fields | Kept v7; valid v8 installed at tick 22. |
| 25 | scheduled | Missing envelope | Kept v8 through tick 29; cap exhaustion deferred retry to tick 30, which installed v9. |
| 55 | scheduled | Missing envelope | Kept v13 through tick 59; v14 installed at tick 60. |
| 65 | scheduled | Missing envelope | Kept v14 through tick 69. |
| 70 | `invalid_retry` | Missing envelope | v14 had expired; no active plan at ticks 70--74; v15 installed at tick 75. |
| 80 | scheduled | Missing envelope | Kept v15 through tick 84; v16 installed at tick 85. |
| 95 | scheduled | Missing envelope | Kept v17 through tick 99; horizon ended before another retry. |

`parsed_plan` in an invalid debug record is the older retained effective plan,
not a partially accepted version of the rejected output. At tick 70 it is
`null`, because no older lease remained. Rejections never extended a lease.

The study failures are separate from the engineering preflight. The first
preflight launch stopped before a provider call because its cache key exceeded
the provider limit. The next provider-reaching attempt produced an invalid
plan, but the then-current logger did not preserve enough raw output to classify
it. A diagnostic retry was rejected for inventing non-environment
synchronization actions such as `report_status` and `report_recon`; after the
prompt listed canonical physical actions, the final two-call preflight passed.
Across provider-reaching attempts, preflight used six logical calls. Preflight
calls and failures are excluded from the 630-call study ledger and all outcome
metrics.

### Manual trace and safety audit

| Check | Finding |
| --- | --- |
| Baseline isolation | Pass: 100 records, zero commander-planning records, command prompts, or commander routes. |
| Plan membership | Pass: every accepted plan covered exactly agents `[0, 1, 2]`. |
| Same-tick delivery | Pass: every accepted version and correct highlighted task appeared in all three same-tick worker prompts. |
| Action authority | Pass only at the structural level: no worker sent a squad plan and Agent 0 used the executor contract; Agent 2's tick-89 table placement shows semantic scope drift. |
| Status identity/version/task | Pass: 300/300 valid; zero stale or wrong-assignment acceptances. |
| No-plan behavior | Pass as designed: all 15 turns at ticks 70--74 used `VERSION=0`, `TASK=NONE`, `WAITING`, and `Noop`. |
| Planner input boundary | Pass by code path and prompt inspection: each of 30 planner prompts had only public rules, Agent 0's legal observation, the accepted plan, validated statuses, and private planner scratchpad. |
| Semantic status grounding | Mixed: many statuses match visible positions/actions, but some `COMPLETE` states preceded environmental confirmation and v12 described two different trees as one synchronized target. |
| Task pursuit | Mixed-positive: the early craft pipeline and late handover were materially executed; prolonged ladder search, passive forager assignment, and tick-89 scope drift were unproductive. |

The trace establishes plan-before-action dependency because the new plan is in
each same-tick worker prompt. It does not log independent provider-call start and
end timestamps, so it cannot separately prove temporal non-overlap. Likewise,
the zero hidden-state/authority counters are construction invariants rather than
independent semantic detectors.

### Achievement-label audit issue

The canonical episode's convenience `agent_*_achievements` dictionary and the
generated aggregate `achievement_counts` label the tick-88 pair as
`COORD_MINE_COAL_SOFT/HARD`. That label conflicts with the raw tree-handover
trace and with the value-indexed `user_info`, which reports
`HANDOVER_COMPLETE`, `COORD_MINE_HANDOVER`, one handover success, and reward 10.
The cause is a reporting-order defect: the achievement enum is declared in an
order that differs from its numeric values, while the convenience serializer
zips a value-indexed vector against declaration iteration order. The report
therefore treats the value-indexed `user_info` and trace as authoritative and
records the event as a handover, not coal mining. Outcome totals are unaffected,
but label-specific aggregate output should be fixed before publication.

### Gate result

| Preregistered check | Result |
| --- | --- |
| Both episodes complete with zero transport errors | Pass |
| At least one valid plan | Pass |
| Valid plans at least 90% | **Fail: 63.33%** |
| Active-plan coverage at least 90% | Pass: 95.0% |
| Both action parse rates at least 95% | Pass: 100% / 100% |
| Conditional status validity at least 90% | Pass: 100% |
| Valid-status coverage at least 80% | Pass: 100% |
| Zero accepted unauthorized/stale/hidden-state violations | Pass structurally |

Overall automated gate: **FAIL**.

### Decision and next experiment

Do not interpret this run as validation and do not launch the preregistered
200-tick study unchanged. First:

1. use provider-constrained structured output or deterministic envelope/schema
   normalization while preserving the original raw response in the audit log;
2. give invalid-plan repair its own budget or guarantee a next-tick repair so
   event churn cannot disable format recovery;
3. define evidence-backed `COMPLETE` transitions and either require `ACK` or
   count first-version `ACTIVE` as an acknowledgement;
4. add assignment hysteresis and an explicit “new evidence required” rule for
   repeated search objectives;
5. fix value-indexed achievement labeling; and
6. rerun the 100-tick gate on at least three paired seeds before escalating the
   horizon.

The hypothesis worth carrying forward is narrow: a leader-assigned hierarchy
may increase execution of coordination-specific tasks. This run does not support
the stronger claim that it improves general team achievement coverage or
efficiency.

### Audit artifacts

- Published machine summary: `Results/20260723T021700Z_embodied_commander_100.json`
- Published generated overview: `Results/20260723T021700Z_embodied_commander_100.md`
- Ignored canonical run root:
  `outputs/alem_eval/20260723T021700Z_embodied_commander_100/`
- Treatment semantic trace:
  `embodied_commander_broadcast/easy/alem/default/default_run_00_debug.jsonl`
- Source semantic trace:
  `baseline/easy/alem/default/default_run_00_debug.jsonl`
- Append-only ledgers: each arm's
  `easy/alem/default/attempt_ledger.jsonl`
