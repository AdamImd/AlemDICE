# E3a/E3b0 prospective protocol: Open Volunteer transfer into Alem

Status: **prospective and not runnable**

Protocol version: `alem-dice-e3b0-open-joint-hard-mining-v1`

Protocol date: 2026-07-23

Protocol source base:
`2092ebcb3e059f57acaed8e2b56460ec2b69ab8d`

No E3 implementation or provider request is authorized by this document. E3a
must be implemented and pass without provider use before E3b0 may make its one
preflight request. This file freezes the first narrow transfer study; it does
not modify the broader E3 design in `experiment_plan.md`.

## 1. Decision that motivates this study

The hosted E2b v4 screen supplied a positive synthetic mechanism signal for
Open Volunteer with the replicated `joint_exact_allocation` selector:

- 12/12 six-agent cells completed across seeds 22000--22002 and four scenario
  families;
- normalized reward and oracle-allocation coverage were 1.0 in every cell;
- all 15 oracle-allocable tasks were covered;
- 293 logical calls produced two invalid records, both followed by successful
  semantic repairs;
- there were zero transport errors;
- usage was 261,965 input tokens, 77,789 output tokens, and 72,168 reasoning
  tokens;
- the latest lock occurred in round 8; and
- every episode's deterministic directory replay matched.

This is an absolute mechanism result, not a comparative method effect. V4 had
no live comparator and does not establish that Open Volunteer is better than a
repaired Mutual Nomination mechanism. It also did not test navigation,
survival, action selection, or execution in Alem.

E3b0 asks the smallest next question: can the promoted formation mechanism
operate inside a real Alem action loop without changing the Source control
path or adding a commander, leader, planner, or second model call?

## 2. Estimand and claim boundary

E3b0 estimates, over three paired 200-tick Debug episodes, the difference
between:

1. the unchanged six-agent Source action and communication path; and
2. the same action path with opt-in Open Volunteer formation, replicated
   public-ledger allocation, and task-team-scoped ordinary messages.

The primary result is a mechanism-transfer screen:

- what fraction of locally observed, analysis-feasible hard-mining
  opportunities forms a valid team before its deadline; and
- whether a locked team completes at least one corresponding Alem mining
  event.

The paired Source arm provides a safety and performance reference. With only
three seeds, E3b0 cannot establish behavioral superiority, statistical
noninferiority, or a communication-efficiency claim. The treatment bundles
formation instructions, the public control bulletin, and team-scoped ordinary
routing. It therefore cannot separately identify the effect of the selector
from the effect of restricted routing. A later component study must add an
Open-Volunteer-plus-global-broadcast active control for that purpose.

E3b0 is limited to hard synchronous mining. It does not test soft mining,
handover, construction, crafting, elite combat, strict information isolation,
adversarial agents, populations above six, or long-term role alignment.

## 3. Shared environment and agent identity

Both paid arms use:

- environment: `Alem-Coop-Symbolic-Debug`;
- task: `alem/default`;
- physical agents: exactly 6;
- coordination difficulty: Easy;
- requested horizon: 200 environment ticks;
- world size: the existing 48-by-48 Debug overworld;
- `god_mode=false`;
- homogeneous GPT-5.4 nano action agents;
- OpenAI Responses adapter;
- provider reasoning effort: `high`;
- maximum output: 8,192 tokens;
- `store=false`;
- temperature: 1.0;
- timeout: 420 seconds;
- client retry setting: the Source value of 5;
- `robust_all`;
- `specific_collaborative`;
- normal reasoning and remembered reasoning;
- scratchpad enabled;
- communication enabled;
- `max_parse_retries=0`;
- no debriefs; and
- one episode worker.

All six physical-agent calls within a tick are concurrent. Every call observes
one immutable pre-action tick snapshot, and results are committed in ascending
agent ID order. Inactive-agent handling, action validation, action fallback,
environment stepping, reward calculation, and trajectory capture remain the
Source implementations.

The only shared environment amendment is an experiment-only cap on generated
hard synchronous requirements. The generator must consume the same random
draws as before, then clamp an outcome that would require all six agents to
three. Existing two-agent outcomes remain two-agent outcomes. The resulting
tasks therefore require two or three agents in both arms.

The cap must:

- be explicitly enabled in every E3a/E3b0 arm;
- be disabled by default;
- leave E0, E1, and ordinary Source runs unchanged;
- apply identically to mining requirements in both paid arms; and
- be recorded in the resolved configuration and world-state provenance.

This creates a fair E3b0 environment but means the Source arm is not a reuse of
the E1 N=6 episodes. "Unchanged Source" here means the policy, prompt,
communication route, timing, parser, and client identity are unchanged; both
arms receive the same prospectively controlled world distribution.

## 4. Exact Source control

The `source` arm retains:

- `team.topology=baseline`;
- `coordination.strategy=free`;
- the existing Source instruction prompt and output-format prompt;
- one optional free-form `<communication>` payload of at most 400 characters;
- one-tick-delayed broadcast to every other physical agent;
- the existing sender ordering and history representation; and
- exactly one action-model call per submitted physical-agent tick.

No TFP1 instructions, task cards, team state, selector advice, structured
opportunity data, or recruitment feedback may enter a Source prompt. The
opportunity extractor may run after the fact for paired analysis, but its
output must not affect Source prompts, messages, actions, or environment
state.

Before paid execution, golden provider-free tests must prove that, for a fixed
state and scripted completion:

- the complete Source instruction and turn-prompt bytes are unchanged from the
  source base;
- the parsed action and optional communication are unchanged;
- a Source message is delivered after one tick to exactly the other five
  agents in ascending receiver order;
- emitted and delivered bytes equal existing Source accounting; and
- disabling the E3 topology and world cap reproduces the source-base state and
  route hashes.

## 5. Dynamic treatment

The `open_joint_team_scoped` arm uses the same six action agents and the same
single action call. Its only behavioral additions concern task teams.

### 5.1 One response, one action, and at most one message

Every treatment response still contains exactly one Alem `<action>`. The
optional `<communication>` field is interpreted as follows:

- an unteamed agent may omit it or emit one canonical TFP1 control record;
- a locked team member may emit either one permitted TFP1 lifecycle record or
  one ordinary Source-style message;
- an agent may not emit both a control record and ordinary text in one tick;
  and
- no failed control parse or transition creates another model call.

The action is parsed and executed through the Source action path even when the
communication is absent or invalid. Invalid, oversized, stale, unauthorized,
or free-form unteamed messages are rejected, logged, and delivered to no one.

The TFP1 byte ceiling is 256 complete UTF-8 bytes. The runtime must inspect the
unmodified raw `<communication>` content before any existing display/history
length operation. It must reject an oversized message rather than truncate it.
Ordinary locked-team messages retain the Source 400-character limit.

### 5.2 Public control and private ordinary routes

Control records use the existing one-tick-delayed public TFP1 bulletin. An
accepted control is visible to all six agents after the delay. Once a roster
locks:

- ordinary text from a member is queued for the other members only;
- a nonmember receives no ordinary copy;
- an ordinary message is revalidated against the same lease at delivery;
- a cancelled, completed, or expired lease releases every member; and
- control and ordinary traffic are accounted separately.

The directory enforces one active task per agent, reciprocal acceptance,
canonical rosters, deterministic transition ordering, and replayable leases.
Requests, targeted `Give`, and global teammate observations remain unchanged,
so this is team-scoped ordinary communication rather than strict information
isolation.

### 5.3 Replicated public allocation

Open Volunteer uses the E2b v4 `joint_exact_allocation` selector without a
model call. Each of six logical replicas receives exactly:

- delivered public task cards;
- delivered public `APPLY` self-claims;
- current public leases; and
- the sorted eligible agent IDs.

It receives no pending control, hidden world state, other agents' unreported
inventory, true-feasibility label, terminal score, or oracle output. All six
replica hashes must agree before a roster plan may be published. The
directory independently verifies exact requested size, delivered
applications, claimed demand coverage, lease exclusivity, and cross-task
exclusivity.

Agents still publish `ACCEPT`, and the announcing sponsor still publishes
`LOCK`. The selector never chooses or alters an Alem action.

## 6. Hard-mining task definition

### 6.1 Legal opportunity source

The treatment wrapper exposes a structured list of hard synchronous mining
opportunities derived from the same local view bounds and light mask used by
the current text observation. A legal opportunity must satisfy all of:

- it is on the acting agent's current dungeon level;
- its map coordinate is inside that agent's current local view;
- the corresponding light-mask cell is visible;
- the coordination-map value is positive;
- the soft-coordination mask is false;
- the tile is a mineable resource rather than a construction site; and
- the clamped requirement is two or three agents.

Unlike the current human-readable cue section, the structured list must not
deduplicate sites by coordination value and softness. It records every legal
coordinate. It may clarify already legal information but may not include an
off-screen or dark opportunity.

### 6.2 Stable identity and public card

The stable task ID is:

```text
mine:<level>:<x>:<y>
```

where `x` is the map column and `y` is the map row. IDs must satisfy the TFP1
32-character identifier limit. A task is active only while the same
coordination opportunity remains at that level and coordinate.

The first accepted announcement defines:

- `required_size`: the clamped coordination-map requirement;
- `demand`: `(0,0,100)`;
- `reward`: 100;
- `deadline_round`: announcement tick plus 25; and
- `sponsor_id`: the announcing sender.

An `ANNOUNCE` is accepted only if its sender's archived legal-opportunity list
for that exact tick contains the task ID and matching required size. The
middleware does not announce automatically. Duplicate, altered, hidden,
expired, or no-longer-active announcements are rejected with stable codes.

### 6.3 Task-specific capability and cost

The capability axes remain `(combat, sustain, mining)`. The role base vectors
are:

| Specialization | Base capability |
| --- | --- |
| Warrior | `(80,20,20)` |
| Forager | `(20,80,20)` |
| Miner | `(20,20,80)` |

For a particular mining card, an agent's mining component becomes zero if its
current inventory lacks the pickaxe tier required for that resource. The
other two components remain the role-base values. The task-specific cost is:

```text
min(
  100,
  2 * visible_manhattan_distance
  + 20 * is_not_miner
  + 20 * has_health_at_most_3
  + 40 * lacks_required_pickaxe
)
```

The prompt supplies only the acting agent's own computed profile for each
currently legal opportunity. The model chooses whether to announce or apply
and must reproduce its own profile in a valid record. A valid delivered
`APPLY` makes that claim public. The runtime may validate the record against
the sender's archived own-profile projection, but it may not generate an
application on the sender's behalf.

The analysis-only true profile uses the same deterministic formula from the
archived state. It is unavailable to live allocation except through a valid
public self-claim.

### 6.4 Completion evidence

A formation is a valid lock of the selector's exact roster before the task
deadline. A locked task is an embodied completion only when:

- the corresponding hard synchronous mining event succeeds at the exact
  level and coordinate while the lease is active;
- at least the required number of locked members participate in the
  successful synchronized action;
- no nonmember participation is counted as recruited-team execution; and
- the coordinate-specific state change and relevant coordination counter
  delta agree.

System-side completion may close the lease using a digest of this evidence.
It may not choose an action or award a completion based solely on an agent
claim.

## 7. Provider-free E3a gate

E3a uses:

- output root: `outputs/alem_eval/e3a_recruitment_route_v1`;
- Debug Alem;
- six agents;
- seeds 14000 and 14001;
- 20 requested ticks;
- `god_mode=true`;
- the experiment-only team-size cap;
- the `source` and `open_joint_team_scoped` topologies; and
- deterministic scripted actors with zero client construction and zero model
  or provider calls.

A deterministic visible hard-mining fixture may be used to guarantee complete
protocol coverage. It must be identical across the two E3a arms and must be
identified as a routing fixture rather than behavioral evidence.

Across the two seeds, scripted treatment records must exercise:

1. a locally valid `ANNOUNCE`;
2. multiple `APPLY` records;
3. six identical selector-replica hashes;
4. reciprocal `ACCEPT`;
5. sponsor `LOCK`;
6. a team-private ordinary message;
7. rejection of one nonmember ordinary message;
8. rejection of one oversized control without truncation;
9. completion or cancellation and lease release; and
10. byte-identical export and deterministic replay.

E3a passes only if:

- all four episodes produce complete canonical artifacts;
- Source golden prompt and route checks pass;
- Source delivery retains exact one-tick all-peer semantics;
- treatment control delivery is public and one-tick delayed;
- unauthorized ordinary deliveries are exactly zero;
- roster agreement and lease exclusivity are 100%;
- all accepted announcements have legal local provenance;
- no record over 256 bytes enters the directory;
- every replay and audit-chain hash matches;
- the feature flag is absent or false in the ordinary default profile; and
- model calls, provider attempts, transport errors, and tokens are all zero.

Any E3a failure blocks the paid preflight and E3b0.

## 8. Paid E3b0 matrix

The frozen matrix is:

| Seed | First arm | Second arm | Requested ticks |
| ---: | --- | --- | ---: |
| 14100 | `source` | `open_joint_team_scoped` | 200 each |
| 14101 | `open_joint_team_scoped` | `source` | 200 each |
| 14102 | `source` | `open_joint_team_scoped` | 200 each |

This is six episodes and at most 7,200 decision calls:

```text
2 arms * 3 seeds * 200 ticks * 6 agents = 7,200
```

Natural all-agent death is a valid terminal outcome and reduces actual ticks
and calls. Requested and executed ticks and physical-agent exposure must both
be reported.

### 8.1 Staging

After E3a passes:

1. make one GPT-5.4-nano-high compatibility preflight using a distinct cache
   route;
2. run the paired seed-14100 canary sequentially;
3. recompute every integrity and exposure gate from canonical artifacts; and
4. only if the canary is promotable, run seeds 14101 and 14102 in their frozen
   orders.

A passing infrastructure canary requires at least one legal hard-mining
opportunity to become observable in the dynamic episode. If seed 14100 has no
such exposure, the paid matrix stops as uninformative. A new fixture or seed
protocol must be frozen under a new identity; seed 14100 may not be silently
replaced.

### 8.2 Concurrency

- Six physical-agent calls are concurrent within each tick.
- There is one episode worker.
- Arms and seeds are sequential in the frozen order above.
- E3b0 may not overlap another hosted experiment using the same provider
  account.
- The two paired arms may not run concurrently.

This makes tick latency comparable. The provider-free E3a checks may use
ordinary test-level parallelism if their artifacts remain isolated.

## 9. Hypotheses and promotion gates

### H3b0.1: formation transfer

At least 80% of analysis-feasible observed opportunities form a true-feasible
locked roster before deadline.

An observed opportunity enters the denominator only when it remains legally
visible to at least one alive treatment agent for two consecutive ticks while
active. It is analysis-feasible at its first qualifying tick when an exact-size
roster of alive, actionable agents on that level covers `(0,0,100)` under
their archived true task profiles. Unannounced qualifying opportunities remain
in the denominator.

If fewer than three analysis-feasible observed opportunities occur across the
matrix, or fewer than two seeds contain at least one, the formation result is
**uninformative**, not a method failure or pass.

### H3b0.2: formation-to-execution bridge

At least one valid locked roster completes its exact Alem mining task before
lease expiry. Formation-to-execution conversion and latency are otherwise
descriptive at this sample size.

A formation pass with zero embodied completion is a negative execution-transfer
result. It does not support an Alem coordination claim.

### H3b0.3: communication signal

The dynamic arm is expected to reduce ordinary delivered bytes relative to
paired Source broadcast. The preregistered paper-scale target is at least 30%.
At three seeds, this is reported as a signal with raw paired differences, not
as a confirmed efficiency or noninferiority claim. Control-plane bytes are
reported separately and are never subtracted from total communication cost.

### Integrity gates

Every canonical paid episode must satisfy:

- zero unauthorized cross-team ordinary deliveries;
- 100% roster agreement for every lock;
- one-team-per-agent lease exclusivity;
- 100% accepted-announcement local provenance;
- zero accepted records above 256 complete UTF-8 bytes;
- zero silent control-message truncation or repair;
- exact selector-replica agreement;
- exact directory replay and audit-chain agreement;
- action parse validity of at least 95%;
- recruitment-transition validity of at least 95%;
- zero incomplete provider responses;
- zero budget abstentions;
- zero unrecovered provider failures;
- a stable resolved model accepted by the frozen model-binding rule;
- exactly one decision call per submitted physical-agent tick and no other
  behavioral model call; and
- complete episode, response, route, opportunity, and state artifacts.

The Source arm must additionally pass its golden prompt and route invariant.
An integrity failure invalidates behavioral interpretation even if performance
is high.

## 10. Performance and cost outcomes

### Mechanism outcomes

Report per seed and in aggregate:

- unique legal opportunities seen;
- analysis-feasible observed opportunities;
- announcement rate and first-seen-to-announcement latency;
- valid and rejected records by stable code;
- applications per task;
- selector plans and replica hashes;
- lock count, true-feasible lock count, and lock latency;
- roster churn, cancellations, completions, and expiries;
- formation rate;
- formation-to-execution conversion;
- lock-to-execution latency; and
- unauthorized delivery attempts and accepted copies.

### Alem outcomes

Report both raw seed values and arm means for:

- Base percentage;
- Coordination percentage;
- Total percentage;
- mean per-agent episode return;
- unique team achievements;
- summed per-agent first unlocks;
- raw cumulative environment event counters;
- hard synchronous mining attempts and successes;
- deaths and death causes;
- alive and actionable agent steps;
- requested and executed ticks; and
- action parse success, failure, and inactive skips.

### Communication and inference cost

Report:

- emitted control records and payload bytes;
- delivered control copies and fan-out-weighted bytes;
- emitted ordinary messages and payload bytes;
- delivered ordinary copies and fan-out-weighted bytes;
- rejected messages and bytes;
- total decision calls and calls per actual agent tick;
- provider attempts and transport errors;
- input, output, reasoning, cached, and cache-write tokens;
- summed model latency;
- per-tick complete wall time; and
- maximum observed within-tick call concurrency.

No outcome may use requested ticks as the efficiency denominator when an
episode terminates early.

## 11. Hard campaign ceilings

The single preflight and paid matrix have these immutable ceilings:

- logical responses: 7,201;
- observed provider attempts: 9,001;
- prompt bytes per attempt: 64,000;
- output tokens per successful response: 8,192;
- worst-case provider-attempt token reservation:
  659,017,216; and
- observed input-plus-output token stop: 100,000,000.

The worst-case reservation is:

```text
9,001 * (64,000 prompt bytes + 1,024 framing tokens + 8,192 output tokens)
= 659,017,216
```

One prompt byte is charged as one input token for conservative pre-dispatch
reservation. A provider attempt, including a retry, must reserve independently
before dispatch. Returning actual usage reconciles but never restores a
provider-attempt count. An unresolved crash reservation remains spent and
blocks automatic resume.

The paired seed-14100 canary has subordinate ceilings:

- logical responses: 2,401, including the shared preflight;
- provider attempts: 3,001;
- worst-case provider-attempt token reservation: 219,721,216; and
- observed input-plus-output token stop: 30,000,000.

The 100-million and 30-million observed-usage stops are operational cost
limits checked before every new dispatch and after every reconciliation. The
larger reservation limits are the conservative maximum exposure under the
frozen prompt and output bounds. A cap conflict produces no request, stops the
campaign, and is not converted into a Noop behavioral observation.

The launcher must archive a zero-call dry-run projection before the preflight.
Changing any cap changes the configuration hash and requires a new study
identity.

## 12. No-go and stop rules

Stop before E3b0 preflight if:

- E3a fails any gate;
- the Source prompt or route golden changes;
- the baseline feature flag changes an ordinary non-E3 world;
- any task metadata includes a dark or off-screen opportunity;
- middleware can announce, apply, or select an action on behalf of a model;
- raw control content can be truncated before validation;
- treatment adds a commander, leader, planner, or second call; or
- the committed implementation/config/protocol hashes are dirty or
  inconsistent.

Stop the hosted campaign and preserve every artifact if:

- a call, provider-attempt, prompt, or token cap conflicts;
- requested or resolved model identity changes;
- a provider response is incomplete;
- an unrecovered provider failure invalidates a cell;
- any cross-team ordinary copy is delivered;
- any accepted announcement lacks local provenance;
- directory replicas or replay hashes disagree;
- an accepted control exceeds 256 bytes;
- a Source prompt or route invariant fails;
- the seed-14100 canary has no legal opportunity exposure; or
- the canary fails any integrity gate.

After all three seeds, classify:

- insufficient exposure as **uninformative**;
- adequate exposure but formation below 80% as a **negative formation
  transfer**;
- formation at or above 80% but zero task execution as a **negative execution
  transfer**;
- mechanism and execution gates passing as a **positive bridge signal**; and
- any integrity failure as **invalid for behavioral interpretation**.

Do not repair or overwrite a failed live root. Any prompt, task mapping,
deadline, selector, routing, cap, model, seed, or horizon amendment requires a
new protocol version and output root.

## 13. Canonical outputs and resumption

Planned roots:

- E3a: `outputs/alem_eval/e3a_recruitment_route_v1`;
- E3b0: `outputs/alem_eval/e3b0_alem_transfer_v1`.

Each root must contain:

- exact CLI invocation and resolved configuration;
- protocol, source, dependency-lock, and dirty-tree hashes;
- campaign and per-cell manifests;
- preflight artifact and model binding;
- append-only attempt and provider-reservation ledgers;
- atomic complete, failed, and cancelled markers;
- raw structured prompts and exact provider completions;
- provider response IDs, status, model, usage, retries, and latency;
- per-tick action and parse records;
- per-agent legal-opportunity projections;
- public cards and submitted raw TFP1 records;
- parse and transition codes;
- selector public inputs, six replica outputs, and hashes;
- lease and membership transitions;
- receiver-indexed control and ordinary route envelopes;
- rejected-message records;
- coordinate-specific execution evidence;
- environment trajectories and replay states;
- terminal evaluator summaries;
- deterministic directory replay and audit-chain hashes; and
- the exact JSON/CSV used by every result table.

A complete marker must bind the episode, debug, route, opportunity, state, and
ledger artifacts by SHA-256. Resume skips only a comprehensively valid complete
cell. A partial artifact, unresolved reservation, invalid marker, unexpected
file, or copied artifact stops automatic resume. A failed cell may be retried
once under the existing campaign cap only after its original attempt is
immutably archived and the launcher can prove that no provider request is
unresolved.

## 14. Analysis boundary

The three paired seeds support:

- exact raw mechanism accounting;
- integrity decisions;
- paired descriptive effects;
- detection of gross performance or routing failure; and
- a decision about whether a broader E3 study is warranted.

They do not support:

- a claim that Open Volunteer outperforms Mutual Nomination;
- a claim that dynamic teams outperform Source or fixed teams;
- a 5% performance-noninferiority claim;
- a statistically confirmed 30% communication reduction;
- attribution of a performance effect separately to recruitment or routing;
- generalization beyond hard synchronous mining; or
- strict information-isolation claims.

Report all three paired points. Confidence intervals, if shown, are descriptive
and must not replace the small-sample qualification. The preregistered
ten-seed directional rule and final noninferiority rule remain reserved for a
later E3 stage.

## 15. Source and result provenance

This protocol branch starts from the clean E2 v4 implementation commit:

```text
2092ebcb3e059f57acaed8e2b56460ec2b69ab8d
```

The immutable Source scaling measurement base is:

```text
49bc152e2b70609aa9a4518b86b1f8f1fced5a14
```

The independently reviewed E1 analysis correction tip is:

```text
1839b28e929d6bcca48588e8a8cbf04c84051df4
```

The E3 implementation must begin only after E1b finishes and from a clean
consolidation of the E2 v4 source and the accepted E1 correction. The older
E2 arena-integration tip
`7f1f5f0e654e1481c2c4bfb83c930a1f2696f0d7` is not a sufficient E3 base.

The hosted E2b v4 source-of-truth root is:

```text
/home/adam/Desktop/AlemDICE/outputs/recruitment_llm/e2b_luna_screen_v4
```

Its full manifest is bound to source commit
`2092ebcb3e059f57acaed8e2b56460ec2b69ab8d` and configuration SHA-256
`120b3e64c56af032827f922d7f02980d66079f2f1cdc542316c7e50b869bf295`.
The full-manifest and canary-gate SHA-256 values observed when this protocol
was frozen are:

```text
run_manifest_full.json
d263c7aecbc032cc117b1084c9eac246e3b04ac614592befa949d4664b50ad02

canary_gate.json
d1ae8b56f9032d93043f102efa5566f47dab5c446442cbebe43d8bce645015c1
```

The dependency lock SHA-256 at the E2 source base is:

```text
d75773f66d8a5af4ea339ef8be9c4f9a2cec08e088a74dcacd9f4e746654b128
```

The E2b v4 protocol at the source base still describes hosted v4 as unrun.
Before E3 implementation is frozen, a separate read-only result update must
bind and report the completed v4 artifacts. This prospective E3 protocol does
not rewrite that earlier document.

## 16. Why E3b0 is not runnable

At source commit `2092ebc`, all E2 formation code is standalone from the Alem
evaluator. The following implementation blockers remain:

1. `baselines/llm/config/config.yaml` has no recruitment topology.
2. `baselines/llm/eval_utils/evaluator.py` accepts only the existing
   baseline, bodyless-leader, and embodied-commander topologies.
3. The evaluator delivers one global sender-to-message map to every receiver
   and hard-codes full peer fan-out for Source-like topologies.
4. `TeamDirectory` and `joint_exact_allocation` are not connected to the Alem
   tick loop.
5. The language wrapper collapses multiple same-type coordination cues and
   exposes no complete structured legal-opportunity list.
6. World generation selects either two or the full physical population and
   does not apply the existing maximum-agent field.
7. the RobustAll communication path can truncate content before a TFP1
   reject-never-truncate decision.
8. No treatment prompt supplies visible task cards, an own-profile projection,
   delayed public state, roster advice, or lease state.
9. No coordinate-specific task-completion bridge closes a lease from verified
   environment evidence.
10. No E3 profile, launcher, durable campaign budget, summarizer, canonical
    output schema, or E3-specific test exists.

The minimum implementation delta is therefore:

- add one opt-in E3 topology without altering the baseline branch;
- add a treatment-only agent context and communication instruction path;
- preserve raw complete communication for strict TFP1 validation;
- add receiver-indexed treatment inboxes and route journals;
- integrate `TeamDirectory` and six replicated public selectors;
- expose every legally visible hard-mining candidate from the existing masks;
- add the default-off requirement cap;
- add coordinate-specific completion evidence;
- add separate control and ordinary communication accounting;
- add deterministic export/replay and complete artifact binding;
- add E3a scripted actors and Source golden regressions; and
- add the staged, bounded E3b0 launcher and read-only analysis.

Until those changes are implemented, committed, independently reviewed, and
E3a passes, E3b0 remains prospective and no provider call may be made.
