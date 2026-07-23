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

- what fraction of prospectively classified, primary-eligible hard-mining
  opportunity lifecycles forms a valid team before its deadline; and
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
- a Source ordinary message beginning with the literal text `TFP1` still
  follows that unchanged five-peer Source route and is not treatment-parsed;
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

Before treatment message routing, the runtime classifies the exact,
unmodified raw `<communication>` content. If its content begins at byte zero
with the literal ASCII prefix `TFP1`, it is a control attempt. That
classification is irreversible: malformed, oversized, stale, unauthorized,
or transition-invalid attempts are rejected and can never fall through to
ordinary team text. This classifier is enabled only for the E3 treatment;
Source messages, including Source text that happens to begin with `TFP1`,
retain the unchanged Source route.

The TFP1 byte ceiling is 256 complete UTF-8 bytes. The runtime must inspect the
unmodified raw `<communication>` content before any existing display/history
length operation. It must reject an oversized message rather than truncate it.
Ordinary locked-team messages retain the Source 400-character limit.

`DECLINE` is not available in the E3b0 treatment prompt or permitted-record
set. A submitted `DECLINE` is rejected as `method.record_not_allowed`, and the
canonical delivered-decline collection is always empty. Absence of a response
or omission of `<communication>` is the only way not to participate; it does
not create a public decline.

### 5.2 Public control and private ordinary routes

Agent-authored control records use the existing one-tick-delayed public TFP1
bulletin. At its delivery tick, one accepted record produces exactly five
routed network copies, one for every physical peer except its sender. The
sender sees the same accepted canonical record through its local public
ledger, so logical public visibility is exactly six agents. Routed peer copies
and bytes, sender-local ledger visibility, and six-agent logical visibility
must be recorded separately; the sender-local view is not a sixth network
copy. Once a roster locks:

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
model call. Each of six logical replicas receives exactly the same canonical
public projection:

- the sorted active public cards, including each card's own demand and
  required size;
- delivered public `APPLY` self-claims associated with those active cards;
- the public phase of every admitted nonterminal card and all current public
  leases; and
- the single sorted vector of configured physical agent IDs that are publicly
  unleased in directory state.

The eligible-ID vector is exactly the sorted configured physical agent IDs
that are publicly unleased in the directory's `agent_to_task` state. It is not
filtered by application presence, life/death, actionability, dungeon level,
position, visibility, inventory, capability truth, or any other environment
state. Delivered `APPLY` records remain a separate public self-claim input;
the selector can place an ID on a card only through that delivered claim. The
delivered-decline input is the canonical empty tuple and is not an extension
point in E3b0.

For each sender and card instance, the selector reads the canonical `CAP` and
`COST` values exactly from the latest live delivered claim defined in
Section 6.6. It does not recompute either value from archived environment
truth.

For selector input, an active card is defined only by public directory state:
it is admitted, its public deadline has not passed, and its public phase is
`ANNOUNCED` or `FORMING`. The selector does not re-read the coordinate; the
prior tick-start closure pass is responsible for publishing any
vanished-coordinate transition. Locked cards remain in the public
phase-and-lease projection but are not allocation candidates.

It receives no pending control, alive/actionable flags, agent levels or
positions, hidden world state, other agents' unreported inventory,
true-feasibility label, terminal score, or oracle output. All six replica
hashes must agree before a roster plan may be published. The directory
independently verifies exact requested size, delivered applications, coverage
of that card's size-dependent demand, public phase, the public unleased-ID
vector, lease exclusivity, and cross-task exclusivity.

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
- `demand`: `(0,0,80)` when `required_size=2`, or `(0,0,100)` when
  `required_size=3`;
- `reward`: 100;
- `deadline_round`: the announcement's emission tick plus 25; and
- `sponsor_id`: the announcing sender.

An `ANNOUNCE` is accepted only if its sender's archived legal-opportunity list
for that exact tick contains the task ID and matching required size. The
middleware does not announce automatically. Duplicate, altered, hidden,
expired, or no-longer-active announcements are rejected with stable codes.

The demand is a deterministic function of the verified size, not a free
model-supplied choice. An announcement whose `DEMAND`, `SIZE`, reward, or
deadline does not match the canonical card is rejected before capacity is
considered.

### 6.3 Bounded admission and card generations

The public directory admits at most two nonterminal cards at a time.
Announced, forming, and locked cards each consume one slot; completed,
cancelled, and expired cards consume none.

After the tick-start closure pass in Section 6.4, delivered announcements are
considered in canonical `(delivery_tick, sender_id, emission_sequence)` order.
Whenever a physical-opportunity lifecycle owns no nonterminal card and is
eligible to open an instance, the first announcement that passes raw parsing,
local provenance, canonical-field, activity, and capacity checks is the
canonical card for that instance. It pins the sponsor and all card fields. A
later announcement for that nonterminal task is rejected as
`state.duplicate_task`.

There is no eviction, replacement, priority overwrite, or implicit queue. If
two other nonterminal cards own the two slots, an otherwise admissible
announcement for a distinct task is rejected as
`capacity.active_task_limit`. The rejection and the blocked opportunity are
archived, but no card is created and no old card changes. A later retry
requires a newly emitted announcement with its own current provenance; the
rejected record is never queued.

A terminal card is archived and releases its slot. The same stable wire task
ID may be announced again only by a record emitted after that terminal
transition and backed by a new legal-opportunity projection from its own
emission tick. A queued or copied pre-closure record cannot reopen it. The
directory assigns a monotonically increasing internal `instance_id` and keys
audit/replay state by `(task_id, instance_id)`; applications, accepts, selector
plans, leases, and deadlines never carry across instances. The public task ID
remains `mine:<level>:<x>:<y>`.

Physical-opportunity lifecycles and public-card instances are distinct. A
reannouncement while one coordinate remains continuously active does not
create another analysis opportunity. If the coordinate vanishes and later
becomes a legal hard-mining opportunity again, that is a new physical
lifecycle with fresh provenance.

### 6.4 Tick-start closure and admission order

At the start of every treatment tick, before any pending model-authored
control is delivered or any new `ANNOUNCE` is admitted, the runtime performs
one deterministic closure pass over nonterminal cards sorted by
`(task_id, instance_id)`. It reads only archived evidence from the just
completed Alem environment step:

1. If a locked card has the coordinate-specific success evidence in
   Section 6.7, the directory records the canonical system `COMPLETE`
   transition with code `task.completed`.
2. Otherwise, if the exact level and coordinate no longer contains the same
   active hard synchronous mining opportunity, the directory records the
   canonical system `CANCEL` transition with reason `coordinate_vanished` and
   code `task.cancelled.coordinate_vanished`.
3. Existing deterministic deadline expiry is then applied.

Completion takes precedence when successful mining both produces the required
event and removes the resource. A disappearance without locked-roster evidence
is never credited as completion. Every accepted system closure immediately
releases its roster lease and public-card slot before announcement admission.
The system transition, evidence or activity digest, release, resulting public
snapshot, and visibility to all six public replicas are append-only audit
events and must reproduce exactly in directory replay. A system lifecycle
publication is not an agent-authored network control: it records zero
agent-to-agent routed copies, six logical visibility entries, and its
fan-out-weighted logical publication bytes separately. Those logical bytes
remain part of total control-plane accounting.

Model-authored control remains one-tick delayed. Tick-start system closures
are not model messages or additional calls; they are canonical directory
transitions published before the current tick's admission and selector
projection.

### 6.5 Task-specific capability and cost

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
currently legal opportunity and that opportunity's canonical size-dependent
demand. The model chooses whether to announce or apply and must reproduce its
own profile in a valid record. A valid delivered `APPLY` makes that claim
public. The runtime validates the record against the sender's archived
own-profile projection for that exact card instance, but it may not generate
an application on the sender's behalf.

The analysis-only true profile uses the same deterministic formula from the
archived state. It is unavailable to live allocation except through a valid
public self-claim. Claimed feasibility, true feasibility, selector
enumeration, directory validation, and every analysis denominator cover
`(0,0,80)` for a size-two card and `(0,0,100)` for a size-three card.

### 6.6 APPLY overwrite and instance binding

At emission, every parsed `APPLY` envelope is bound by trusted runtime
metadata to the exact `(task_id, instance_id)` in the sender's delivered
public ledger. The instance is not a model-supplied TFP1 field. At delivery,
the binding is checked before any live application or acceptance state can
change. If there is no active instance for that task ID, or the active
instance differs from the emission binding, the record is rejected with the
stable code `state.stale_task_instance`. It is never retargeted to a reopened
card.

Within one active card instance, the live application map contains at most one
claim per physical sender and therefore at most six claims. Valid delivered
applications are applied in canonical
`(delivery_tick, sender_id, emission_sequence)` order. A later valid `APPLY`
from the same sender atomically replaces that sender's prior live capability,
cost, and arrival record. After every valid application update, the native
`TeamDirectory` transition clears the card's complete `ACCEPT` set; in
particular, the sender/card's prior acceptance cannot survive a changed claim.
The next selector projection uses only the resulting latest live claim map.

Every raw attempt, parse result, instance binding, transition result, and
superseded live value remains in append-only history. Invalid or stale records
never alter the live map. Closing and reannouncing a task starts an empty map
under the new instance; no old claim, acceptance, or arrival order can roll
forward.

### 6.7 Completion evidence

A formation is a valid lock of the selector's exact roster before the task
deadline. A locked task is an embodied completion only when:

- the corresponding hard synchronous mining event succeeds at the exact
  level and coordinate while the lease is active;
- at least the required number of locked members participate in the
  successful synchronized action;
- no nonmember participation is counted as recruited-team execution; and
- the coordinate-specific state change and relevant coordination counter
  delta agree.

The Section 6.4 closure pass must close the lease with a digest of this
evidence. It may not choose an action or award a completion based solely on an
agent claim. A model-emitted `COMPLETE` is only a public report and never
supplies the system evidence or closes the card by itself.

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
protocol coverage. It must contain both a size-two opportunity with demand
`(0,0,80)` and a size-three opportunity with demand `(0,0,100)`, plus a third
simultaneously legal opportunity that can exercise the two-card limit. The
fixture must be identical across the two E3a arms and must be identified as a
routing fixture rather than behavioral evidence.

Across the two seeds, scripted treatment records must exercise:

1. canonical locally valid `ANNOUNCE` records for both required sizes;
2. initial and replacement `APPLY` records from the same sender and card,
   including replacement after that sender has accepted;
3. card-dependent claimed-feasibility checks for both demands;
4. exact unleased physical-ID projections that remain unchanged under
   analysis-only changes to agent life, actionability, and level;
5. six identical selector-replica hashes and exact rosters for both sizes;
6. reciprocal `ACCEPT` and sponsor `LOCK` for both sizes after any replacement
   invalidates prior acceptance;
7. a third otherwise valid announcement rejected as
   `capacity.active_task_limit`, with neither admitted card evicted;
8. a submitted `DECLINE` rejected as `method.record_not_allowed`, with an
   empty delivered-decline collection;
9. a team-private ordinary message;
10. rejection of one nonmember ordinary message;
11. fail-closed classification and rejection of one oversized raw `TFP1`
   control attempt without truncation or ordinary-text fallthrough;
12. coordinate-specific locked-roster evidence producing deterministic
    `COMPLETE` for one admitted task;
13. disappearance without completion evidence producing system
    `CANCEL|REASON=coordinate_vanished` for the other admitted task (the
    inactive-coordinate closure case);
14. release of both the affected lease and card slot before later admission;
15. an `APPLY` bound to the old instance rejected at delivery as
    `state.stale_task_instance`, with no mutation of a current instance;
16. rejection of a stale pre-closure reopen, followed by acceptance of a
    fresh-provenance announcement after the coordinate reappears;
17. successful admission of a previously capacity-blocked opportunity after
    a slot is released; and
18. byte-identical export and deterministic replay of every raw application,
    overwrite, stale rejection, and other transition.

E3a passes only if:

- all four episodes produce complete canonical artifacts;
- Source golden prompt and route checks pass;
- Source delivery retains exact one-tick all-peer semantics;
- every accepted agent-authored treatment control is one-tick delayed, has
  exactly five routed peer copies, and has exactly six logical public viewers
  after including the sender-local ledger;
- every exact raw `TFP1` prefix remains a control attempt on failure, while
  the Source path bypasses the treatment classifier;
- unauthorized ordinary deliveries are exactly zero;
- roster agreement and lease exclusivity are 100%;
- all accepted announcements have legal local provenance;
- size-two cards always use `(0,0,80)` and size-three cards always use
  `(0,0,100)` in cards, claims, selectors, truth checks, and replay;
- every selector eligible-ID vector is exactly the sorted publicly unleased
  physical IDs and is invariant to alive, actionable, level, position,
  visibility, inventory, and application-status changes;
- each sender/card instance has at most one live application and each card
  has at most six;
- a valid repeated `APPLY` replaces only that sender's live claim in canonical
  delivery order, clears the card's prior `ACCEPT` set, and preserves all raw
  and superseded history; the selector uses the replacement and never the
  superseded value;
- an emission-bound `APPLY` never crosses an instance boundary and every such
  attempt is rejected as `state.stale_task_instance`;
- nonterminal card occupancy never exceeds two;
- the capacity rejection, no-eviction rule, blocked-opportunity record, and
  canonical first-accepted card all reproduce;
- `DECLINE` deliveries and selector decline inputs are exactly zero;
- completion and vanished-coordinate cancellation precede admissions and
  release leases and slots exactly once;
- a reopened task has a new instance, fresh provenance, and no carried claim,
  acceptance, plan, lease, or deadline;
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

### Opportunity-lifecycle denominators

A physical-opportunity lifecycle begins when one exact
`mine:<level>:<x>:<y>` coordinate becomes an active hard synchronous mining
opportunity with one required size, and ends when that coordinate ceases to be
that opportunity. Reappearance after an inactive interval or a size change
starts a new lifecycle. Public-card closure or reannouncement alone does not.

For each lifecycle, let `q` be the first tick for which the opportunity was
legally visible in the collective treatment projection at both `q-1` and `q`.
At least one treatment agent must see it at each tick, but the observer at
`q-1` and the observer at `q` may be different agents; no individual agent is
required to see it twice. A lifecycle without such two-consecutive-tick
collective visibility has no `q` and enters none of the feasibility
denominators.

At `q`, the secondary **physical-feasibility** test ignores both current team
leases and public-card capacity. It passes when at least one exact-size roster
consists entirely of alive, actionable agents on the opportunity's level and
covers the archived true demand for that size: `(0,0,80)` for two agents or
`(0,0,100)` for three.

Alive, actionable, and same-level status in this paragraph is analysis-only.
It is computed from archived state after behavioral execution and never enters
the live selector eligible-ID vector or any selector hash.

Every physically feasible lifecycle is classified exactly once at `q` in this
ordered, disjoint partition:

1. **lease-blocked:** no exact-size true-feasible roster can be formed entirely
   from currently unleased alive, actionable agents on that level;
2. **capacity-blocked:** such an unleased true-feasible roster exists, but the
   lifecycle neither owns a nonterminal public card nor has an unused one of
   the two public-card slots; or
3. **primary eligible:** an unleased exact-size true-feasible roster exists
   and the lifecycle either owns a nonterminal card or at least one card slot
   is available.

Thus lease blocking takes precedence when both resource constraints would
apply, making the lease- and capacity-blocked counts disjoint. Physically
infeasible lifecycles are reported separately. The archived `q` snapshot,
candidate rosters, card-dependent truth calculation, lease map, slot
occupancy, category, and hashes are immutable. Later feasibility, lease
release, card admission, closure, or reannouncement never adds or reclassifies
that lifecycle.

### H3b0.1: formation transfer

At least 80% of primary-eligible opportunity lifecycles form a true-feasible
locked roster before the applicable public-card deadline.

A lifecycle contributes at most one primary-denominator observation. Its
formation numerator is one if any canonical card instance for that lifecycle
locks an exact-size roster that covers that card's size-dependent true demand;
otherwise it is zero. Unannounced, capacity-rejected after `q`, expired, and
cancelled primary-eligible lifecycles remain zeroes. Repeated announcements or
locks cannot increase the numerator.

If fewer than three primary-eligible lifecycles occur across the matrix, or
fewer than two seeds contain at least one, the formation result is
**uninformative**, not a method failure or pass. Physical-feasibility,
lease-blocked, and capacity-blocked rates remain secondary mechanism
diagnostics and cannot replace this exposure gate.

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
- exact size-to-demand mapping in every card, claim check, selector input,
  truth calculation, and denominator;
- every selector eligible-ID vector equals the sorted publicly unleased
  physical IDs and contains no alive, actionable, level, position, inventory,
  visibility, application-presence, or other environment-derived filter;
- no card instance has more than one live `APPLY` per sender or more than six
  live claims total;
- every valid repeated application atomically overwrites the sender's live
  claim in canonical delivery order, clears the card's `ACCEPT` set, and
  preserves complete history;
- every delivered `APPLY` matches its emission-bound instance or is rejected
  without mutation as `state.stale_task_instance`;
- every accepted agent-authored public control has five routed peer copies and
  six logical public viewers, with sender-local visibility excluded from
  network delivery bytes;
- every raw treatment communication beginning exactly with `TFP1` is
  fail-closed as a control attempt and never becomes ordinary text;
- no more than two nonterminal public cards and zero card eviction;
- canonical first-accepted admission ordering and exact
  `capacity.active_task_limit` rejection;
- zero accepted or delivered `DECLINE` records and an empty selector decline
  input;
- system completion, vanished-coordinate cancellation, lease release, and
  slot release before every tick's admissions;
- fresh provenance and a new empty instance for every reannouncement;
- lifecycle-once denominator classification with disjoint lease- and
  capacity-blocked counts;
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

- unique legal opportunities and physical-opportunity lifecycles seen;
- lifecycles with two-tick collective visibility;
- secondary physically feasible, physically infeasible, lease-blocked,
  capacity-blocked, and primary-eligible lifecycle counts;
- the disjoint category and frozen `q` for every lifecycle;
- public-card occupancy by tick, slot-release events, and maximum occupancy;
- `capacity.active_task_limit` rejections and unique blocked opportunities;
- fresh and stale reannouncement attempts and accepted new instances;
- announcement rate and first-seen-to-announcement latency;
- valid and rejected records by stable code;
- raw, accepted, live, overwritten, superseded, and stale applications per
  card instance;
- maximum live claims per card, acceptance clears caused by overwrites, and
  size-dependent claimed coverage;
- exact selector unleased-ID vectors and their public lease-state hashes;
- selector plans and replica hashes;
- lock count, card-dependent true-feasible lock count, and lock latency;
- roster churn, system and model cancellations, completions, and expiries;
- primary formation rate overall and separately for size two and size three;
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

- emitted agent-authored control records and payload bytes;
- routed peer control copies and fan-out-weighted network bytes;
- sender-local accepted-control ledger views;
- six-agent logical public-control views and logical fan-out bytes;
- system lifecycle publications, zero agent-routed copies, six-agent logical
  views, and logical publication bytes;
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
- size two or size three can map to any demand other than `(0,0,80)` or
  `(0,0,100)`, respectively;
- the directory can admit more than two nonterminal cards, evict a card, or
  reuse a terminal instance's claims or lease;
- tick-start completion, inactive-coordinate cancellation, or release can
  occur after announcement admission;
- `DECLINE` can enter a treatment prompt, accepted transition, or selector
  input;
- a live selector eligible-ID vector is anything other than the sorted
  publicly unleased physical IDs, or reads alive, actionable, level, position,
  inventory, visibility, application-status, or other environment truth;
- repeated `APPLY` records can create multiple live claims for one
  sender/card, exceed six live claims, avoid canonical overwrite and
  acceptance clearing, or discard raw/superseded history;
- an `APPLY` lacks an emission-time instance binding or can mutate a different
  active instance instead of failing as `state.stale_task_instance`;
- an agent-authored accepted control can produce any count other than five
  routed peer copies and six logical public viewers;
- an exact treatment `TFP1` prefix can fall through to ordinary text, or the
  treatment classifier can affect Source;
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
- nonterminal occupancy exceeds two, an admitted card is evicted, or a
  capacity rejection uses a code other than `capacity.active_task_limit`;
- a reannouncement lacks post-closure provenance or inherits prior-instance
  state;
- a closure lacks coordinate-specific evidence/activity state, occurs out of
  order, or fails to release its lease and slot;
- any `DECLINE` is accepted or appears in a selector input;
- a selector eligible-ID vector differs from the sorted public unleased set or
  contains an environment-derived filter;
- live application cardinality, overwrite order, `ACCEPT` clearing, or raw
  history disagrees with Section 6.6;
- a stale-instance application is accepted, retargeted, or rejected under any
  code other than `state.stale_task_instance`;
- agent-authored public-control routed-copy or logical-visibility accounting
  differs from five and six, respectively;
- a failed exact-prefix `TFP1` attempt is delivered as ordinary text;
- a lifecycle denominator record is repeated, mutable, or cannot reproduce
  the disjoint lease/capacity classification;
- directory replicas or replay hashes disagree;
- an accepted control exceeds 256 bytes;
- a Source prompt or route invariant fails;
- the seed-14100 canary has no legal opportunity exposure; or
- the canary fails any integrity gate.

After all three seeds, classify using only the primary lifecycle denominator:

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
- physical-opportunity lifecycle records, frozen `q` snapshots, candidate
  true rosters, and disjoint feasibility classifications;
- public card instances, per-tick slot occupancy, admission order, and
  blocked-opportunity records;
- submitted raw TFP1 records, including rejected capacity,
  `state.stale_task_instance`, and disallowed-`DECLINE` records;
- emission-time card-instance bindings, per-delivery live-application maps,
  superseded values, overwrite order, and `ACCEPT`-clear events;
- parse and transition codes;
- exact selector public inputs, sorted publicly unleased physical-ID vectors,
  empty decline tuples, six replica outputs, and hashes;
- lease, membership, card-phase, completion, cancellation, expiry, and slot
  release transitions;
- receiver-indexed control and ordinary route envelopes, sender-local public
  ledger entries, and separate routed-copy and logical-visibility counts;
- rejected-message records;
- coordinate-specific execution and vanished-coordinate evidence with system
  closure digests;
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
9. `TeamDirectory` has no two-card admission limit, archived same-ID card
   generations, post-closure provenance rule, or deterministic no-eviction
   capacity rejection.
10. No E3 adapter freezes the selector eligible-ID vector to public
    `agent_to_task` state while excluding alive, actionable, level, and other
    environment truth.
11. TFP1 envelopes have no emission-time card-instance binding or stable
    `state.stale_task_instance` delivery rejection across reannouncement.
12. No coordinate-specific tick-start bridge completes a locked task,
    system-cancels a vanished task, and releases its lease and card slot before
    new admissions.
13. No lifecycle-once opportunity ledger freezes collective visibility,
    card-dependent physical truth, leases, capacity, and the disjoint primary,
    lease-blocked, and capacity-blocked classification.
14. No E3 profile, launcher, durable campaign budget, summarizer, canonical
    output schema, or E3-specific test exists.

The minimum implementation delta is therefore:

- add one opt-in E3 topology without altering the baseline branch;
- add a treatment-only agent context and communication instruction path;
- preserve raw complete communication for strict TFP1 validation;
- add the treatment-only exact-prefix fail-closed control classifier;
- add receiver-indexed treatment inboxes, sender-local public-ledger entries,
  and separate five-copy route versus six-view visibility journals;
- integrate `TeamDirectory` and six replicated public selectors using only
  sorted public unleased physical IDs;
- preserve native one-live-claim overwrite and `ACCEPT` clearing while adding
  full application history and emission-bound instance rejection;
- expose every legally visible hard-mining candidate from the existing masks;
- add the default-off requirement cap;
- enforce the size-two/80 and size-three/100 demand mapping everywhere;
- add two-card admission, canonical first acceptance, no eviction, archived
  card generations, and fresh-provenance reannouncement;
- remove `DECLINE` from the treatment surface and selector input;
- add the ordered coordinate-specific completion/vanish closure pass and
  release leases and slots before admissions;
- add immutable lifecycle-once denominator and blocker classification records;
- add separate control and ordinary communication accounting;
- add deterministic export/replay and complete artifact binding;
- add E3a scripted actors and Source golden regressions; and
- add the staged, bounded E3b0 launcher and read-only analysis.

Until those changes are implemented, committed, independently reviewed, and
E3a passes, E3b0 remains prospective and no provider call may be made.
