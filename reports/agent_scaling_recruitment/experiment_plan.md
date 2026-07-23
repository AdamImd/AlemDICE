# Agent Scaling and Decentralized Recruitment Experiment Plan

Status: approved for staged implementation
Protocol date: 2026-07-22
Implementation branch: `research/agent-scaling-recruitment-pipeline`

## 1. Goal and claim boundary

This program tests three increasingly difficult claims without changing several
mechanisms at once:

1. how the unchanged Alem Source language-agent baseline behaves as the number
   of physical agents changes;
2. whether six decentralized agents can form useful task teams through bounded
   recruitment messages; and
3. whether a promoted recruitment mechanism transfers into Alem when ordinary
   communication is restricted to current team members.

All behavioral studies use homogeneous `gpt-5.4-nano` action agents at high
reasoning. The Source condition retains `robust_all`, the
`specific_collaborative` prompt, scratchpad memory, the existing action parser,
and one model call per physical agent per tick. No bodyless planner or additional
planning call is part of this program.

The completed sequence may support claims about population scaling,
decentralized team formation, and team-scoped communication. It does not by
itself establish long-horizon role coherence, strict information isolation,
robustness to failed or hostile agents, or scaling beyond six agents.

## 2. E0: free compatibility gate

Run scripted, provider-free episodes at `N = {1, 2, 3, 4, 6}` using seeds
`13000` and `13001` for 10 ticks.

The gate verifies:

- reset and step execution for every population;
- observation, action, reward, trajectory, and saved-state dimensions;
- one indexed configuration slot per physical agent;
- warrior/forager/miner specialization cycling;
- every targeted `Give` mapping;
- zero model/provider calls;
- exact baseline broadcast accounting;
- complete canonical episode artifacts;
- deterministic initial-state and saved-state replay hashes; and
- no agent-index, serialization, or rendering failures.

E0 is an engineering result, not evidence that adding agents improves behavior.
Any failed population blocks paid scaling experiments at that population.

## 3. E1: Source performance versus agent count

Freeze the current strongest Source configuration:

- model: `gpt-5.4-nano`;
- reasoning effort: `high`;
- topology: `baseline`;
- coordination strategy: `free`;
- agent harness: `robust_all`;
- prompt: `specific_collaborative`;
- ordinary reasoning, scratchpad, and peer broadcast enabled;
- one episode worker, with agents called concurrently within a tick;
- debriefs and bulk image generation disabled.

Run the following funnel:

| Stage | Populations | Seeds | Horizon | Purpose |
| --- | --- | --- | ---: | --- |
| E1a | 1, 2, 3, 4 | 13100–13102 | 200 | Supported-range screen |
| E1b | 6 | 13100–13102 | 200 | Gated extension |
| E1c | Every passing count | 13200–13209 | 1,000 | Confirmed descriptive curve |
| E1d | 3 and 6 | 13300–13319 | 10,000 | Paper-scale comparison |

E1d reports Easy, Medium, and Hard separately and requires an explicit cost
authorization after a dry-run call/token estimate.

The headline output is a three-panel performance-versus-agent-count figure:

1. paper Total%;
2. mean per-agent episode return; and
3. raw unique team achievements.

Supporting panels report Base%, Coordination%, deaths, action parsing, total and
per-agent-step tokens, delivered bytes, and tick latency. Every plot includes raw
seed points and 95% bootstrap intervals. `N=1` is shown as a separate solo
reference because it uses a different wrapper; `N=6` is marked as beyond the
documented 1–4 range.

The graph is labeled a fixed-world broadcast population curve. Increasing
population also changes spawn geometry, mob pressure, role balance, prompt size,
and some coordination requirements, so it is not interpreted as a pure causal
effect of agent count.

## 4. E2: RecruitmentArena-6

Create a deterministic, pure-Python coalition-formation environment before
altering Alem routing.

Each episode contains:

- six agents with two warrior-, two forager-, and two miner-oriented capability
  vectors;
- private true capabilities and task-specific costs;
- public task cards with stable ID, three-dimensional demand, team size two or
  three, reward, and deadline;
- one active team per agent;
- task-bound membership ending on completion, cancellation, or expiry;
- 12 recruitment rounds and one-round control-message delay;
- one recruitment record per agent per round, capped at 256 bytes;
- no ordinary conversation before a team locks; and
- Source-style ordinary communication delivered only to locked teammates.

Four paired scenario families isolate different allocation problems:

1. one feasible complementary task;
2. two disjoint tasks;
3. competing tasks requiring a scarce capability; and
4. oversubscription with more candidates than roster slots.

### 4.1 Recruitment methods

Compare:

- **Open volunteer quorum:** candidates publish applications, all peers derive
  the same first-valid roster, and selected members accept.
- **Mutual nomination:** a roster activates only when every listed member
  publishes the same reciprocal nomination.
- **Peer Contract Net:** a task-specific embodied sponsor publishes a call,
  candidates bid, the sponsor awards a roster, and every member accepts.

Controls are one preformed all-six team, balanced fixed `3+3` teams, no team and
no ordinary communication, deterministic random allocation, and an offline
oracle that never enters an agent prompt.

Run:

| Stage | Seeds | Purpose |
| --- | --- | --- |
| E2a | 20000–20999 | Scripted correctness, no model calls |
| E2b | 22000–22002 | LLM mechanism screen |
| E2c | 22100–22109 | Paired efficacy estimate for passing methods |
| E2d | 22200–22209 | Mathematical selection-rule ablation |
| E2e | 22300–22309 | Five-round team execution bridge |

Formation-only tasks complete deterministically when a locked roster's true
capabilities satisfy the task. E2e additionally requires team-private
communication and complementary member contributions.

### 4.2 Mathematical member-selection experiment

Hold Peer Contract Net fixed and compare:

1. `first_valid`: first claimed-feasible roster by arrival round and agent ID;
2. `random_valid`: deterministic seeded sampling from claimed-feasible rosters;
3. `exact_utility`: exact subset enumeration.

For claimed capability vector \(q_i\), cost \(c_{it}\), and task demand \(d_t\),
the exact selector maximizes

\[
U(S,t)=
\min_{j:d_{tj}>0}
\frac{\sum_{i\in S}q_{ij}}{d_{tj}}
-
0.25\frac{\sum_{i\in S}c_{it}}{100|S|}.
\]

The roster must have the requested size and claimed capability coverage of at
least one in every demanded dimension. Ties resolve by lexicographically sorted
agent IDs. The analysis-only oracle enumerates `(tasks + idle)^6` using true
capabilities and costs.

Primary outputs are normalized task reward and oracle regret. Formation
latency, feasibility, churn, truthful-report calibration, messages, delivered
bytes, tokens, and wall time are secondary outcomes.

### 4.3 E2a pre-run implementation amendment — 2026-07-23

Provider-free diagnostics identified three ambiguities before any canonical E2
artifact was created. The canonical E2a run therefore freezes the following
clarifications:

- all public task cards are active concurrently rather than being processed in
  task-ID order;
- locked rosters remain committed until a common allocation close after 12
  acting rounds and a one-round delivery drain, so the live allocation obeys
  the same one-task-per-agent constraint as the static
  `(tasks + idle)^6` oracle;
- the weak miner capability is `(20,20,35)`, because the earlier draft value
  `65` plus a nonspecialist's `20` would satisfy an `80` mining demand and
  eliminate the intended scarce-resource condition;
- every decentralized protocol is crossed with two bounded task-choice
  policies:
  - `local_commit`, where an agent selects one task from public cards using
    only its own capability/cost and a stable tie-break (Mutual Nomination uses
    only the public role proxy because it publishes no capability bid); and
  - `public_sweep`, where agents submit bounded records to every task in stable
    order and resolve overlapping candidate rosters from the delivered public
    ledger;
- the 80% method gate uses **oracle-allocation coverage**, the fraction of
  simultaneously allocable oracle tasks completed. Separately reported
  feasible-card formation uses every individually feasible card as its
  denominator, including both mutually exclusive scarce cards;
- achieved true utility, equal-reward utility regret, achieved/oracle raw cost,
  oracle assignment, candidate-roster overlap, multi-offer rounds, and roster
  revisions are retained in addition to reward regret; and
- all-six and fixed-`3+3` topology controls report raw reward and communication
  exposure, but exact-oracle normalized reward/regret are `N/A`. The fixed
  groups receive tasks by public task order, never by hidden true feasibility,
  and are not reused during the simultaneous allocation window.

The canonical E2a artifact comprises seeds `20000–20999`, four scenario
families, six protocol/policy arms, and four controls. It runs with process
shards, writes deterministic gzip JSONL traces and replay hashes, and makes
zero model or provider calls. A method/policy arm may fail its efficacy gate
without invalidating the campaign; malformed records, unauthorized delivery,
replay disagreement, provider use, or other integrity failures invalidate the
run.

## 5. E3: deployment into Alem

Add the promoted recruitment mechanism as an opt-in topology. Baseline topology
must remain behaviorally unchanged.

Communication semantics:

- unteamed agents may emit only a valid typed recruitment record on a bounded
  public bulletin;
- invalid or free-form unteamed messages are rejected and logged;
- after a roster locks, ordinary `<communication>` messages retain Source
  semantics but route only to current teammates;
- nonmembers never receive team messages;
- control-plane and ordinary delivery costs are recorded separately;
- membership transitions are deterministic and replayable; and
- middleware never selects an environment action or reads hidden state to make
  a recruitment decision.

An embodied agent that locally observes an Alem opportunity announces it.
Stable task IDs derive only from observable task type, level, and location.

The first Alem recruitment study uses the 48×48 world and task requirements of
two or three agents in every arm. This removes the native six-agent-task
confound. Ordinary global teammate observations remain visible initially, so
the treatment is called team-scoped communication rather than strict
information isolation.

Run:

| Stage | Environment | Seeds | Horizon |
| --- | --- | --- | ---: |
| E3a | Alem Debug, scripted god mode | 14000–14001 | 20 |
| E3b | Alem Debug, Easy | 14100–14102 | 200 |
| E3c | Full Alem, Easy | 14200–14209 | 1,000 |
| E3d | Full Alem, all difficulties | 14300–14319 | 10,000 |

E3d requires explicit cost authorization.

Compare unchanged six-agent Source broadcast, balanced fixed `3+3` communication
teams, dynamic recruitment, and a bulletin-only/no-ordinary-communication floor.
The final paper-scale comparison retains Source, fixed teams, and dynamic
recruitment.

## 6. Promotion, analysis, and audit rules

Every implemented mechanism must satisfy:

- zero unauthorized cross-team deliveries;
- 100% roster agreement for activated teams;
- at least 95% action and recruitment-transition validity;
- zero unrecovered provider failures in canonical episodes;
- at least 80% of feasible recruitment opportunities formed before deadline;
- deterministic replay hash agreement; and
- no baseline prompt or route change under baseline topology.

Bounded transport retries are allowed by the frozen provider configuration and
are not themselves unrecovered failures. A canonical episode may retain one
only when every logical decision has a final completed response, provider
attempts equal completed logical responses plus typed transport errors globally
and per worker, usage and attempt-ledger evidence agree exactly, no response is
incomplete, and the declared campaign call ceilings remain unexceeded.

Ten-seed behavioral promotion requires positive paired direction on at least
seven seeds for at least two of Total%, return, and unique achievements, with no
material regression in the third. A final communication-efficiency claim
requires a paired 95% interval no worse than a 5% relative performance margin
on all three outcomes while reducing delivered bytes by at least 30%.

If no recruitment method passes its mechanism gate, stop before Alem, preserve
the negative result, version the revised protocol, and rerun E2b. Live outputs
are never silently repaired.

Each study records resolved configuration, Git and dependency hashes, model
snapshot, call ceiling, seeds, protocol version, raw model responses, roster
events, route envelopes, rejected deliveries, usage, latency, replay hashes,
terminal summaries, and the exact inputs used to generate every table and
figure.

## 7. Deferred work

The following remain outside this sequence:

- strict filtering of nonmember observations, requests, and `Give`;
- message loss, crash-stop, Byzantine, or adversarial agents;
- performance-history reputation learning;
- teams larger than three;
- populations above six;
- long-horizon context management and role-drift experiments; and
- learned communication or routing policies.

## 8. Campaign execution amendment — 2026-07-23

The user subsequently authorized autonomous execution through the complete
staged pipeline without additional approval pauses. This removes the approval
holds originally stated for E1d and E3d, but does not remove scientific or
engineering promotion gates. Before each paid stage, the implementation must
still archive its resolved configuration, estimated and hard call ceilings,
current cumulative usage, and upstream gate result. A failed mechanism,
routing, artifact, or reproducibility gate still blocks the associated efficacy
claim and redirects the next experiment according to the observed failure.

E0.1's strict 200-tick result remains a failure. Its valid natural
all-agent-death terminations are waived only as a blocker to E1 because all
structural checks passed; early death and actual agent-tick exposure remain E1
outcomes rather than being discarded.
