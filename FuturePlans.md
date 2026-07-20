# Future plans: Alem toward DICE

This file is a roadmap only. None of the capabilities below are part of the
baseline reproduction, and baseline profiles must remain frozen as extensions
are introduced.

The `team_leader_200` experiment is an explicit centralized comparison, not a
roadmap change: its fourth logical participant aggregates only legal worker
views and issues assignments while the world still contains exactly three
players. Results from that treatment must remain labeled separately from the
decentralized DICE direction below.

## Design constraints

- Preserve a runnable three-agent Alem control so changes can be measured
  against the pinned baseline.
- Keep control decentralized: an evaluator may advance the synchronous world
  and route permitted messages, but it must not assign roles, choose actions,
  aggregate private state, or act as a privileged team planner.
- Give each agent only its local observation, local memory, local failure state,
  and messages actually delivered through the configured network.
- Version every environment, observation, action, communication, role, and
  failure schema in artifacts.

## Scale ladder

- [ ] Establish 3-agent correctness and cost/latency baselines with the fake and
  reduced OpenAI profiles.
- [ ] Add an 8-agent all-LLM profile and scale map area, resource density, task
  density, and evaluation goals so added agents create meaningful work rather
  than spawn congestion.
- [ ] Gate 16 agents on stable environment stepping, bounded prompt growth,
  sparse-message delivery, and reproducible metrics.
- [ ] Gate 32 all-LLM agents on bounded request concurrency, backpressure,
  per-agent timeout isolation, and acceptable end-to-end tick latency.
- [ ] Measure success and coordination quality against token use, delivered
  message bytes, wall-clock latency, and concurrency—not agent count alone.

The current world defaults to a 48×48 map, the published LLM setup is three
agents, and upstream documentation describes 1–4 agents as the supported range.
Targeted Give actions and teammate text grow with population, and unconstrained
broadcast traffic is quadratic. Passing the 32-agent milestone therefore
requires workload and interface redesign, not just a larger `player_count`.

## Decentralized sparse communication

- [ ] Replace global team broadcast with a proximity graph computed from the
  world state at each tick.
- [ ] Configure observation radius, communication radius, maximum peer degree,
  per-agent byte/message budgets, delay, loss, and network partitions.
- [ ] Route messages peer-to-peer without a global mailbox or shared blackboard;
  record sender, intended peer, delivery tick, loss reason, and payload size.
- [ ] Prevent implicit global leakage through teammate dashboards, achievement
  summaries, task lists, debug metadata, or prompts.
- [ ] Add decentralized discovery and relay experiments in which multi-hop
  information transfer must be produced by agent actions.

## Failed agents

- [ ] Define process failure separately from ordinary in-world injury or death.
- [ ] Add seeded, permanent crash-stop schedules. After its failure tick, an
  actor receives no model call, emits no action beyond environment-safe `Noop`,
  and sends or receives no messages for the remainder of the episode.
- [ ] Isolate a failed or timed-out actor so healthy actors and environment ticks
  continue; transient provider errors remain transport failures, not simulated
  crash-stop events.
- [ ] Record time to failure detection, abandoned obligations, reassignment
  latency, recovery of team performance, and mission completion under increasing
  failed-agent fractions.
- [ ] Leave Byzantine behavior, adversarial messages, equivocation, spoofing, and
  consensus protocols out of scope until crash-stop experiments are stable.

## Role coherence

- [ ] First measure adherence to Alem's fixed warrior/forager/miner roles:
  specialist action share, useful contribution, duplication, idle time, and role
  drift over long horizons.
- [ ] Introduce decentralized role claims and peer acknowledgements; do not add a
  central role allocator.
- [ ] Add explicit handoff, relinquish, and takeover messages so responsibilities
  can migrate when workload or connectivity changes.
- [ ] Add failure-driven reassignment experiments and measure acknowledgment
  agreement, coverage gaps, duplicated responsibility, handoff success, and time
  to restore coverage.
- [ ] Evaluate coherence over at least 1,000 steps and distinguish a stable but
  ineffective role allocation from one that improves mission outcomes.

## Prompt-cache experiment

Follow the [OpenAI prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
when implementing this experiment.

- [ ] Preserve the exact current-main prompt order as the control condition.
- [ ] Build an A/B variant with identical shared simulation rules, action schema,
  role catalog, and experiment specification first, followed by dynamic agent
  identity, observation, inbox, memory, and tick state.
- [ ] Use a stable cache-routing key derived from model, prompt-schema revision,
  and experiment revision; exclude run, process, episode, and agent identifiers.
- [ ] Compare actions, achievements, parse rate, cached/write tokens, latency, and
  cost before adopting the reordered layout. Prompt caching must never silently
  change baseline semantics.

## Explicit 500-agent DICE gap

Thirty-two all-LLM participants is an intermediate systems milestone, not the
DARPA DICE large-scale objective. A credible 500-agent experiment still needs:

- [ ] spatial partitioning and scalable procedural worlds rather than a single
  fixed-size crowded map;
- [ ] observation construction and action masking whose cost is local rather
  than proportional to the full population;
- [ ] sparse routing and local discovery in place of team-wide broadcasts and
  full teammate dashboards;
- [ ] task density, interaction mechanics, and mission metrics that require
  multiple semi-independent coalitions;
- [ ] asynchronous or staged execution that avoids 500 frontier-model calls on
  every environment tick while preserving agent autonomy;
- [ ] load, cost, replay, and failure-injection tests at 50, 100, 250, and 500
  simulated participants.

The architecture should not claim DICE-scale collaboration until a 500-agent
run demonstrates bounded per-agent context/communication, reproducible failure
behavior, and measurable role coherence without centralized control.
