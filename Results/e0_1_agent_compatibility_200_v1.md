# E0.1 200-Step Variable-Agent Compatibility Extension

Strict result: **FAIL — requested horizon not reached in 3/10 episodes**

Structural result: **PASS**

E0.1 extended the provider-free E0 engineering gate from 10 to 200 requested
environment turns at `N = {1, 2, 3, 4, 6}` and added turn-time
instrumentation. All compatibility checks passed, all ten episode artifact sets
were complete, and no provider was contacted. Seven episodes reached turn 200.
Three `Noop` episodes ended normally when every agent had died from mob combat:

- `N=1`, seed 13000: 114/200 turns, one death;
- `N=4`, seed 13001: 182/200 turns, four deaths; and
- `N=6`, seed 13001: 192/200 turns, six deaths.

These are clean environment terminations, not API, evaluator, parsing, or
artifact failures. The preregistered exact-horizon gate is nevertheless false,
so the strict E0.1 result remains a failure rather than being reinterpreted
after inspection.

## Method

The run used `alem/default`, baseline communication, seeds 13000 and 13001, and
200 requested turns per episode. One deterministic scripted actor represented
each physical agent. Every active actor selected `Noop` and emitted:

```text
E0|AGENT=<id>|TICK=<tick>|STATE=ACTIVE
```

The normal evaluator performed reset and stepping, action canonicalization,
one-tick communication routing, accounting, trajectory and state serialization,
and artifact generation. The runner checked client/agent cardinality, every
trajectory agent axis, the Warrior/Forager/Miner specialization cycle, all
ordered `Give to Agent <id>` mappings, exact broadcast accounting, action parse
rate 1.0, complete artifacts, deterministic replay projections, and zero model,
provider, and transport calls.

Timing was measured with a monotonic wall clock:

- **Mean turn** is the step-weighted evaluator mean over the complete turn path.
- **Max turn** is the slowest complete turn.
- **Mean excluding slowest** removes exactly one slowest turn from each episode,
  then divides the remaining accumulated turn time by the remaining turns. It
  is a descriptive cold-start sensitivity check, not a separately timed
  steady-state benchmark.
- **Worker/turn** divides the cumulative parallel scripted-worker action phase
  by actual turns. It excludes the rest of environment and evaluator work.
- **Episode wall** spans the complete episode evaluation. Population values are
  the arithmetic mean of the two episodes.

## Population results and turn times

| Agents | Actual/requested turns | Structural | Horizon | Mean turn (s) | Mean excluding slowest (s) | Max turn (s) | Worker/turn (ms) | Mean episode wall (s) |
| ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 314/400 | PASS | FAIL | 0.151301 | 0.076475 | 11.947751 | 0.155216 | 23.809163 |
| 2 | 400/400 | PASS | PASS | 0.175483 | 0.124436 | 10.524361 | 0.176712 | 35.201104 |
| 3 | 400/400 | PASS | PASS | 0.235863 | 0.179381 | 11.495492 | 0.165132 | 47.301170 |
| 4 | 382/400 | PASS | FAIL | 0.296600 | 0.236695 | 12.074029 | 0.176943 | 56.794711 |
| 6 | 392/400 | PASS | FAIL | 0.437702 | 0.379037 | 11.999178 | 0.186564 | 85.997763 |

The approximately 10.1–12.1 s maximum in every episode is consistent with a
cold-start or compilation outlier, but the raw summary stores only the maximum,
not its turn index or cause. Removing only each episode's slowest turn makes the
population-dependent evaluator cost easier to see: approximately 0.076 s at
`N=1`, 0.124 s at `N=2`, 0.179 s at `N=3`, 0.237 s at `N=4`, and 0.379 s at
`N=6`. The worker phase remained only 0.155–0.187 ms per turn because these
actors did not perform inference. These timings therefore characterize the
local scripted E0 path, not LLM latency or provider concurrency.

## Per-seed audit

| Agents | Seed | Turns | End reason | Deaths (mob) | Episode wall (s) | Mean turn (s) | Mean excluding slowest (s) | Max turn (s) | Worker total (s) |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13000 | 114 | environment terminated | 1 | 20.763365 | 0.181709 | 0.077585 | 11.947751 | 0.013506 |
| 1 | 13001 | 200 | horizon truncated | 0 | 26.854962 | 0.133969 | 0.075845 | 11.700650 | 0.035231 |
| 2 | 13000 | 200 | horizon truncated | 1 | 34.489646 | 0.171863 | 0.121755 | 10.143545 | 0.036325 |
| 2 | 13001 | 200 | horizon truncated | 0 | 35.912561 | 0.179103 | 0.127117 | 10.524361 | 0.034359 |
| 3 | 13000 | 200 | horizon truncated | 0 | 47.281806 | 0.235647 | 0.179264 | 11.455933 | 0.033071 |
| 3 | 13001 | 200 | horizon truncated | 0 | 47.320533 | 0.236078 | 0.179498 | 11.495492 | 0.032981 |
| 4 | 13000 | 200 | horizon truncated | 3 | 58.405996 | 0.291226 | 0.235991 | 11.283071 | 0.034421 |
| 4 | 13001 | 182 | environment terminated | 4 | 55.183426 | 0.302505 | 0.237469 | 12.074029 | 0.033171 |
| 6 | 13000 | 200 | horizon truncated | 2 | 86.549050 | 0.431751 | 0.373623 | 11.999178 | 0.037591 |
| 6 | 13001 | 192 | environment terminated | 6 | 85.446475 | 0.443901 | 0.384677 | 11.755686 | 0.035543 |

`environment_truncated` means the configured 200-turn horizon was reached.
Deaths in those episodes do not invalidate the horizon when at least one agent
remains active.

## Aggregate results

The matrix executed 1,888 of 2,000 requested environment turns (94.4%) and
6,194 of 6,400 requested scripted agent-turns. Across ten complete artifact
sets it recorded:

- 498.207821 s summed episode wall time;
- 0.263203 s step-weighted mean full-turn time;
- 0.203699 s mean after excluding each episode's single slowest turn;
- 0.326201 s total worker action phase, or 0.172776 ms per actual turn;
- 6,194 emitted messages, 19,544 delivered copies, and 614,408 delivered bytes;
- 1,888 saved states; and
- zero model calls, provider requests, and transport errors.

Broadcast delivery remained exact for the actual number of live turns:

| Agents | Emitted | Delivered | Delivered bytes |
| ---: | ---: | ---: | ---: |
| 1 | 314 | 0 | 0 |
| 2 | 800 | 800 | 25,160 |
| 3 | 1,200 | 2,400 | 75,480 |
| 4 | 1,528 | 4,584 | 144,048 |
| 6 | 2,352 | 11,760 | 369,720 |

## Decision and claim boundary

E0.1 strengthens the evidence that the canonical evaluator, communication
router, action interface, trajectories, and artifacts support variable agent
counts over substantially longer runs. It does **not** show that a passive
`Noop` policy can survive for exactly 200 turns, and it does not test behavioral
performance, LLM reasoning, provider concurrency, recruitment, or role
coherence.

No post-hoc survival aid was enabled. If an exact 200-turn infrastructure stress
test is needed, it should be preregistered as a separate E0.2 condition using
either explicit environment invulnerability or a deterministic survival actor.
That would answer a different question and must not replace this result.

## Reproduction and provenance

Timing/horizon harness commit:
`5c4c64338b9aae9132f9cc32b6eec3d746964529`

```bash
uv run --extra baselines-llm --python 3.12 \
  python scripts/run_e0_agent_compatibility.py \
  --steps 200 \
  --output outputs/alem_eval/e0_1_agent_compatibility_200_v2
```

The measured run began at `2026-07-23T06:57:25.963656Z`, ended at
`2026-07-23T07:08:25.236294Z`, and took 659.272638 s. The raw result is:

```text
outputs/alem_eval/e0_1_agent_compatibility_200_v2/e0_results.json
SHA-256 6d28e885e241b4f58553291d6fa205fb982878f2575f6bdd3a24d711fda9b529
```

The raw directory contains 82 files. A first diagnostic attempt is preserved at
`outputs/alem_eval/e0_1_agent_compatibility_200_v1`; its original strict
validator stopped after completing both `N=1` episodes when seed 13000 ended at
turn 114. The harness was then changed to preserve strict horizon failure while
continuing the remaining populations and reporting structural validity.

The source manifest recorded the pre-existing untracked replay
`Results/replays/nano_high_source_full_world.mp4`. It is outside the E0.1 run
root and was not modified or used.
