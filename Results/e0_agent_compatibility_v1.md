# E0 Variable-Agent Compatibility Gate

Result: **PASS**

E0 is a provider-free engineering gate. It establishes that the canonical Alem
evaluation path can represent and execute the planned population sizes; it does
not establish that adding agents improves behavior.

## Method

The run used `alem/default` with baseline communication topology at
`N = {1, 2, 3, 4, 6}`. Seeds 13000 and 13001 each ran for 10 environment ticks.
One validated client configuration slot was created per physical agent, but a
deterministic scripted agent replaced inference. Every agent selected `Noop` and
emitted one 30-byte message per tick:

```text
E0|AGENT=<id>|TICK=<tick>|STATE=ACTIVE
```

The standard evaluator still performed environment reset and stepping, action
canonicalization, one-tick communication routing, episode accounting, trajectory
serialization, state serialization, and artifact generation. The harness then
reconstructed each seed and compared stable initial-state projection hashes.

The assertions covered:

- client/agent cardinality and every trajectory agent axis;
- warrior/forager/miner specialization cycling;
- all ordered `Give to Agent <id>` mappings;
- complete canonical artifacts and ten saved states per episode;
- action parse rate 1.0 and zero model, provider, or transport calls;
- exact broadcast counts, with emitted messages \(NT\) and delivered messages
  \(N(N-1)T\); and
- identical saved-state and independently reconstructed state hashes.

A post-run renderer smoke test encoded one full-world H.264 frame from seed
13000 for every population and checked it with `ffprobe`.

## Results

| Agents | Episodes | Ticks | Observation | Actions | Give mappings | Emitted | Delivered | Bytes | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 2 | 20 | 9,476 | 55 | 0 | 20 | 0 | 0 | PASS |
| 2 | 2 | 20 | 9,597 | 55 | 2 | 40 | 40 | 1,200 | PASS |
| 3 | 2 | 20 | 9,718 | 56 | 6 | 60 | 120 | 3,600 | PASS |
| 4 | 2 | 20 | 9,839 | 57 | 12 | 80 | 240 | 7,200 | PASS |
| 6 | 2 | 20 | 10,081 | 59 | 30 | 120 | 600 | 18,000 | PASS |

Across ten episodes, E0 executed 100 environment ticks and 320 scripted
agent-ticks. It emitted 320 messages, delivered 1,000 copies totaling 30,000
bytes, saved 100 states, and made zero model calls and zero provider requests.
All five 768×832 renderer checks passed.

The observation vector grew by 121 values for each added teammate. The targeted
handoff surface contained \(N(N-1)\) ordered mappings, while baseline broadcast
delivery grew as \(N(N-1)\) per tick. These are interface and communication-cost
facts, not task-performance findings.

## Decision and limitations

E0 removes the variable-agent compatibility blocker for the staged E1 Source
screen, including the gated six-agent extension. E1 still needs behavioral
evidence: real language-model actions, longer horizons, paired seeds, and task
outcomes.

This result does not test model reasoning, useful coordination, dynamic
recruitment, long-horizon role coherence, provider concurrency, or behavioral
scaling. `Noop` deliberately isolates infrastructure from policy quality.

## Reproduction and provenance

Harness commit:
`587e7176ce3b02296120e9504c1d7d609e9dcf2a`

```bash
uv run --extra baselines-llm --python 3.12 \
  python scripts/run_e0_agent_compatibility.py \
  --output outputs/alem_eval/e0_agent_compatibility_v1
```

Raw result:
`outputs/alem_eval/e0_agent_compatibility_v1/e0_results.json`

Raw result SHA-256:
`7b67b56544a00df63b1e07106c9eeffe51eedc9bc62dc7d8d914af734ad8c47d`

The raw run directory contains 82 canonical files plus five renderer smoke
videos. The raw manifest records the pre-existing untracked
`Results/replays/nano_high_source_full_world.mp4`; it is outside the E0 run root
and was not modified or used.
