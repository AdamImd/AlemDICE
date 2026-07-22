# Experiment audit log

This append-only narrative accompanies machine-readable manifests and raw Alem outputs. The pilot was preregistered before inspecting treatment results.

## 2026-07-22 — implementation checkpoint

- Branch: `research/scalable-decentralized-llm-coordination`
- Base commit: `903120b843caebc0e4fb43e6dd7ec5e067e37388`
- Added strict `DCP1` parsing, agent-local bounded ledgers, evaluator-only protocol metrics, and five Hydra arms.
- Focused tests: 3 passed.
- Hydra composition check: integrated strategy, one episode, seed 9999, 30 steps, W&B disabled.
- No treatment outcomes had been inspected when `registry.yaml` was written.

## Live runs

### v1 early diagnostic (outcomes still running)

- The first free-arm tick produced one valid action and two `length` stops. Both exhausted 768 tokens inside internal reasoning before finishing visible tags; the successful agent's model latency was 61.6 seconds and the truncated calls were about 75 seconds.
- This falsifies the assumption that a 768-token internal-reasoning budget is sufficient for reliable action emission with this served model.
- `registry_v2.yaml` was written before v1 terminal outcomes. It changes only the response mode to concise tags-only output and lowers the now-unneeded cap to 512.

### v2 completed

- Output: `outputs/alem_eval/coordination_pilot_v2_20260722`
- All five arms completed 30 ticks with exit code 0 and action parse rate 1.00.
- Model digest: `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7` (Gemma 4 31.3B, Q4_K_M).
- Source diff SHA-256: `262d995159c4d3672aa1b5dba3bc78f2de6865f2d54fc3c2e56d32253ead5e79`.
- Free/consensus/roles/integrated each returned 0.333 with one achievement and zero environment coordination successes. Cohesion returned 10.667 with six achievements and one coordination success; this is not an efficacy claim at one seed.
- Consensus: 90/90 protocol parse, 3 agreements, 64 commits, but inconsistent commit action strings and 26 failed coordination attempts.
- Roles: 85/90 protocol parse, 2 accepted awards, 0 valid awards, and numerous orphan accepts.
- Cohesion: 90/90 status messages and full coverage, but excessive every-tick reporting.
- Integrated: 89/90 protocol parse and full coverage, but zero bid/award messages; status crowded out allocation.

### v3 preregistration

`registry_v3.yaml` freezes a targeted 14-tick roles/cohesion/integrated rerun using rotating coordinator epochs and explicit phases. New metrics distinguish valid/coherent commits and orphan accepts.

### v3 completed

- Output: `outputs/alem_eval/coordination_pilot_v3_20260722`; all three arms exited 0, ran 14 ticks, and parsed every environment action.
- Roles: return 0.333, one achievement, 18/42 protocol messages, one valid of two awards, both awards accepted, four orphan accepts. Shared epoch IDs repaired one allocation but phase adherence remained poor.
- Cohesion: return 9.0, two achievements, one coordination success, 42/42 valid statuses and full coverage. The requested rate limit was ignored.
- Integrated: return 0.333, one achievement, 42/42 parse, one agreement, 15 valid action-coherent commits, no bids/awards, and 0.667 status coverage. Explicit phases did not induce full composition.
- Post-pilot hardening rejects multiple DCP records and adds phase-adherence metrics. No result is claimed for these post-pilot changes.
