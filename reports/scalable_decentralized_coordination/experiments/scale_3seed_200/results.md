# Three-seed, 200-step signal study

**Execution status:** stopped at the user's request on 2026-07-22 at 16:22:27 PDT. The `free_concise` and `cohesion_concise` arms completed all three 200-step episodes. The `free_thinking` arm stopped after nine steps per seed and has no valid terminal episodes. Its telemetry is retained as partial, non-efficacy evidence.

## Findings

### Structured cohesion: positive by the registered rule, but mixed and costly

Cohesion improved total achievements on two paired seeds but hurt the strongest baseline seed: paired changes were **-3, +2, and +2**. It therefore passes the preregistered exploratory rule, while the mean gain was only 0.333 achievements (7.333 versus 7.000). Mean return fell by 0.533 (9.511 versus 10.044). With only three seeds and one large negative result, this is a heterogeneous signal rather than evidence of a reliable overall win.

The mechanism did not increase aggregate environment coordination success: both arms recorded three successes. Cohesion increased attempts from 50 to 194, so the observed success-per-attempt ratio fell from 6.0% to 1.5%. Action parsing remained 1.00 in both arms.

The status scaffold was mostly protocol-parseable and maintained high sender-window coverage, but its rate control failed:

- Protocol parse rates were 0.998, 0.998, and 0.812; status-window coverage was 1.000, 1.000, and 0.917. These metrics measure syntax and sender presence, not semantic task or role coherence.
- Agents produced 599, 599, and 487 valid statuses. Schedule adherence was only 0.200, 0.200, and 0.205 because statuses were emitted much more often than the requested heartbeat interval.
- Both arms emitted 600 communications per seed, so cohesion did not increase total message frequency relative to this baseline. Its larger prompts/context and 12.8% higher output tokens per call are the more plausible sources of the extra compute cost.
- Total tokens increased 3.5%, arm elapsed time increased 11.3%, and delivered bytes per step fell 8.9% because typed statuses were generally shorter than free-form messages.

Overall, cohesion deserves another controlled study, but the next version should enforce heartbeat throttling in code and investigate why seed 10001 generated 145 coordination attempts for one success.

### Concise response mode: strong operational signal; task efficacy unresolved

The full response-mode contrast is **indeterminate** because the thinking arm was stopped. Over the matched first nine steps, however, concise mode was more reliable and efficient:

- Concise parsed 81/81 actions (1.00). Thinking parsed 73/81 (0.901), with paired seed rates of 0.926, 0.852, and 0.926.
- Thinking used 7.6x, 8.5x, and 11.3x as many output tokens per call as concise.
- Thinking calls were 4.4x, 4.6x, and 3.8x slower on average.
- Thinking hit the 2,048-token ceiling nine times and produced eight empty visible outputs; concise had neither failure in the matched prefix.

This was not a pure quality win for concise. First-nine-step reward sums were `[1, 2, 2]` for concise and `[1, 12, 13]` for thinking. Those prefixes are too short for efficacy claims, but they warn that removing reasoning may trade away useful planning even while dramatically improving throughput and action delivery. A future run should test a bounded-reasoning mode that reserves tokens for the required action tags.

### Bottom line

- **Cohesion:** a preregistered positive achievement-direction signal in two of three seeds, offset by a negative seed, unchanged coordination-success count, many more failed attempts, and about 11% wall-time overhead.
- **Concise output:** a strong matched-prefix reliability and throughput benefit, but no completed efficacy comparison; early reward favors thinking on two seeds.
- **What is supported:** cohesion statuses were mostly protocol-parseable with high sender-window coverage for 200 steps, and the thinking-enabled response bundle under the 2,048-token cap was operationally expensive and failure-prone.
- **What is not supported:** that cohesion reliably raises average task performance, that it improves environment coordination efficiency, that the status contents were semantically coherent, or that tags-only responses preserve the planning quality of internal reasoning.

This report evaluates three paired seeds (9999, 10000, and 10001). The three episodes within each arm ran in parallel. Arms ran sequentially.

A valid episode must have the canonical episode schema, a complete artifact status, no error, a terminal reason, no repeated-length early stop, and the preregistered seed. Natural environment termination before 200 steps remains valid.

## Arm-level results

| Arm | Valid | Reached 200 | Return mean | Achievements mean | Coord. successes mean | Parse rate mean | Input/call | Output/call | Wall/step | Delivery/step | Recorded tokens | Arm elapsed (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free_thinking | 0/3 | 0 | — | — | — | — | — | — | — | — | 614,117 partial | — |
| free_concise | 3/3 | 3 | 10.044 | 7.000 | 1.000 | 1.000 | 7,300.704 | 113.411 | 29.331 | 500.067 | 13,345,407 | 5,913.9 |
| cohesion_concise | 3/3 | 3 | 9.511 | 7.333 | 1.000 | 1.000 | 7,543.010 | 127.902 | 32.662 | 455.737 | 13,807,641 | 6,583.1 |

## Seed-level results

| Arm | Seed | Valid | Steps | Return | Achievements | Coord. successes | Parse rate | Input tokens | Output tokens | Calls | Input/call | Output/call | Wall/step | Delivery/step |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free_thinking | 9999 | no (missing_artifact) | — | — | — | — | — | — | — | — | — | — | — | — |
| free_thinking | 10000 | no (missing_artifact) | — | — | — | — | — | — | — | — | — | — | — | — |
| free_thinking | 10001 | no (missing_artifact) | — | — | — | — | — | — | — | — | — | — | — | — |
| free_concise | 9999 | yes | 200 | 25.000 | 16.000 | 2.000 | 1.000 | 3,805,769 | 73,704 | 600 | 6,342.948 | 122.840 | 29.343 | 530.260 |
| free_concise | 10000 | yes | 200 | 4.133 | 3.000 | 1.000 | 1.000 | 4,391,476 | 62,723 | 600 | 7,319.127 | 104.538 | 29.247 | 444.480 |
| free_concise | 10001 | yes | 200 | 1.000 | 2.000 | 0.000 | 1.000 | 4,944,022 | 67,713 | 600 | 8,240.037 | 112.855 | 29.402 | 525.460 |
| cohesion_concise | 9999 | yes | 200 | 19.133 | 13.000 | 1.000 | 1.000 | 4,001,948 | 76,571 | 600 | 6,669.913 | 127.618 | 32.694 | 435.920 |
| cohesion_concise | 10000 | yes | 200 | 4.733 | 5.000 | 1.000 | 1.000 | 4,632,055 | 72,159 | 600 | 7,720.092 | 120.265 | 32.535 | 448.230 |
| cohesion_concise | 10001 | yes | 200 | 4.667 | 4.000 | 1.000 | 1.000 | 4,943,415 | 81,493 | 600 | 8,239.025 | 135.822 | 32.757 | 483.060 |

## Interrupted partial telemetry (non-efficacy)

These per-step debug streams have no canonical episode artifact. They describe only an observed prefix and are excluded from efficacy aggregates, paired contrasts, and preregistered decisions.

| Arm | Seed | Steps | Reward sum | Calls | Input | Output | Mean latency (s) | Length stops | Empty | Parse ok | Parse fail | Parse rate | Transport errors | Comms | Delivery bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free_thinking | 9999 | 9 | 1.000 | 27 | 164,878 | 27,536 | 75.864 | 2 | 2 | 25 | 2 | 0.926 | 0 | 25 | 5,764 |
| free_thinking | 10000 | 9 | 12.000 | 27 | 170,059 | 30,220 | 71.459 | 4 | 4 | 23 | 4 | 0.852 | 0 | 23 | 3,886 |
| free_thinking | 10001 | 9 | 13.000 | 27 | 187,297 | 34,127 | 73.257 | 3 | 2 | 25 | 2 | 0.926 | 0 | 25 | 6,602 |

## Matched response prefix (descriptive only)

The interrupted thinking arm is compared with the same observed steps from the completed concise arm. These rows are excluded from the registered efficacy decision. Exact source slices and hashes are in the artifact inventory.

| Seed | Steps | Complete | Concise reward | Thinking reward | Concise parse | Thinking parse | Concise output/call | Thinking output/call | Concise latency | Thinking latency | Concise length/empty | Thinking length/empty |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9999 | 9 | yes | 1.000 | 1.000 | 1.000 | 0.926 | 134.889 | 1,019.852 | 17.238 | 75.864 | 0/0 | 2/2 |
| 10000 | 9 | yes | 2.000 | 12.000 | 1.000 | 0.852 | 132.444 | 1,119.259 | 15.536 | 71.459 | 0/0 | 4/4 |
| 10001 | 9 | yes | 2.000 | 13.000 | 1.000 | 0.926 | 111.704 | 1,263.963 | 19.203 | 73.257 | 0/0 | 3/2 |

## Paired contrasts

### response_mode

Treatment minus control: free_concise − free_thinking.

| Seed | Pair valid | Δ parse | Δ achievements | Δ return | Δ coord. successes | Δ output/call | Δ wall/step |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9999 | no | — | — | — | — | — | — |
| 10000 | no | — | — | — | — | — | — |
| 10001 | no | — | — | — | — | — | — |

Decision: **indeterminate_incomplete_pairs**.

- Decision criteria were not evaluated because the required three valid seed pairs were unavailable.

### cohesion

Treatment minus control: cohesion_concise − free_concise.

| Seed | Pair valid | Δ parse | Δ achievements | Δ return | Δ coord. successes | Δ output/call | Δ wall/step |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9999 | yes | 0.000 | -3.000 | -5.867 | -1.000 | 4.778 | 3.351 |
| 10000 | yes | 0.000 | 2.000 | 0.600 | 0.000 | 15.727 | 3.288 |
| 10001 | yes | 0.000 | 2.000 | 3.667 | 1.000 | 22.967 | 3.356 |

Decision: **positive_signal**.

- achievements_improve_on_at_least_two_seeds: yes
- parse_lower_on_at_most_one_seed: yes

## Audit trail

- The preregistered design and pre-study interpretation are preserved in [`registry.yaml`](registry.yaml) and [`current_signals.md`](current_signals.md).
- Machine-readable outcomes are in [`summary.json`](artifacts/summary.json), [`episodes.csv`](artifacts/episodes.csv), [`paired_contrasts.csv`](artifacts/paired_contrasts.csv), and [`matched_prefix_response.csv`](artifacts/matched_prefix_response.csv).
- [`artifact_inventory.json`](artifacts/artifact_inventory.json) records hashes for the derived tables and the archived raw episodes, event log, configs, attempt ledgers, interrupted telemetry, and exact matched-prefix source slices.
- The runner recorded source commit `4c0a35fcde2bc8bc5d5172fd48413d9856c19e54`; the stop event and interrupted command are preserved in [`events.jsonl`](artifacts/audit/study/events.jsonl).

## Interpretation limits

- Three paired seeds are descriptive and underpowered; no p-values or confidence intervals are claimed.
- All three valid pairs are required for either preregistered decision; otherwise the result is indeterminate.
- Failed or missing seed pairs receive no numeric delta and cannot count as an improvement.
- Partial debug telemetry is non-efficacy evidence and cannot make a missing episode or pair valid.
- Paired behavioral and response-cost contrasts use the final stable episode artifact. Cumulative attempt usage separately includes failed retries from the durable attempt ledger.
- Coordination successes are a secondary cohesion diagnostic; the cohesion decision uses total achievements and action parsing.
- Episode wall times overlap because seeds ran concurrently. Arm elapsed time is derived from runner events; model latency is not treated as wall time.
