# E1 Source Scaling Summary

Source root: `outputs/alem_eval/e1_source_scaling_3seed_200_v2`

| Agents | Seeds | Total % | Base % | Coord % | Per-agent return | Unique team first-unlocks | Summed agent first-unlocks | Alive-turn fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 3.379 | 3.379 | — | 7.133 | 7.333 | 7.333 | 1.000 |
| 2 | 3 | 4.699 | 5.376 | 3.774 | 14.283 | 10.333 | 16.000 | 1.000 |
| 3 | 3 | 9.752 | 12.596 | 5.870 | 21.678 | 19.333 | 36.333 | 0.980 |
| 4 | 3 | 9.574 | 8.602 | 10.901 | 20.375 | 19.333 | 42.333 | 0.994 |
| 6 | 3 | 6.738 | 8.141 | 4.822 | 11.883 | 15.000 | 42.333 | 0.996 |

Intervals in `summary.json` and the figure resample seed IDs within each population. With three canonical seeds they are coarse descriptive uncertainty intervals, not hypothesis tests. Raw seed values are retained in `episodes.csv`.

Base/Coord/Total are reward-weighted paper scores on a 0–100 scale. Achievement counts are cumulative binary first-unlock types, not repeated events. Team-unique counts each achievement type once across the team; the summed agent count can count the same type once for every attaining agent.

## Exposure, cache, and time

| Agents | Actual/requested ticks | Alive-turn fraction | Actionable-turn fraction | Cached input | Uncached input | Cache fraction | Episode wall (s) | Summed model latency (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200.0/200.0 | 1.000 | 1.000 | 0.0 | 769764.7 | 0.000 | 909.78 | 873.06 |
| 2 | 200.0/200.0 | 1.000 | 0.958 | 709632.0 | 1665833.3 | 0.302 | 1380.96 | 1947.57 |
| 3 | 197.0/200.0 | 0.980 | 0.869 | 1036373.3 | 2499944.7 | 0.294 | 1708.81 | 2902.96 |
| 4 | 200.0/200.0 | 0.994 | 0.941 | 1421056.0 | 3691075.7 | 0.279 | 2068.07 | 4318.82 |
| 6 | 200.0/200.0 | 0.996 | 0.901 | 2121728.0 | 6214951.3 | 0.258 | 2933.41 | 7057.82 |

Summed model latency adds per-call latency and can exceed episode wall time because physical-agent requests overlap within a tick.

| Agents | Input tokens/submitted | Input tokens/actionable | Total tokens/submitted | Total tokens/actionable | Broadcast bytes/submitted | Broadcast bytes/actionable |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3848.82 | 3848.82 | 4313.51 | 4313.51 | 0.00 | 0.00 |
| 2 | 5938.66 | 6201.47 | 6481.98 | 6769.90 | 49.18 | 50.97 |
| 3 | 5978.80 | 6956.08 | 6555.39 | 7625.79 | 113.04 | 129.97 |
| 4 | 6390.16 | 6808.00 | 6989.24 | 7445.84 | 185.22 | 196.78 |
| 6 | 6947.23 | 7697.21 | 7605.03 | 8427.78 | 351.34 | 388.53 |

Byte rates are Source ordinary peer-broadcast delivered fan-out bytes, not serialized prompt bytes or provider network traffic.

## Provider retry audit

| Agents | Logical responses | Provider attempts | Recovered retries | Episodes with retries |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 600 | 600 | 0 | 0 |
| 2 | 1200 | 1200 | 0 | 0 |
| 3 | 1773 | 1773 | 0 | 0 |
| 4 | 2400 | 2401 | 1 | 1 |
| 6 | 3600 | 3600 | 0 | 0 |

Recovered transport error types: `APIConnectionError`=1. Campaign usage including preflight is 9574/12001 logical responses and 9575/15000 provider attempts.

A recovered retry is accepted only when every logical decision has one final completed response and `provider_attempts = logical_responses + transport_errors` reconciles globally, by worker, and against the attempt ledger. Unrecovered failures, incomplete responses, untyped errors, and unaccounted attempts remain disqualifying.

## Noop audit

| Agents | Intentional actionable | Parse fallback | Validation fallback | Inactive effective | Residual | Canonical submitted | Effective environment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.33 | 0.33 |
| 2 | 16.67 | 2.00 | 0.00 | 17.00 | 0.00 | 35.67 | 35.67 |
| 3 | 7.00 | 5.00 | 0.00 | 78.00 | 0.00 | 90.00 | 90.00 |
| 4 | 17.67 | 11.33 | 0.00 | 47.33 | 0.00 | 76.33 | 76.33 |
| 6 | 34.67 | 15.33 | 0.00 | 119.33 | 0.00 | 169.33 | 169.33 |

Noop provenance: `debug_jsonl_legacy_reconstruction`=15. Per-episode reconstruction notes are retained in `episodes.csv`.

Canonical submitted Noops are the post-validation actions passed to `env.step` and are the only values compared with legacy `action_frequency.Noop`. Effective environment Noops additionally include every inactive worker turn, because Alem masks those actions to Noop. The five cause columns form an exact partition of effective environment Noops when accounting is complete.

## Terminations

- N=1: environment_truncated=3
- N=2: environment_truncated=3
- N=3: environment_terminated=1, environment_truncated=2
- N=4: environment_truncated=3
- N=6: environment_truncated=3

Actual tick counts are reported separately from the manifest-requested tick cap; termination reasons are never silently treated as full-horizon runs.

## Paired population contrasts

| Contrast | Common seeds | Reward-weighted Total difference (points) | 95% descriptive interval |
| --- | ---: | ---: | ---: |
| n2_minus_n1 | 3 | 1.319 | [-2.144, 4.487] |
| n3_minus_n1 | 3 | 6.372 | [4.008, 9.540] |
| n4_minus_n1 | 3 | 6.195 | [3.176, 9.327] |
| n6_minus_n1 | 3 | 3.358 | [0.816, 6.615] |
| n3_minus_n2 | 3 | 5.053 | [2.394, 7.713] |
| n4_minus_n2 | 3 | 4.876 | [1.596, 7.713] |
| n6_minus_n2 | 3 | 2.039 | [-0.798, 4.787] |
| n4_minus_n3 | 3 | -0.177 | [-3.457, 5.319] |
| n6_minus_n3 | 3 | -3.014 | [-3.191, -2.926] |
| n6_minus_n4 | 3 | -2.837 | [-8.511, 0.532] |

Contrasts are higher-population minus lower-population values on common seed IDs and use a paired seed bootstrap. Three-seed intervals remain descriptive and coarse.

Coordination event totals are canonical environment counters, but their subdomains use heterogeneous counting units (for example per-timestep sync attempts versus per-setup handovers). Do not add the event fields together. Coordination is undefined for the N=1 wrapper and remains missing rather than zero.

The curve is descriptive: changing population also changes spawn geometry, mob pressure, specialization balance, prompt size, and coordination requirements.
