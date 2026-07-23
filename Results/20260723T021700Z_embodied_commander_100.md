# Embodied Commander 100-Step Study

This is a one-seed, 100-tick Easy comparison; findings are descriptive, not confirmatory evidence.

Preregistered automated gate: **FAIL**.

## Episode results

| Arm | Seed | Steps/end | Ach. % | Return | Parse | Plan coverage | Plan valid | Status valid/coverage | Calls/requests/errors | Tokens | Wall/step |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| baseline | 12000 | 100/environment_truncated | 0.075 | 4.033 | 100.0% | unavailable | unavailable | unavailable/unavailable | 300/300/0 | 1,541,646 | 2.783 |
| embodied_commander_broadcast | 12000 | 100/environment_truncated | 0.065 | 5.333 | 100.0% | 0.950 | 0.633 | 1.000/1.000 | 330/330/0 | 1,920,993 | 3.661 |

## Paired Source contrasts

- Seed 12000: achievement Δ -0.011; return Δ 1.300; parse Δ 0.000; token regression 0.246; wall/step regression 0.316.

## Gate details

- PASS — `complete_no_transport_errors`
- PASS — `at_least_one_valid_plan`
- FAIL — `valid_plan_calls_at_least_90pct`
- PASS — `active_plan_coverage_at_least_90pct`
- PASS — `both_action_parse_rates_at_least_95pct`
- PASS — `valid_status_reports_at_least_90pct`
- PASS — `valid_status_coverage_at_least_80pct`
- PASS — `zero_accepted_unauthorized_plans`
- PASS — `zero_accepted_stale_statuses`
- PASS — `zero_hidden_state_guard_violations`

## Audit limits

- Automated leakage checks establish construction-level data boundaries and zero accepted unauthorized/stale records; semantic trace review remains manual.
- Failed attempts remain in token and transport accounting through the append-only attempt ledger, while outcome metrics use only complete canonical episodes.
- The baseline is the unchanged Source action path; the treatment adds a serial Agent 0 planning call, leased assignments, executor authority prompts, and SCP1 status validation.
