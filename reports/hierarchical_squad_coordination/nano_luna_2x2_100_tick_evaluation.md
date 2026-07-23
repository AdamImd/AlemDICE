# Nano/Luna Commander Evaluation: 100-Tick 2×2 Study

## Executive summary

This experiment tested whether a high-reasoning GPT-5.6 Luna commander improves a
three-agent squad whose action agents are GPT-5.4 nano. The action agents were
evaluated at both `none` and `high` reasoning, with and without the embodied
commander broadcast. All four arms used the same seed and ran concurrently for 100
ticks.

The commander produced a positive signal when the nano action agents used no
reasoning: return increased from 1.27 to 3.47 and team achievement rate increased
from 2.15% to 8.60%. It produced a strong negative signal when the action agents
used high reasoning: return decreased from 12.33 to 3.03 and team achievement rate
decreased from 13.98% to 8.60%. The high-reasoning Source baseline was the best arm
by a wide margin.

The result suggests a conditional mechanism rather than a generally superior
hierarchy: explicit plans may scaffold weak executors, while the same plans can
constrain capable executors or introduce a planning bottleneck. This is a
one-seed study, so the differences are descriptive signals and not statistically
reliable treatment effects.

## Research question and design

The study asks two related questions:

1. Does the embodied commander improve behavior relative to the unchanged Source
   action path?
2. Does its effect depend on the reasoning capability of the action agents?

The matrix was:

| Arm | Action agents | Action reasoning | Commander |
| --- | --- | --- | --- |
| Source / none | GPT-5.4 nano | `none` | none |
| Commander / none | GPT-5.4 nano | `none` | GPT-5.6 Luna, `high` |
| Source / high | GPT-5.4 nano | `high` | none |
| Commander / high | GPT-5.4 nano | `high` | GPT-5.6 Luna, `high` |

All arms used:

- Alem `default`, Easy difficulty;
- three embodied action agents;
- seed `12100`;
- a 100-tick environment limit;
- one episode;
- the same normal Source prompt and action path in the control arms;
- concurrent execution of all four arms; and
- W&B disabled, with local artifacts retained.

The commander treatment adds one planning role for physical Agent 0, leased
per-agent assignments, executor authority prompts, and validated SCP1 status
messages. Planning is serial with respect to a treatment arm, but the three action
agents are called concurrently each tick. The four experimental arms were also
launched concurrently.

## Results

| Arm | Return | Team achievement | Coordination achievement | Action parse | Calls | Tokens | Wall seconds/tick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source / nano none | 1.27 | 2.15% | 0.00% | 96.43% | 300 | 1,672,278 | 2.86 |
| Commander / nano none | 3.47 | 8.60% | 0.00% | 99.33% | 330 | 2,548,721 | 10.25 |
| Source / nano high | **12.33** | **13.98%** | **7.41%** | 99.00% | 300 | 2,457,664 | 9.92 |
| Commander / nano high | 3.03 | 8.60% | 0.00% | 98.77% | 328 | 2,535,733 | 13.94 |

### Commander effect with nano reasoning disabled

Relative to Source, the commander changed:

- return by **+2.20**;
- team achievement rate by **+6.45 percentage points**;
- action parse rate by **+2.90 percentage points**;
- total tokens by **+52.4%**; and
- wall time per tick by **+258.0%**.

The treatment expanded the achieved set from two achievements
(`collect_sapling`, `collect_wood`) to eight. The additions included coal and
stone collection, wood-tool crafting, and table and plant placement. It did not
produce a coordination achievement.

### Commander effect with nano high reasoning

Relative to Source, the commander changed:

- return by **−9.30**;
- team achievement rate by **−5.38 percentage points**;
- coordination achievement rate by **−7.41 percentage points**;
- action parse rate by **−0.23 percentage points**;
- total tokens by **+3.2%**; and
- wall time per tick by **+40.6%**.

The Source arm completed thirteen distinct listed achievements and was the only
arm to obtain `coord_mine_handover` and `handover_complete`. The commander arm
completed eight listed achievements, obtained no coordination achievement, and
experienced one Agent 1 death. The descriptive difference-in-differences was
−11.50 return and −11.83 percentage points of team achievement. This large
interaction is the central signal from the matrix.

## What the mechanism did well

The runtime integration itself was reliable:

- all four arms reached 100 ticks;
- the action and planner models were routed as specified;
- no provider transport request failed;
- treatment active-plan coverage was 95% in both reasoning conditions;
- treatment status validity and coverage were about 92–93%; and
- no accepted authority, stale-plan, or context-leak violation was detected.

Leased plans also provided useful fault tolerance. Of 21 invalid commander calls,
19 retained a previously accepted plan, so 64% call validity still yielded 95%
active-plan coverage. Most rejected calls recovered at the next review or within
five ticks.

This explains how the weak-executor treatment could show a behavioral gain despite
frequent planning protocol failures: accepted plans persisted across failed
reviews and gave otherwise non-reasoning workers stable task direction.

## What did not work

### The commander did not improve the strongest tested squad

The high-reasoning Source agents substantially outperformed every other arm.
Adding the commander suppressed rather than amplified their performance. A
plausible interpretation is that high-reasoning agents already adapt locally and
coordinate through the normal environment interface; binding them to centrally
generated leases makes them pursue stale, over-specific, or costly plans instead
of exploiting their own observations. The present experiment establishes the
performance pattern, but a trace-level behavioral annotation study is needed to
distinguish over-constraint from other explanations.

### High planner reasoning did not ensure protocol compliance

Across both treatment arms, the Luna commander made 58 calls:

- 37 were accepted;
- 21 were rejected;
- overall validity was 63.8%;
- the nano-none treatment accepted 19/30 (63.3%); and
- the nano-high treatment accepted 18/28 (64.3%).

The automated gate required at least 90% plan validity, so the preregistered gate
failed. All other automated integrity and runtime gates passed.

Failure taxonomy:

| Failure code | Count | Meaning |
| --- | ---: | --- |
| `response.plan_envelope_missing` | 13 | The response omitted the required plan envelope, often while returning otherwise plausible bare JSON. |
| `assignment.sync_invalid` | 6 | A proposed synchronization window violated the active lease bounds. |
| `assignment.fields_invalid` | 1 | Required assignment fields were absent. |
| `assignment.text_field_invalid` | 1 | A bounded text field had the wrong type or size. |

These errors show two different problems. Missing envelopes and malformed fields
are representation failures. Invalid synchronization windows are semantic
contract failures: the planner generated coherent-looking future plans but did
not respect the bounded lifetime in which those plans were valid.

### Representative failed planner calls

At tick 29 in the high-reasoning treatment, the commander proposed a coordinated
tree harvest with sync ticks 46–55. A plan issued at tick 29 under a ten-tick
lease must synchronize before tick 39. The validator rejected the plan as
`assignment.sync_invalid` and retained the previous plan.

At tick 27 in the no-reasoning treatment, the commander proposed synchronized
table placement at ticks 42–70. This was again well outside the current lease.
The call contained a detailed and apparently intentional multi-stage plan, so the
failure cannot be explained as a simple JSON typo.

At tick 48 in the no-reasoning treatment, all three assignments omitted the
required `dependencies` field. The proposed resource plan was intelligible, but
it could not pass the assignment schema.

At tick 65 in the high-reasoning treatment, an emergency retreat proposal was
returned without the required plan envelope. Rejection retained the previous
plan for ten ticks. This is an important failure mode: strict validation protects
the executor contract, but can also discard a useful safety intervention.

At tick 70 in the high-reasoning treatment, an out-of-lease synchronization plan
was rejected when there was no valid retained plan. The squad therefore had no
active plan until a valid replacement five ticks later.

At tick 95 in the no-reasoning treatment, `target` was emitted as a structured
object rather than the required bounded string. No retained plan was available,
and recovery was not observed before the episode ended.

## Cost and scalability

The planner was expensive relative to its measured benefit. With non-reasoning
workers, it added about 876,000 tokens and increased wall time per tick from 2.86
to 10.25 seconds. With high-reasoning workers, token count increased only 3.2%,
but wall time still increased 40.6% because planning lies on the treatment's
critical path. Commander planning alone consumed approximately 662 seconds in the
nano-none treatment and 496 seconds in the nano-high treatment.

This implementation therefore does not yet support the paper's scalability claim.
It is a useful auditable prototype, but planner scheduling and selective
intervention must improve before extending it to multiple squads.

## Interpretation

The evidence supports a narrower hypothesis than “a commander improves a team”:

> A structured commander may scaffold action agents with weak deliberation, but
> unconditional command can interfere with action agents that already deliberate
> effectively.

That hypothesis is consistent with the positive weak-worker contrast and the
negative strong-worker contrast. It also suggests that the best next system is
not a permanently controlling hierarchy. A better design may preserve the Source
agent's autonomy and invoke command only when coordination is genuinely required,
when agents report low confidence, or when progress stalls.

No confirmatory claim should be made from one seed. Stochastic environment layout,
model sampling, and the single death in one treatment arm could materially affect
the result.

## Recommended next experiments

1. Repeat the full matrix for at least three preregistered seeds. Report paired
   seed-level differences, confidence intervals, deaths, and achievement sets.
2. Add a structured-output or grammar-constrained planner arm. This isolates plan
   quality from envelope and field-shape failures without silently changing the
   present method.
3. Replace absolute sync ticks in planner output with relative offsets, or derive
   bounded sync windows in deterministic middleware. Compare this with the strict
   current protocol.
4. Test an adaptive commander that intervenes only on coordination tasks, stalled
   progress, or explicit executor requests. Leave high-capability agents on the
   Source action path at other times.
5. Annotate a stratified sample of accepted and rejected plans for feasibility,
   grounding, safety, staleness, and whether executor compliance helped or harmed
   progress.

The implementation deliberately does not apply these fixes retroactively; doing
so would make the current evidence harder to audit.

## Reproducibility and audit artifacts

- Machine-readable results:
  [`Results/20260723T031757Z_nano_luna_2x2_100.json`](../../Results/20260723T031757Z_nano_luna_2x2_100.json)
- Generated compact results:
  [`Results/20260723T031757Z_nano_luna_2x2_100.md`](../../Results/20260723T031757Z_nano_luna_2x2_100.md)
- Run manifest:
  [`commander_study_manifest.json`](../../outputs/alem_eval/20260723T031757Z_nano_luna_2x2_100/commander_study_manifest.json)
- Commander failure summary:
  [`commander_failure_summary.md`](../../outputs/alem_eval/20260723T031757Z_nano_luna_2x2_100/commander_failure_summary.md)
- Failed-call records:
  [`commander_failures.jsonl`](../../outputs/alem_eval/20260723T031757Z_nano_luna_2x2_100/commander_failures.jsonl)
- Human annotation sidecar:
  [`commander_failure_annotations.jsonl`](../../outputs/alem_eval/20260723T031757Z_nano_luna_2x2_100/commander_failure_annotations.jsonl)

Every native commander call was replayed through the validator: all 58 acceptance
decisions reproduced. Commander journals are append-only and durably flushed;
episode records retain journal counts and hashes. The human annotation sidecar is
intentionally blank. Subjective judgments about semantic plan quality are
deferred rather than presented as completed evidence.

Implementation verification before the paid run included Ruff and the focused
LLM baseline suite, with **32 tests passing**. A three-call routing preflight
confirmed exactly two nano action calls and one high-reasoning Luna planner call,
including nonzero planner reasoning tokens.
