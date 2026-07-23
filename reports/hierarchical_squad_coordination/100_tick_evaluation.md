# 100-tick Source vs. embodied-commander evaluation

Status: **FROZEN BEFORE RUN — RESULTS PENDING**

Protocol: `alem-dice-embodied-commander-study-v1`, stage `100`

Date authorized: 2026-07-22

Branch: `research/embodied-squad-commander`

## Question and scope

This evaluation compares two coordination conditions using the same
`gpt-5.6-luna` worker model:

1. `baseline`: unchanged Source decentralized behavior and peer broadcast.
2. `embodied_commander_broadcast`: a serial Agent 0 planning phase creates
   leased assignments before the same three parallel worker action calls.

Calling these the “two models” refers to two agent-system conditions, not two
different provider model IDs. Using one model in both arms isolates the
coordination intervention.

This is one matched Easy seed (`12000`) with a 100-tick cap per arm. It can
provide mechanism and descriptive outcome evidence, but cannot establish an
average treatment effect or statistical significance.

## Frozen configuration

| Field | Value |
| --- | --- |
| Profile | `embodied_commander_100` |
| Task/difficulty | `alem/default`, Easy |
| Seed | `12000` in both arms |
| Tick cap | 100 per arm |
| Agents | Three physical and three logical participants |
| Model/API | `gpt-5.6-luna`, OpenAI Responses |
| Reasoning/temperature | `none` / `1.0` |
| Maximum output | 8,192 tokens per logical call |
| Episode workers | One; arms run sequentially |
| Within-tick actions | Three concurrent worker calls |
| Commander review/lease | Every 5 ticks / 10 ticks |
| Unscheduled review cap | 10 |
| Debriefs/W&B | Disabled / disabled |

The baseline permits at most 300 worker calls. The commander arm permits the
same 300 worker calls, 20 scheduled planning calls, and at most 10 unscheduled
repair/event planning calls. The study ceiling is 630 logical calls, excluding
transport retries. The separate preflight adds exactly two logical calls.

## Frozen structural gate

The automated mechanism gate passes only if:

- both canonical episodes complete with zero transport errors;
- the commander produces at least one valid plan;
- plan validity is at least 0.90;
- active-plan coverage is at least 0.90;
- both action-parse rates are at least 0.95;
- conditional status validity is at least 0.90;
- valid-status coverage is at least 0.80 of all eligible agent-turns; and
- no unauthorized plan, stale status, or structural hidden-state violation is
  accepted.

Valid-status coverage over all eligible agent-turns is reported separately from
conditional status validity. The commander executor contract requires a status
on each active turn so the feedback loop is meaningfully exercised; the lower
0.80 gate tolerates occasional malformed or omitted outputs without hiding them.

The last three zero counters are construction invariants, not independent
semantic detectors. They must not be used as substitutes for raw-trace review.

## Required trace review

The final report will inspect every commander planning call; all invalid,
replacement, event, death, reward, and parse-failure ticks; and systematic
10-tick action samples. It will check:

- baseline prompts contain no commander intervention;
- planner prompts contain only Agent 0's legal observations and previously
  validated status reports;
- every accepted plan covers exactly Agents 0, 1, and 2;
- all three same-tick worker prompts receive the accepted full plan and the
  correct highlighted assignment;
- Agent 0 does not revise assignments during its action phase;
- wingmen do not invent objectives or reassign work;
- status versions/tasks match the sender's active assignment and evidence is
  grounded in its visible observation; and
- actions materially pursue assignments rather than only formatting valid
  statuses.

Raw `*_debug.jsonl` is the semantic source of truth. The existing HTML viewer
does not render commander planning records. Serial plan-before-action ordering
is supported by the same-tick prompt dependency and frozen code path; the debug
schema does not contain independent per-call start/end timestamps.

## Results

Pending. This section will be populated only from canonical episode artifacts,
append-only attempt ledgers, the generated summary, and the raw-trace audit.
