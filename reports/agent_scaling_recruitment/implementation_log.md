# Agent Scaling and Recruitment Implementation Log

This is the append-only human-readable tracker for
[`experiment_plan.md`](experiment_plan.md). Existing dated entries must not be
rewritten after results are inspected. Corrections are appended as new entries.
Machine-readable manifests and raw artifacts remain the source of truth.

## Milestone status

| Milestone | Status | Evidence |
| --- | --- | --- |
| Plan frozen | Complete | `experiment_plan.md`, 2026-07-22 |
| E0 harness | Complete | `scripts/run_e0_agent_compatibility.py`, commit `587e717` |
| E0 free compatibility run | Complete: PASS | `Results/e0_agent_compatibility_v1.{json,md}` |
| E0 LaTeX report | Complete | `e0_report.tex` and compiled `e0_report.pdf` |
| E0.1 200-step extension | Complete: structural PASS, strict horizon FAIL | `Results/e0_1_agent_compatibility_200_v1.{json,md}` |
| E1 Source scaling | Launch-ready | Source-only provider preflight pending |
| E2 RecruitmentArena-6 | Deferred | Begins after E0 |
| E3 Alem recruitment routing | Deferred | Begins after E2 promotion |

## 2026-07-22 — protocol freeze

- Created branch `research/agent-scaling-recruitment-pipeline` from commit
  `0edcd5a`.
- Recorded the approved E0–E3 experiment sequence.
- Fixed the behavioral model choice to homogeneous GPT-5.4 nano with high
  reasoning.
- Fixed the scale ladder to the documented 1–4 range followed by a gated
  six-agent extension.
- Fixed ordinary communication to team members only after team formation, with
  a bounded typed bulletin as the only pre-team channel.
- Fixed task-bound leases, variable two- and three-agent teams, locally
  announced Alem tasks, and the three-rule mathematical roster ablation.
- Restricted the current implementation checkpoint to E0. No hosted model call
  is authorized by this checkpoint.

## 2026-07-22 — E0 implementation and compatibility result

- Implemented the provider-free E0 runner and eight focused tests. The harness
  was committed at `587e7176ce3b02296120e9504c1d7d609e9dcf2a` before the
  measured run.
- Focused verification passed: 8/8 pytest cases, Ruff clean, Hydra configuration
  dry run complete.
- Final regression verification passed all 40 lightweight `baselines/llm`
  pytest cases; the E0 runner and focused tests remained Ruff clean.
- Executed `N={1,2,3,4,6}`, seeds 13000 and 13001, and 10 ticks per episode
  through the canonical evaluator. The run began at
  `2026-07-23T06:24:16.409143Z`, ended at
  `2026-07-23T06:28:23.047401Z`, and took 246.638 seconds.
- E0 result: **PASS**. All 10 episodes completed. The run covered 100
  environment ticks and 320 scripted agent-ticks, with action parse rate 1.0,
  zero model calls, zero provider requests, and zero transport errors.
- Every population passed client/agent cardinality, trajectory dimensions,
  warrior/forager/miner specialization cycling, all ordered targeted-`Give`
  mappings, exact broadcast accounting, complete state/trajectory artifacts,
  and deterministic saved-state reconstruction hashes.
- Communication accounting matched the preregistered formulas. Across the
  matrix, 320 messages produced 1,000 delivered copies and 30,000 delivered
  bytes.
- Full-world renderer smoke checks passed for every population. Each seed-13000
  state produced a one-frame 768×832 H.264 artifact accepted by `ffprobe`.
- Raw outputs are under
  `outputs/alem_eval/e0_agent_compatibility_v1/`. The raw result JSON SHA-256 is
  `7b67b56544a00df63b1e07106c9eeffe51eedc9bc62dc7d8d914af734ad8c47d`.
- Published compact results are
  `Results/e0_agent_compatibility_v1.json` and
  `Results/e0_agent_compatibility_v1.md`. Methods, results, limitations, and the
  promotion decision are recorded in `e0_report.tex` and the compiled
  `e0_report.pdf`.
- Promotion decision: E0 removes the compatibility blocker for E1a at
  `N={1,2,3,4}` and the engineering blocker for the separately labeled `N=6`
  extension. It is not behavioral scaling evidence.
- The source-status manifest recorded the pre-existing untracked replay
  `Results/replays/nano_high_source_full_world.mp4`. It was outside the E0 root
  and was not modified or used.

## 2026-07-22 — E0.1 200-step extension and timing result

- Extended E0 to 200 requested turns per episode at `N={1,2,3,4,6}` and seeds
  13000 and 13001. Added monotonic timing for complete episode wall time,
  complete evaluator turns, the slowest turn, and the cumulative parallel
  worker-action phase.
- Committed the timing and natural-termination-aware validator at
  `5c4c64338b9aae9132f9cc32b6eec3d746964529` before the measured run.
  Verification passed 9/9 focused E0 tests, all 41 lightweight
  `baselines/llm` tests, Ruff, published-JSON/raw-result consistency assertions,
  and `git diff --check`.
- Preserved the first diagnostic attempt at
  `outputs/alem_eval/e0_1_agent_compatibility_200_v1/`. Its original strict
  validator stopped after the two `N=1` episodes because seed 13000 naturally
  ended at turn 114. No diagnostic artifacts were overwritten.
- Re-ran the complete matrix at
  `outputs/alem_eval/e0_1_agent_compatibility_200_v2/`. It began at
  `2026-07-23T06:57:25.963656Z`, ended at
  `2026-07-23T07:08:25.236294Z`, and took 659.273 seconds. The run contains 82
  files; its raw result SHA-256 is
  `6d28e885e241b4f58553291d6fa205fb982878f2575f6bdd3a24d711fda9b529`.
- E0.1 strict result: **FAIL**, because only 7/10 episodes reached the requested
  turn 200. `N=1` seed 13000 ended at 114, `N=4` seed 13001 at 182, and `N=6`
  seed 13001 at 192 after all agents died from mob combat.
- E0.1 structural result: **PASS**. All ten episode artifact sets were complete;
  action parse rate was 1.0; agent axes, role cycling, targeted-`Give` mappings,
  broadcast accounting, and replay hashes passed; and model calls, provider
  requests, and transport errors remained zero. The three short episodes were
  clean environment terminations rather than evaluator or API failures.
- The matrix executed 1,888/2,000 requested environment turns and 6,194/6,400
  requested scripted agent-turns. It emitted 6,194 messages, delivered 19,544
  copies totaling 614,408 bytes, and saved 1,888 states.
- Step-weighted mean full-turn time was 0.263203 seconds over the matrix.
  Population means rose from 0.151301 seconds at `N=1` to 0.437702 seconds at
  `N=6`. Each episode had a 10.1--12.1 second maximum consistent with
  cold-start or compilation overhead, although the raw summary does not record
  its turn index or cause. After removing exactly one slowest turn per episode,
  population means ranged from 0.076475 to 0.379037 seconds.
- The cumulative scripted-worker action phase was 0.326201 seconds, or 0.172776
  ms per actual environment turn. Because no inference occurred, this worker
  timing is not an LLM latency or provider-concurrency measurement.
- Published the strict/structural decision, methods, per-population and
  per-seed timings, death causes, communication totals, limitations, and
  provenance in `Results/e0_1_agent_compatibility_200_v1.{json,md}` and
  `e0_1_report.{tex,pdf}`.
- Decision: retain the strict horizon failure and the structural pass as
  separate findings. Do not reinterpret passive survival as infrastructure
  compatibility. If an exact 200-turn infrastructure stress test is desired,
  preregister a distinct E0.2 using explicit invulnerability or a deterministic
  survival actor.

## 2026-07-23 — E0.1 descriptive performance-statistics addendum

- Added the evaluator's standard task-performance fields to the published E0.1
  Markdown, JSON, LaTeX, and compiled PDF: mean per-agent episode return,
  average episode length, team achievement percentage and count, player level,
  exact seed returns, full-horizon frequency, and normalized mortality.
- Verified from every terminal episode artifact that all
  `Achievements/*` values, team normalized reward, normal/shared achievement
  counts, player level, and monster-kill counts were zero.
- Recorded two-seed mean returns of -0.4500, -0.5250, -0.0333, -0.8625, and
  -0.6833 for `N={1,2,3,4,6}`, respectively. These are descriptive
  `Noop` outcomes. They are not population rankings because the sample has only
  two seeds, three episodes ended early, and population changes alter mortality
  exposure and world dynamics.
- Documented reward semantics: each episode value is the mean cumulative return
  per physical agent; per-step reward adds achievement reward to 0.1 times
  health change. With no achievements, the observed nonpositive returns reflect
  net health change rather than task accomplishment.
- Added multi-agent alive-step exposure: 95.88%, 100.00%, 78.73%, and 86.48%
  for `N={2,3,4,6}`; the solo wrapper does not record this key. All recorded
  multi-agent coordination and item-give attempt counts were zero; the solo
  wrapper omits those fields.
- Explicitly excluded the evaluator's generic 100% `success_rate`: it is
  computed as the fraction of episodes with `done=true` and therefore measures
  bookkeeping completion, not task success.

## 2026-07-23 — E1 launch readiness and campaign authorization

- The user directed the implementation to proceed autonomously through the
  staged population, six-agent recruitment, and Alem-transfer experiments
  without additional approval pauses. This supersedes the earlier
  cost-authorization hold; promotion gates and recorded campaign ceilings still
  apply.
- The E0.1 strict 200-tick failure was not relabeled. It is explicitly waived
  only as a blocker to behavioral evaluation because all three short episodes
  were valid natural all-agent-death outcomes and every structural gate passed.
  E1 treats early death and actual agent-tick exposure as measured outcomes.
- Added the `source_scaling_200` profile for the unchanged Source path:
  homogeneous `gpt-5.4-nano`, high provider reasoning, `robust_all`,
  `specific_collaborative`, scratchpad and communication enabled, free
  coordination, baseline broadcast topology, seeds 13100--13102, 200 requested
  ticks, one episode worker, and no debriefs. The profile validates at
  `N={1,2,3,4,6}`.
- Added behavior-preserving `alem-dice-performance-v1` episode metrics. They
  distinguish reward-weighted Base/Coordination/Total scores, unique team
  achievement first-unlocks, summed per-agent first-unlocks, cumulative raw
  environment event counters, and completed/submitted/alive/actionable
  agent-turn exposure. Achievement counts are not represented as repeated world
  interactions.
- Added a strict E1 analysis tool that validates canonical artifacts and emits
  raw episode CSV, bootstrap JSON, Markdown, a generated LaTeX table, and the
  three-panel performance-versus-agent-count figure.
- Added a dedicated population launcher with immutable per-count resolved
  configurations, population-and-agent-specific deterministic cache routes, a
  distinct one-call preflight route, an exclusive study lock, atomic control
  files, exact seed/population/topology/artifact completion checks, append-only
  attempt logs, bounded retry attempts, process-group cleanup, and resume
  validation.
- The nominal episode-call ceiling is 6,000 for E1a plus 3,600 for E1b. The
  campaign hard ceilings are 12,001 logical responses including preflight and
  15,000 observed provider attempts, leaving a bounded retry margin.
- Created the combined E0--E3 LaTeX report with completed E0/E0.1 evidence,
  detailed downstream methods, hypotheses, promotion rules, risk register, and
  audit requirements. Its initial 13-page PDF compiled successfully.
- Focused verification passed: eight metric/profile tests, Ruff, Python
  compilation, profile dry-run at all five populations, a provider-free real
  evaluator smoke, analysis-output smoke, launcher lock/config-resume smoke,
  and `git diff --check`.
- No provider request had been made at this checkpoint. The next action is the
  single GPT-5.4 nano high preflight at the committed source revision.

## 2026-07-23 — E1 recovered-transport-retry accounting correction

- Corrected the analyzer gate to match the frozen `max_retries=5` provider
  protocol. A retryable transport error followed by a final completed response
  is recovered evidence, not an API-failed episode.
- Added exact global and per-worker reconciliation:
  `provider attempts = completed logical responses + transport errors`. Error
  type counts, final response IDs/models/tokens, completion status, usage
  records, and the append-only attempt ledger must agree; incomplete responses,
  lost logical calls, exhausted failures, and unaccounted attempts still fail
  closed.
- New versioned evaluator records now persist per-call attempt counts, error
  counts/types, final provider status, stop/incomplete status, and per-worker
  typed-error maps. Frozen legacy artifacts are accepted only when their
  aggregate and worker evidence is exact. A typed legacy retry without a
  per-worker type map is accepted only when exactly one worker owns all errors,
  making type ownership exact by elimination.
- Pinned and checked the original campaign ceilings: 9,600 nominal episode
  decisions, 12,001 successful logical responses including preflight, and
  15,000 provider attempts.
- Read-only audit of the live tree accepted 11 completed cells. Ten recorded no
  retry. `N=4`, seed 13101 recorded 800 completed decisions, 801 provider
  attempts, one `APIConnectionError` on worker 2, zero incomplete responses,
  and no call loss. Including preflight, observed usage was 5,174 logical
  responses and 5,175 provider attempts. The analysis remains watermarked
  incomplete at 11/15 cells.
