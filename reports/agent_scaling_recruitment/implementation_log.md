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
| E1 Source scaling | Deferred | Requires hosted-model cost authorization |
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
