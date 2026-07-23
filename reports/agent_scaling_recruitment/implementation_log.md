# Agent Scaling and Recruitment Implementation Log

This is the append-only human-readable tracker for
[`experiment_plan.md`](experiment_plan.md). Existing dated entries must not be
rewritten after results are inspected. Corrections are appended as new entries.
Machine-readable manifests and raw artifacts remain the source of truth.

## Milestone status

| Milestone | Status | Evidence |
| --- | --- | --- |
| Plan frozen | Complete | `experiment_plan.md`, 2026-07-22 |
| E0 harness | In progress | Pending implementation |
| E0 free compatibility run | Pending | `N={1,2,3,4,6}`, seeds 13000–13001 |
| E0 LaTeX report | Pending | `report.tex` and compiled PDF |
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
