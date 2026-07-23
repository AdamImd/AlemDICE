# E2b v2 failed hosted diagnostic

> **Status: failed and permanently non-promotable.** This is partial
> diagnostic evidence, not a between-method efficacy result. Cancelled,
> failed, zero-call, and poison-affected cells are not imputed.

## Bound source

- Preserved root: `/home/adam/Desktop/AlemDICE/outputs/recruitment_llm/e2b_luna_screen_v2`
- Tree SHA-256: `406b4f09c1d5ae8e532bfaee0c9aa591324eb0603f0063635b54997c1ed374e0`
- Full manifest SHA-256: `14c90f60f5348472f00421842e552d623b282fa1a13ad08b4563d8a72d84fbd0`
- Ledger SHA-256: `950f5a7ce4ec3386bba2853e6409a5310496e5d5c69bc061a59e96cf3bcd4f04`
- Source commit: `345da487a0217776b89b655e693dadcbfb153807`

## Manifest accounting

- Matrix: 24 cells.
- Manifest-completed: 3; failed: 2; cancelled: 19.
- Budget: 72 logical calls, 144 reserved provider attempts, 706,642 reserved tokens.
- Poisoned: `True` (`cell_failure:RuntimeError`).

## Failure mechanism

- 9/72 calls returned `incomplete/max_output_tokens`.
- All 9 truncations used exactly 1024 output tokens entirely as reasoning and returned a blank visible completion.
- `semantic.sender_missing`: 2 calls.
- Validation codes: `{"abstain": 31, "provider.status_not_completed": 9, "semantic.sender_missing": 2, "valid": 30}`.

## Evidence classification

- `cancelled_no_evidence`: 19
- `eligible_partial_diagnostic`: 1
- `excluded_poison_affected_completed`: 1
- `excluded_zero_call_poisoned`: 1
- `failed_diagnostic_only`: 2

Only `eligible_partial_diagnostic` cells are shown below. They remain
descriptive because the campaign failed and the method matrix is incomplete.

| Seed | Family | Method | Calls | Coverage | Reward |
|---:|---|---|---:|---:|---:|
| 22000 | single_complementary | open_volunteer | 18 | 1.000 | 100 |

A poison-affected completed artifact is retained only as an excluded
observation. It is not included in the table or any aggregate:

- `open_volunteer/two_disjoint/seed-22000`: 22 calls before/around poison, 12 poison-budget abstentions, coverage 0.500, reward 100 (excluded).

## Consequence for the next protocol

The one-cell canary tested only Open Volunteer and therefore could not
detect the Mutual Nomination failure mode. The successor must use a fresh
root, increase the high-reasoning output allowance prospectively, and
require every Open/Mutual × single/two-disjoint canary cell to pass before
the remaining matrix can dispatch.
