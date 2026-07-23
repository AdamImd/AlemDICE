# E2b v3 failed hosted canary diagnostic

> **Status: failed and permanently non-promotable.** This is a
> mechanism-screen diagnostic, not a between-method efficacy estimate.
> Failed, cancelled, and poison-cascade cells are not imputed.

## Bound source

- Preserved root: `/home/adam/Desktop/AlemDICE/outputs/recruitment_llm/e2b_luna_screen_v3`
- Tree SHA-256: `d88e134f203f11363aadbce502bf9c9b22682c660046465870477c887fb70899`
- Canary manifest SHA-256: `fd565c10968300d81ec60876ecf6a6cf0ba4556b3934abda0b97e6c06f84baad`
- Ledger SHA-256: `f4275aeec46bb2a2338cf9f5b3ad7ea36b4fc1499787eb3817804bfac6bc77e6`
- Source commit: `ddcbb92fb0ff0376216bf0aa3da84e246945bad7`

## Manifest accounting

- Canary matrix: 4 cells.
- Manifest-completed: 1; failed: 2; cancelled: 1.
- Budget: 40 logical calls, 80 reserved provider attempts, 614,968 reserved tokens.
- Poisoned: `True` (`cell_failure:RuntimeError`).

## Failure mechanism

- All 40 archived calls completed; transport errors: 0; max-output truncations: 0.
- `semantic.sender_missing`: 5 calls; 1 semantic repair repeated the identical invalid completion.
- Validation codes: `{"abstain": 17, "semantic.sender_missing": 5, "valid": 18}`.
- Usage: 28,604 input, 19,574 output, and 18,754 reasoning tokens.

Mutual Nomination/single-complementary completed 22 calls without
truncation or transport failure, but five nominations omitted their own
sender. It then locked roster `[0, 2]`, which was truly infeasible, for
zero oracle-allocation coverage and zero reward. This is a negative
screen for the frozen Mutual mechanism.

| Round | Sender | Attempt | Invalid completion |
|---:|---:|---:|---|
| 0 | 2 | 0 | `TFP1|TYPE=NOMINATE|TASK=single.t0|MEMBERS=0,1` |
| 0 | 2 | 1 | `TFP1|TYPE=NOMINATE|TASK=single.t0|MEMBERS=0,1` |
| 0 | 3 | 0 | `TFP1|TYPE=NOMINATE|TASK=single.t0|MEMBERS=0,1` |
| 1 | 5 | 0 | `TFP1|TYPE=NOMINATE|TASK=single.t0|MEMBERS=0,2` |
| 2 | 1 | 0 | `TFP1|TYPE=NOMINATE|TASK=single.t0|MEMBERS=0,2` |

## Evidence classification

- `cancelled_no_evidence`: 1
- `eligible_partial_diagnostic`: 1
- `excluded_poison_cascade_zero_call`: 1
- `failed_mechanism_diagnostic`: 1

The only passing partial cell remains descriptive:

| Seed | Family | Method | Calls | Coverage | Reward |
|---:|---|---|---:|---:|---:|
| 22000 | single_complementary | open_volunteer | 18 | 1.000 | 100 |

Open Volunteer/two-disjoint made zero calls because another worker had
already poisoned the shared campaign budget; its 12 poison-budget
abstentions contain no Open-method behavioral evidence. Mutual/two-disjoint
was cancelled and likewise contains no evidence.

## Consequence for the next protocol

The successor is a fresh Open-only confirmation, not a repaired Mutual
arm. It retains the 4,096-token allowance and joint exact allocator,
requires Open/single and Open/two-disjoint to pass a two-cell canary,
then runs the remaining ten Open cells. Any self-inclusion prompt repair
for Mutual would define a separate exploratory mechanism and is outside
the v4 confirmatory estimand.
