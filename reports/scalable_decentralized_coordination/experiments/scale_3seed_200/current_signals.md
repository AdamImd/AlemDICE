# Current signals before the 3-seed, 200-step study

This document freezes the interpretation of the completed one-seed pilots before the larger study is inspected. These are mechanism signals, not efficacy claims.

## Signal 1: structured cohesion

The cohesion treatment asks each peer to communicate a typed status containing a shared task ID, responsibility, state, target, and tick. Delivered messages enter only that peer's bounded local ledger; there is no bodyless planner.

In the 30-step v2 seed, cohesion returned 10.667 with six team achievements and one successful coordination event. The concise free baseline returned 0.333 with one achievement and no coordination success. Cohesion made nine environment coordination attempts versus 15 for free, while action parsing remained 1.00. A targeted 14-step v3 run again showed a positive trajectory (return 9.0, two achievements, one coordination success).

The cost signal was negative: agents emitted a status every turn instead of once per five turns. Input tokens were 635,382 versus 600,820 for the v2 free baseline (+5.8%). Prompt-only throttling did not work.

## Signal 2: concise tags-only output

The original internal-thinking configuration repeatedly spent a 768-token output budget before producing visible action tags. Initial calls took approximately 62–75 seconds and could default to `Noop`. It was not devoid of task signal: before that incomplete run stopped, it parsed 29 of 36 actions and appears to have completed one three-agent coordinated action. Disabling internal and visible chain-of-thought and requesting only action/communication/scratchpad tags produced action parse rate 1.00 in every completed v2/v3 arm. Early call latency fell to roughly 12–14 seconds, although full-history team ticks later plateaued near 30 seconds.

This is a clear reliability/throughput signal, but it has not been evaluated across independent environment seeds or against an adequately budgeted internal-thinking arm.

## Signals that were not promoted

- Consensus generated explicit agreements and many commits but did not improve return, achievements, or environment coordination success over free communication.
- Prompt-only role auctions produced invalid awards and orphan accepts. Rotating epoch IDs recovered one valid award but reduced overall protocol adherence.
- The integrated prompt suffered component interference: status or commitment messages crowded out role allocation.

## Larger-study interpretation rule

The next study uses seeds 9999–10001 with 200-step caps. It compares:

1. `free_thinking`: ordinary communication with internal and visible reasoning.
2. `free_concise`: ordinary communication with internal/visible reasoning disabled and tags-only output.
3. `cohesion_concise`: structured cohesion with the same concise response mode as arm 2.

Every arm receives the same 2,048-token ceiling, so output budget is not changed in either contrast. Arm 1 versus arm 2 tests an operational response-mode bundle: native reasoning, visible chain-of-thought, and remembered chain-of-thought change together. It cannot attribute an effect to any one toggle. Arm 2 versus arm 3 tests the complete cohesion scaffold, not a single prompt field. Seeds are paired. Three episodes within each arm run in parallel. This is still a small pilot; claims require consistent paired direction, not merely a higher aggregate mean.
