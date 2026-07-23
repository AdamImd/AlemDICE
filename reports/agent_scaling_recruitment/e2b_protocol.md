# E2b preregistration: six-agent LLM recruitment screen

Status: **implementation and fake-client validation only; hosted run not yet
started**. This protocol is frozen before any E2b model response is obtained.

## Question and scope

E2b asks whether six independently prompted language-model agents can operate
the already validated TFP1 formation runtime under a bounded, delayed public
ledger. It is a mechanism screen, not an Alem task-efficacy result. The only
arms are:

1. Open Volunteer with Public Sweep; and
2. Mutual Nomination with Public Sweep.

The public-sweep policy was promoted by the provider-free E2a experiment.
Contract Net is not screened here because E2a found its scripted sweep arm had
identical reward to Open Volunteer but more control bytes and later locks.

The selector interface is frozen as a truth-free extension point. The initial
screen uses `selector=none`. The public-ledger exact/joint selector may be added
as a separately labelled arm only after E2d2 chooses and freezes it; it may
never receive private truth or oracle output.

## Sequential amendment A1 — 2026-07-23, before hosted E2b execution

No E2b hosted response had been obtained when this amendment was recorded.
After the original freeze, the provider-free E2d2 replication completed
12,000 episodes and promoted `joint_exact_allocation`: it achieved complete
task coverage and zero utility regret in its truthful-claim constructed
allocation tests. The canonical result is preserved in
`Results/e2d2_joint_allocation_v1.md`. This is evidence for deterministic
allocation from truthful public claims, not evidence of robustness to
misreporting or of embodied Alem efficacy.

This amendment supersedes only the original `selector=none` sentence:

- Open Volunteer/Public Sweep uses replicated `joint_exact_allocation`.
  Every peer can compute the same exact plan from public task cards, delivered
  `APPLY` records and their self-claimed capability/cost fields, and current
  public leases. The selector maximizes public task reward, then claimed
  roster utility, then minimizes claimed raw cost, then uses the canonical
  lexicographic assignment tie-break.
- Mutual Nomination/Public Sweep remains its native reciprocal-nomination
  rule, labelled `native_mutual_reciprocal`, and is the cost-sensitive
  comparator. It receives no joint roster advice.
- The joint selector is a deterministic pure function. It receives no private
  true capability, private true cost, pending control, terminal feasibility,
  or oracle output. Only each acting model receives its own private profile;
  true values outside public self-claims remain terminal-scorer-only.
- Every nonempty selected roster has the task's exact requested size. Joint
  enumeration assigns each eligible agent one task-or-idle label, preserving
  one-team-per-agent across simultaneous tasks. The directory independently
  rechecks delivered applications, claimed feasibility, public leases, exact
  sizes, and cross-task exclusivity before publishing the Open Volunteer plan.
- Agents still form the team through the original delayed `ACCEPT` and `LOCK`
  controls. The selector chooses no action and creates no extra model call.

This is a prospective mechanism substitution within the already frozen two
method cells, rather than a third arm. Seeds, families, methods, 24-cell
episode count, 12-round horizon, generation settings, staged canary, outcome
definitions, and every logical/provider/token ceiling below remain unchanged.
The source/config hashes and dry-run manifests now identify the selector used
by each method, so pre-amendment markers cannot silently resume.

## Sequential amendment A2 — 2026-07-23, before hosted E2b execution

No hosted E2b request had been made when this safety amendment was recorded.
An independent prelaunch review identified fail-closed accounting and replay
requirements that do not change an agent prompt, selector, scenario, outcome,
or behavioral estimand:

- Hosted launch now requires a nonblank `OPENAI_API_KEY`, a canonical Git
  `HEAD`, a clean status except the exact hashed allowlisted replay
  `Results/replays/nano_high_source_full_world.mp4`, and matching hashes for
  `uv.lock`, every E2b source, and this protocol. These checks happen before
  output creation or client construction and are repeated after acquiring the
  campaign lock.
- One nonblocking whole-output-root `flock` is held for the complete hosted
  invocation. Every dispatch first appends and `fsync`s a hash-chained durable
  reservation keyed by seed, family, method, round, agent, and semantic
  attempt. The append and its parent directory are durable before the request.
  A crash leaves an unresolved reservation; it remains spent and automatic
  resume stops rather than risking a duplicate billed request. Resolutions,
  actual usage, marker coverage, and the ledger chain must agree exactly.
- The conservative per-attempt input charge now includes 1,024 framing tokens
  in addition to one token per prompt byte. Actual returned input above prompt
  bytes plus that allowance, output above 1,024, attempts above two, or any
  malformed/noninteger usage fails closed. `Retry-After` is clamped to 30
  seconds.
- The E2b adapter alone preserves the provider completion byte-for-byte,
  including outer spaces and CR/LF. All other clients retain their previous
  stripping default. Exact raw bytes, byte length, hash, strict 256-byte TFP1
  reparse, repair input, normalized record, and response metadata are
  cross-checked from the compressed debug shard.
- A provider response is admissible only when it is `completed`, has no
  incomplete reason, has a nonempty response ID, and returns either the exact
  requested `gpt-5.6-luna` alias or that alias plus a valid `YYYY-MM-DD`
  snapshot suffix. The first accepted resolved model is stable throughout the
  canary; its exact value is bound into the promotion gate and every full-stage
  cell.
- Completion markers use schema v2. Validation recomputes cell/scenario/config
  identity, canonical paths, model binding, prompt and call records, all call
  and token counts, durable reservations, replay state/audit hashes,
  deterministic hashes, and analysis-only outcomes. The canary gate is a pure
  recomputation over the one expected canary marker, episode, and debug shard;
  every predicate must be true and the stored gate must be canonically equal
  to the recomputation.
- Cross-cell execution is sequential by default. Operators may explicitly
  enable bounded in-process concurrency with `--parallel-cells --workers N`.
  Cells have disjoint artifacts, while the campaign budget and durable ledger
  serialize reservations. Canary promotion itself remains a one-cell
  sequential stage.

These controls alter only provenance, cost accounting, and acceptance of
provider envelopes. They do not repair or normalize model text and do not
change the frozen behavioral comparison.

## Frozen matrix

- agents: exactly 6;
- model: OpenAI Responses adapter, `gpt-5.6-luna`;
- reasoning effort: `high`;
- seeds: `22000`, `22001`, `22002`;
- scenario families: `single_complementary`, `two_disjoint`,
  `scarce_capability`, and `oversubscribed`;
- acting rounds: 12, indexed 0--11;
- delivery-only drain: nominally round 12, or the next consecutive round after
  a preregistered event stop;
- methods: Open Volunteer/Public Sweep and Mutual Nomination/Public Sweep;
- episodes: \(3\times4\times2=24\);
- outer cell workers: 1 by default, or an explicit bounded value with
  `--parallel-cells --workers N`; and
- within-round workers: up to 6, one client call per eligible agent.

All eligible agents in a round are prompted concurrently from one immutable
projection captured after that round's delayed deliveries. Valid records are
then submitted in ascending agent-ID order. Thus wall-clock completion order
cannot alter public state.

The launch is staged without changing the 24-cell estimand:

1. canonical canary: seed 22000, `single_complementary`, Open
   Volunteer/Public Sweep;
2. promotion only if the canary passes every integrity gate, forms at least
   one truly feasible team, has at most 25% invalid model calls, at most 5%
   transport errors, at most one repair per four initial calls, and no budget
   exhaustion; and
3. full stage: resume the same output root and complete the other 23 cells.

The full stage refuses to create a client unless the canary marker, artifact,
debug shard, source/config hashes, and promotion gate all still agree.

## Information boundary

Every agent prompt may contain:

- all public task cards;
- the six public role labels;
- the delivered public task ledger and public team assignments;
- public selector advice, when a preregistered selector is enabled; and
- only that agent's private true capability vector and per-task costs.

It may not contain another agent's private capability/cost, pending
undelivered controls, any true-feasibility label, an oracle roster, oracle
reward, or oracle utility. Self-reported capabilities and costs become public
only after a valid `APPLY` record is delivered, as required by the Open
Volunteer protocol. True feasibility and oracle comparisons are computed in a
separate analysis-only phase after formation.

## Output grammar and failure policy

A response is exactly `ABSTAIN` or one canonical TFP1 record of at most 256
UTF-8 bytes. Open Volunteer permits `APPLY`, `ACCEPT`, and `LOCK`; Mutual
Nomination permits `NOMINATE` and `LOCK`. The existing typed TFP1 parser
enforces field order, field types, sorted unique member IDs, integer bounds,
and the byte ceiling.

One semantic repair is allowed after a malformed, oversized, schema-invalid,
method-invalid, or publicly preflighted transition-invalid completion. A
second failure becomes a safe abstention.
Transport retries remain inside the Responses adapter and are capped at one
retry beyond the initial provider attempt, with provider `Retry-After`
bounded at 30 seconds. Episode and campaign call budgets are reserved
atomically and durably before calls.

An episode stops requesting models after two consecutive acting rounds with
both (a) no accepted delivered public transition and (b) no valid current
submission. It immediately performs the next consecutive delivery drain and
records the requested horizon, executed rounds, and stop reason. At that point
there is no newly pending record; the remaining frozen horizon would be
silent, so this event rule reduces cost without inserting a synthetic action
or success. The same rule applies to every cell.

## Audit and resumption

The episode artifact contains:

- normalized parse/transition ledgers;
- semantic-repair and transport-attempt ledgers;
- input, output, reasoning, cache-read, and cache-write token ledgers;
- per-call and per-round wall latency;
- peak observed within-round call concurrency;
- call ceilings and actual usage;
- terminal directory state/audit-chain hashes; and
- the deterministic directory replay.

Each episode also has a compressed per-call JSONL debug shard containing the
complete structured prompt projection and messages, raw initial/repair
completion, repair feedback, normalized typed parse, provider response ID,
status, usage, and prompt/completion hashes. It contains no API key,
authorization header, SDK header, environment dump, or request credential.
Human-facing artifacts retain only bounded redacted excerpts for failures.

An episode is resumable only when its atomic completion marker agrees with the
Git/lock/config binding, protocol hash, every source-file hash, the episode
artifact hash, both compressed and decompressed debug-shard hashes, all
embedded per-call hashes, the reservation ledger, and the directory replay
hashes. A completed episode is never called again. An interrupted episode with
any reserved request but no valid marker stops automatic resume: the
reservation stays spent and requires explicit offline reconciliation.

## Pre-run ceilings

The dry-run command is:

```bash
uv run --extra baselines-llm --python 3.12 \
  python scripts/run_recruitment_llm_screen.py
```

It makes zero provider calls. Under the deliberately conservative assumption
that all six agents are eligible in all rounds, every initial output needs a
repair, every transport call needs its one retry, every permitted prompt byte
is one input token, and every response exhausts its output allowance:

- initial logical model calls: 1,728;
- semantic-repair calls: 1,728;
- maximum logical calls: 3,456;
- maximum provider attempts: 6,912;
- maximum recorded successful-call usage: 58,834,944 input plus 3,538,944
  output tokens, or 62,373,888 total; and
- maximum provider-attempt exposure, if every transport retry were also
  billable at the full allowance: 117,669,888 input plus 7,077,888 output
  tokens, or 124,747,776 total.

The input ceiling is a safety cap, not an expected bill: normal prompts are
far shorter than 16,000 bytes, locked agents cease being eligible, abstentions
do not repair, and valid first completions do not repair.

The executable hard caps are deliberately below those theoretical repair
ceilings:

- 2,160 logical calls;
- 4,320 provider-attempt reservations; and
- 77,967,360 provider-attempt token reservations, charging one token per
  prompt byte, 1,024 framing tokens, and the full 1,024-token output allowance
  for both possible transport attempts.

The full-matrix launch projection includes the canary's maximum promotable 25%
semantic-repair rate: 1,728 initial plus 432 repair calls, 4,320 provider
reservations, and 77,967,360 token reservations. Before constructing any
client, the runner rejects a stage when prior valid-marker usage plus this
repair-adjusted remainder projection exceeds any user-visible hard cap. Each
actual call then atomically reserves one logical call, both possible provider
attempts, and their token exposure; a conflict produces a safe budget
abstention and fails the campaign integrity gate.

The hosted canary and promoted full commands are:

```bash
uv run --extra baselines-llm --python 3.12 \
  python scripts/run_recruitment_llm_screen.py \
  --stage canary \
  --execute-hosted

# Continue only after outputs/recruitment_llm/e2b_luna_screen_v2/canary_gate.json says pass.
uv run --extra baselines-llm --python 3.12 \
  python scripts/run_recruitment_llm_screen.py \
  --stage full \
  --execute-hosted \
  --resume \
  --parallel-cells \
  --workers 3
```

Hosted execution remains disabled unless the operator deliberately adds
`--execute-hosted`. A canonical launch requires a clean committed worktree and
records its commit and source hashes. Changing a cap changes the config hash,
so a differently capped run cannot silently resume these markers.

## Screen outcomes and gates

Primary descriptive outcomes are normalized task reward, true-feasible locks,
oracle-allocation coverage, and lock round. Secondary outcomes are malformed
and repaired records, rejected transitions, control bytes, roster agreement,
tokens, cache use, provider attempts/errors, latency, and observed
concurrency.

Integrity requires:

- no prompt-boundary violation;
- no accepted cross-team ordinary delivery;
- exclusive one-task-per-agent leases;
- no record over 256 bytes entering the directory;
- no call beyond the per-episode or campaign ceiling;
- full episode, debug-shard, marker, and replay hash agreement; and
- exactly the frozen seed/family/method matrix.

An integrity failure invalidates the affected episode. A method may fail to
form teams or achieve reward without invalidating the screen; that is an
efficacy result.
