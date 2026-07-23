# E2b preregistration: six-agent LLM recruitment screen

Status: **the v2 full stage and v3 canary both failed and are permanently
non-promotable; the prospective Open-only v4 confirmation is implemented and
has not made a hosted request**. Amendments A1--A3 were frozen before any E2b
model response. A4 was frozen from the preserved v2 failure before v3, and A5
is frozen from the preserved v3 failure before any v4 response.

## Question and scope

E2b asks whether six independently prompted language-model agents can operate
the already validated TFP1 formation runtime under a bounded, delayed public
ledger. It is a mechanism screen, not an Alem task-efficacy result. The
original screen had two arms:

1. Open Volunteer with Public Sweep; and
2. Mutual Nomination with Public Sweep.

Amendment A5 retires Mutual Nomination after its second failed mechanism
screen. The current confirmatory estimand contains Open Volunteer only.

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

## Sequential amendment A3 — 2026-07-23, before hosted E2b execution

No hosted or network-backed E2b request had been made when this amendment was
recorded. A second independent adversarial review found five remaining
fail-open recovery cases. The following changes supersede only A2's local
durability, managed-path, and response-envelope details; the frozen prompts,
agents, selector rules, scenarios, seeds, horizon, outcomes, and cost ceilings
are unchanged:

- The E2b Responses configuration explicitly enables a strict response
  envelope. A response without a nonblank returned model, without a usage
  object, or without exact nonnegative integer `input_tokens` and
  `output_tokens` is rejected before it can create a call record. Boolean,
  string, floating-point, and null values are not coerced. Optional detailed
  cache/reasoning counters remain zero when the provider omits the optional
  detail, but any present counter must also be an exact nonnegative integer.
  This switch defaults off in the shared adapter, independently of exact text
  preservation, so non-E2b callers retain their previous envelope behavior.
- Reservation ledger v2 durably writes one exclusive, `fsync`ed anchor for
  every reservation and resolution record. Each anchor binds the launch,
  record hash, sequence, and previous anchor. Manifests bind an exact ledger
  prefix by record count, prefix hash, record-chain head, anchor count, and
  anchor-chain head. A valid-prefix truncation, including truncation of the
  completed 18-call canary ledger, therefore cannot be interpreted as a fresh
  pending call while its durable high-water anchors remain.
- Pre-existing cell files are never silently downgraded to pending. A partial
  artifact triple, a complete but invalid triple, an unexpected managed file,
  or any cell artifact with an empty ledger is a hard pre-dispatch failure.
  This also makes a crash between an artifact write and its validated marker
  conservative: the operator must audit it offline rather than repeat a
  possibly billed request.
- The output root is lexical-canonical and realpath-contained. Every managed
  read/write uses no-follow opens, regular-file checks, and link-count one;
  every path component and artifact directory is checked against symlinks.
  Parent aliases, root aliases, hardlinked ledgers/locks/artifacts, and two
  distinct locked roots sharing an artifact are rejected. Every newly created
  ancestor, the new directory itself, and its parent are `fsync`ed.
- Provider overage, duplicate/unknown reservation state, or a reservation or
  resolution durability failure permanently poisons the in-process campaign.
  New reserves return without dispatch, queued cells are cancelled where
  possible, and already in-flight calls may only finish their durable
  resolution. The poison and reason are written to the bound manifest; a
  poisoned or failed manifest cannot later resume and reset the stop.
- The campaign manifest, protocol/config binding, prompt-cache key, and
  default output identity advance to v3; the anchored ledger advances to v2.
  Earlier manifests, ledgers, markers, and outputs do not satisfy the new
  binding and cannot silently resume.

The adversarial provider-free regression set covers missing/non-exact usage,
missing returned models, malformed-envelope non-promotion, 18-call valid-prefix
rollback, empty-ledger artifacts, partial and invalid triples, symlink,
hardlink and parent aliases, independent-root artifact sharing, per-ancestor
directory durability, sticky poison with in-flight resolution, resolution
callback failure, poisoned-manifest resume, and ordinary compatibility.
A fresh offline fake-client canary-to-full lifecycle then completed all 24
cells with three explicit cell workers: 464 logical calls, 928 reserved
provider attempts, 4,694,278 reserved tokens, 928 ledger records and 928
anchors, with zero unresolved reservations, zero overages, and no poison. A
second full resume constructed no client and left the 928-record anchored
checkpoint unchanged. These are recovery/orchestration results only; they are
not E2b model-behavior evidence.

## Sequential amendment A4 — 2026-07-23, after failed v2 and before v3

The first hosted attempt used the explicit
`outputs/recruitment_llm/e2b_luna_screen_v2` root at source commit
`345da487a0217776b89b655e693dadcbfb153807`. Its one-cell Open
Volunteer/`single_complementary` canary passed, but the promoted full stage
failed and poisoned itself. The preserved full manifest partitions the frozen
24-cell matrix into 3 manifest-completed, 2 failed, and 19 cancelled cells,
with 72 logical calls, 144 reserved provider attempts, and 706,642 reserved
tokens. One of the three manifest-completed cells made zero calls after poison
and another contains poison-budget abstentions; neither is eligible evidence.
The only unpoisoned partial outcome is the original Open/single canary. The
failed matrix cannot estimate a method effect and must never resume.

The read-only diagnostic is bound and rendered at
`Results/e2b_v2_failed_diagnostic_v1.{json,md}`. Its root tree SHA-256 is
`406b4f09c1d5ae8e532bfaee0c9aa591324eb0603f0063635b54997c1ed374e0`,
the full-manifest SHA-256 is
`14c90f60f5348472f00421842e552d623b282fa1a13ad08b4563d8a72d84fbd0`,
and the ledger SHA-256 is
`950f5a7ce4ec3386bba2853e6409a5310496e5d5c69bc061a59e96cf3bcd4f04`.
The diagnostic reproduces the causal precondition for this amendment:

- 9 of 72 archived calls returned `incomplete/max_output_tokens`;
- all nine used exactly 1,024 output tokens, all 1,024 were reasoning tokens,
  and the visible completion was blank;
- all nine occurred in the two attempted Mutual Nomination cells (4 in
  `single_complementary`, 5 in `two_disjoint`);
- two additional Mutual completions were canonical nominations whose sender
  was absent from its own roster and therefore failed
  `semantic.sender_missing`; and
- Open/`two_disjoint` reached 0.5 oracle-allocation coverage with 22 calls but
  also contains 12 poison-budget abstentions, so it is retained only as an
  explicitly excluded observation.

The deterministic read-only report command is:

```bash
uv run --extra baselines-llm --python 3.12 \
  python scripts/summarize_recruitment_llm_failure.py \
  outputs/recruitment_llm/e2b_luna_screen_v2 \
  --json-output Results/e2b_v2_failed_diagnostic_v1.json \
  --markdown-output Results/e2b_v2_failed_diagnostic_v1.md
```

The summarizer refuses to write inside the preserved campaign root.

This is right-censoring at the old generation bound. Completed calls already
reached 1,020 output tokens, so the data cannot establish that a 2,048-token
cap would clear the high-reasoning tail. The v3 cap is prospectively fixed at
4,096 output tokens, four times the observed censoring boundary. Because the
reservation is input dominated, this raises maximum exposure per provider
attempt only from 18,048 to 21,120 tokens (17.0%). The 256-byte visible TFP1
grammar and all parsing rules remain unchanged. Any v3
`incomplete/max_output_tokens` response still fails its cell and its canary;
the runner never adapts the cap after observing a v3 response.

The v2 canary tested only Open Volunteer and therefore could not detect the
method-specific failure. V3 replaces it with four exact cells, all at seed
22000:

1. Open Volunteer × `single_complementary`;
2. Mutual Nomination × `single_complementary`;
3. Open Volunteer × `two_disjoint`; and
4. Mutual Nomination × `two_disjoint`.

Promotion requires every one of the four cells to be comprehensively valid
and independently satisfy all prior rate/integrity gates, contain no
max-output truncation or budget exhaustion, and form at least one truly
feasible team. Requested and resolved model identities must agree across all
four. A missing, invalid, failed, poison-affected, zero-call, or nonforming
cell makes the aggregate canary fail. Full-stage client construction remains
forbidden until the stored four-cell gate is canonically identical to a pure
recomputation.

The v3 output root remains
`outputs/recruitment_llm/e2b_luna_screen_v3`; config/source binding makes the
v2 root incompatible even if an operator supplies `--resume`. The four-cell
canary's maximum promotable projection is 288 initial plus 72 repair calls,
720 provider reservations, and 15,206,400 token reservations. The unchanged
full logical/provider caps and revised token cap are specified below.
Provider-free regressions and a fresh fake lifecycle cover the exact
four-cell gate. The fake canary and all 24 full cells completed with three
full-stage workers using 440 logical calls, 880 provider reservations,
7,157,450 reserved tokens, 880 anchored ledger events, no unresolved
reservation, no overage, and no poison. A second full resume constructed no
client and left the checkpoint unchanged. No hosted v3 call was made.

## Sequential amendment A5 — 2026-07-23, after failed v3 and before v4

The v3 four-cell canary was launched from commit
`ddcbb92fb0ff0376216bf0aa3da84e246945bad7` into the explicit
`outputs/recruitment_llm/e2b_luna_screen_v3` root. It failed and poisoned
itself after 40 logical calls, 80 provider-attempt reservations, and 614,968
reserved tokens. Its manifest partitions the four canary cells into one
manifest-completed Open cell, two failed cells, and one cancelled cell. The
root is preserved and automatic resume is forbidden.

The deterministic read-only evidence is stored in
`Results/e2b_v3_failed_diagnostic_v1.{json,md}`. The root tree, canary
manifest, and ledger SHA-256 values are, respectively,
`d88e134f203f11363aadbce502bf9c9b22682c660046465870477c887fb70899`,
`fd565c10968300d81ec60876ecf6a6cf0ba4556b3934abda0b97e6c06f84baad`,
and
`f4275aeec46bb2a2338cf9f5b3ad7ea36b4fc1499787eb3817804bfac6bc77e6`.
The same summarizer command used in A4 accepts this canary-only failed root
and refuses to write its report inside the preserved campaign.

V3 removes the observed v2 output-censoring failure within this canary; it
does not prove that every future call lies below the cap. All 40 archived
calls completed with zero transport error and zero max-output truncation at
the fixed 4,096 allowance. Open Volunteer/single-complementary again passed,
using 18 calls to lock the truly feasible `[0,5]` roster for full coverage and
reward 100.
Open/two-disjoint made zero calls because the shared campaign had already
been poisoned; its 12 poison-budget abstentions contain no Open behavioral
evidence. Mutual/two-disjoint was cancelled and contains no evidence.

The attempted Mutual/single-complementary cell is a negative mechanism
screen rather than missing data. Its 22 calls all completed without transport
or truncation failure, but five nominations failed
`semantic.sender_missing`:

- sender 2 nominated `MEMBERS=0,1` in round 0, and its one repair repeated the
  identical self-omitting completion;
- sender 3 also nominated `MEMBERS=0,1`;
- sender 5 nominated `MEMBERS=0,2`; and
- sender 1 later nominated `MEMBERS=0,2`.

The cell ultimately locked roster `[0,2]`, which was truly infeasible, and
recorded zero oracle-allocation coverage and zero reward. Together with the
v2 sender-omission signal, this is sufficient to stop repeatedly tuning the
frozen Mutual arm. It is not a between-method efficacy estimate because only
one Open cell produced uncontaminated behavior. Adding an explicit
self-inclusion instruction or repair would change the Mutual mechanism and
must be preregistered separately as exploratory work.

V4 is therefore a fresh Open Volunteer confirmation, not a prompt repair and
not a post-hoc relabeling of the failed v3 matrix. It keeps the exact model,
high reasoning, 4,096-token output allowance, TFP1 grammar, public-sweep
policy, and truth-free `joint_exact_allocation` selector. Seeds and families
remain unchanged. Removing Mutual yields 12 episodes:
\(3\text{ seeds}\times4\text{ families}\times1\text{ method}\).

The v4 identity is
`e2b-v4-open-joint-confirmation-two-cell-canary`, with campaign schema v4,
canary-gate schema v4, prompt-cache prefix `alem-e2b-v4`, and default root
`outputs/recruitment_llm/e2b_luna_screen_v4`. The canary consists of exactly
Open/single-complementary and Open/two-disjoint at seed 22000. Both must pass
all prior envelope, replay, transport, invalid-call, repair, model-binding,
budget, no-truncation, and true-feasible-formation gates. Only then may the
remaining ten Open cells run in parallel.

At 25% repair allowance, the two-cell canary projects 144 initial plus 36
repair calls, 360 provider reservations, and 7,603,200 reserved tokens. The
12-cell launch projects 864 initial plus 216 repair calls, 2,160 provider
reservations, and 45,619,200 reserved tokens. These become the v4 hard caps:
1,080 logical calls, 2,160 provider attempts, and 45,619,200 tokens.

A provider-free end-to-end lifecycle passed both v4 canary cells, completed
all 12 cells with three full-stage workers, and then resumed without
constructing a client. It used 272 logical calls, 544 provider reservations,
4,555,934 reserved tokens, 544 ledger records/anchors, zero unresolved
reservation, zero overage, and no poison. The repeated resume preserved the
exact checkpoint. This validates orchestration only; no hosted v4 call has
been made.

## Frozen matrix

- agents: exactly 6;
- model: OpenAI Responses adapter, `gpt-5.6-luna`;
- reasoning effort: `high`;
- maximum provider output: 4,096 tokens, including hidden reasoning;
- seeds: `22000`, `22001`, `22002`;
- scenario families: `single_complementary`, `two_disjoint`,
  `scarce_capability`, and `oversubscribed`;
- acting rounds: 12, indexed 0--11;
- delivery-only drain: nominally round 12, or the next consecutive round after
  a preregistered event stop;
- method: Open Volunteer/Public Sweep with
  `selector=joint_exact_allocation`;
- episodes: \(3\times4\times1=12\);
- outer cell workers: 1 by default, or an explicit bounded value with
  `--parallel-cells --workers N`; and
- within-round workers: up to 6, one client call per eligible agent.

All eligible agents in a round are prompted concurrently from one immutable
projection captured after that round's delayed deliveries. Valid records are
then submitted in ascending agent-ID order. Thus wall-clock completion order
cannot alter public state.

The launch is staged within the frozen 12-cell v4 estimand:

1. canonical two-cell canary: seed 22000, Open Volunteer, and both
   `single_complementary` and `two_disjoint`;
2. promotion only if every cell passes every integrity/rate gate, forms at
   least one truly feasible team, has no max-output truncation, at most 25%
   invalid model calls, at most 5% transport errors, at most one repair per
   four initial calls, and no budget exhaustion; and
3. full stage: resume the same output root and complete the other 10 cells.

The full stage refuses to create a client unless both canary markers,
artifacts, debug shards, source/config hashes, model bindings, and the
aggregate promotion gate still agree.

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
UTF-8 bytes. V4 Open Volunteer permits `APPLY`, `ACCEPT`, and `LOCK`. The
existing typed TFP1 parser enforces field order, field types, sorted unique
member IDs, integer bounds, and the byte ceiling. The runtime retains Mutual
records for historical replay, but the v4 runner cannot dispatch a Mutual
cell.

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

- initial logical model calls: 864;
- semantic-repair calls: 864;
- maximum logical calls: 1,728;
- maximum provider attempts: 3,456;
- maximum recorded successful-call usage: 29,417,472 input plus 7,077,888
  output tokens, or 36,495,360 total; and
- maximum provider-attempt exposure, if every transport retry were also
  billable at the full allowance: 58,834,944 input plus 14,155,776 output
  tokens, or 72,990,720 total.

The input ceiling is a safety cap, not an expected bill: normal prompts are
far shorter than 16,000 bytes, locked agents cease being eligible, abstentions
do not repair, and valid first completions do not repair.

The executable hard caps are deliberately below those theoretical repair
ceilings:

- 1,080 logical calls;
- 2,160 provider-attempt reservations; and
- 45,619,200 provider-attempt token reservations, charging one token per
  prompt byte, 1,024 framing tokens, and the full 4,096-token output allowance
  for both possible transport attempts.

The full-matrix launch projection includes each cell's maximum promotable 25%
semantic-repair rate: 864 initial plus 216 repair calls, 2,160 provider
reservations, and 45,619,200 token reservations. Before constructing any
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

# Continue only after outputs/recruitment_llm/e2b_luna_screen_v4/canary_gate.json says pass.
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
