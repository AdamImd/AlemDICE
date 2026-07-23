# AlemDICE experiment protocols

This document separates the faithful Alem baseline from the smaller paid API
validation run. Results are comparable only when they use the same source
revision, profile, model snapshot, and difficulty.

## Common baseline contract

Both OpenAI profiles use:

- `Alem-Coop-Symbolic`, three agents, and roles cycling through warrior,
  forager, and miner.
- Soft specialization enabled, shared reward disabled, and the
  `robust_all` agent with the `specific_collaborative` prompt.
- Visible chain-of-thought field, team broadcast communication, private
  scratchpad memory, and the upstream tagged-text action parser.
- Easy, Medium, and Hard coordination evaluated and reported separately.
- Deterministic world seeds beginning at `9999`.
- Local artifacts as the record of truth; W&B is disabled unless explicitly
  enabled in a derived profile.

The default hosted model is `gpt-5.6-luna` through the OpenAI Responses API with
provider reasoning effort `none`. The upstream visible reasoning/tagged response
remains normal generated text; no JSON schema or tool-call action contract is
used. Three agents make decisions concurrently within a tick. The matrix runner
allows three concurrent episodes, for at most nine in-flight requests.

## Profiles

| Profile | Purpose | Episodes | Step cap | Decision-call ceiling |
| --- | --- | ---: | ---: | ---: |
| `fake_smoke` | Free deterministic wiring check | 1 Easy | 3 | 0 |
| `openai_reduced` | Initial paid validation | 3 per difficulty | 200 | 5,400 |
| `upstream_main_full` | Current-main baseline protocol | 20 per difficulty | 10,000 | 1,800,000 |
| `team_leader_200` | Bodyless-leader topology pilot | 1 Easy per arm | 200 | 1,880 across 3 arms |
| `embodied_commander_30` | Luna wiring/qualitative gate | 1 Easy per arm | 30 | 189 across 2 arms |
| `embodied_commander_100` | Luna intermediate paired evaluation | 1 Easy per arm | 100 | 630 across 2 arms |
| `embodied_commander_200` | Luna exploratory paired study | 3 Easy per arm | 200 | 3,780 across 2 arms |

The ceilings count one decision for each of three agents at every allowed step.
They exclude transport retries. Episodes can terminate early, so actual totals
may be lower. The reduced profile disables debriefs. The full profile enables up
to 180 additional post-episode debrief calls (three agents × 60 episodes), for a
combined ceiling of 1,800,180 logical model calls before transport retries. The
full profile is deliberately not run by setup or tests.

`openai_reduced` runs seeds `9999`, `10000`, and `10001` on each difficulty. The
full profile runs seeds `9999` through `10018`. The reduced matrix is a pipeline
and qualitative-behavior check, not a statistically interchangeable substitute
for the 20-episode baseline.

## Bodyless team-leader pilot

`team_leader_200` keeps exactly three Alem players and compares three matched
topologies at Easy seed 9999: ordinary peer broadcast, a bodyless leader plus
peer broadcast, and the same leader with worker reports delivered only through
the leader. The leader has client index 3 but no observation/action/reward or
trajectory slot. It receives only the three rendered legal text views and
worker reports, replans at steps 0, 5, ..., 195, and assigns each worker before
the same tick's parallel worker calls.

All four clients are pinned to `gpt-5.4-2026-03-05` with high reasoning. The
profile deliberately omits temperature, top-p, and GPT-5.6-only explicit cache
breakpoints. Stable arm/role cache keys use automatic caching with 24-hour
retention. Easy non-specialist efficiency is explicitly 0.70. The launcher runs
arms sequentially, caps logical calls at 1,880 before transport retries, writes
an immutable study manifest, and reports performance against model tokens,
delivered bytes, latency, and estimated cost. One seed supports raw descriptive
contrasts only.

## Embodied squad-commander study

The embodied-commander profiles retain exactly three physical and logical
participants. Agent 0 remains the warrior and receives a separate serial
planning call before the same tick's three parallel action calls. The planning
client is a new client instance cloned from Agent 0's exact Luna request
configuration, with a planner-only cache-key suffix; no fourth client entry is
configured. The planner sees only Agent 0's legal long/short text observation,
public rules, its bounded private planner scratchpad, the previous accepted
plan, and validated prior-tick status reports.

`embodied_commander_broadcast` delivers the full accepted plan to all three
agents before action selection and keeps Source's one-tick peer broadcast.
`embodied_commander_star` uses identical plan content and timing but sends
worker statuses only to Agent 0; it is a later routing ablation, not one of the
initial two arms. Wingmen retain tactical autonomy for movement, prerequisites,
local execution, and immediate survival, but the treatment prompt reserves
objectives, assignments, and replanning for the commander. Plans are versioned,
reviewed every five ticks, leased for ten ticks, and may replan on validated
`BLOCKED`, `COMPLETE`, or `EMERGENCY` transitions. Invalid plans are rejected
atomically and never extend a lease.

The preregistered first comparison is unchanged `baseline` versus
`embodied_commander_broadcast`, both on dedicated development seeds beginning
at 12000. Stage 30 is a wiring and qualitative gate only. Stage 100 is a
one-seed intermediate descriptive comparison requested for the first evaluation;
it allows at most 630 logical calls across the two arms. Stage 200 is a three-seed
exploratory estimate only. None is a full paper result or a basis for significance
claims. The launcher records resolved base and per-arm config hashes, a
prompt-contract hash, Git/lock provenance, exact commands, cache keys, call
ceilings, and gates. The summarizer keeps unavailable semantic/manual metrics
explicit instead of silently treating them as zero. Status validity is reported
both conditionally over emitted messages and as valid-status coverage over every
eligible agent-turn, so silence cannot appear as perfect protocol compliance.

The named profiles are in `baselines/llm/config/experiment/`. The selected YAML
controls the model, generation settings, retry policy, episode concurrency,
episode count, step cap, rendering, and output root. A baseline team must use the
same model in all three `clients` entries; validation rejects heterogeneous model
IDs. Each role has a stable `prompt_cache_key` base identifying the model,
prompt-schema revision, and role. At client construction, the Responses adapter
round-robins that base across `BASE:traffic-N` keys. The reduced and full
profiles configure three traffic shards per role, matching their three concurrent
episode workers. The possible key set is therefore stable across processes and
runs; it never embeds a PID, run, episode, or agent ID. Both the base and shard
count are captured in the resolved run configuration, which is sufficient to
reconstruct every effective key. GPT-5.6 profiles use explicit 30-minute caching
with a breakpoint at the end of the current system prompt. That prefix is stable
for a given role and progressive-disclosure level, but it differs across agent
roles and expands when the team reaches a new dungeon level; dynamic
observation/history messages follow it. Cross-agent reuse through a reordered
shared prefix is a future A/B experiment, not part of this baseline. For an
intentional local model variant, change all three `model_id` values and update
all three cache-key bases, record the resolved configuration, and do not label
the result as the unchanged standard profile.

Optional Hard-difficulty communication, scratchpad, and visible-reasoning
ablations are in `baselines/llm/config/ablation/`. They are not part of either
standard matrix and force the run to Hard difficulty. For example:

```bash
./commands.sh openai-reduced --ablation hard_no_communication --dry-run
./commands.sh openai-reduced --ablation hard_no_communication
```

## Commands

Set up the Python 3.12 Mamba bootstrap and lock-resolved project environment:

```bash
./commands.sh setup
```

Setup creates or updates the `alem-dice` Mamba environment, uses its Python and
UV executable to synchronize the lock, and installs the runnable project into
`.venv`. All `commands.sh` subcommands select `.venv/bin/python` directly; Mamba
activation is unnecessary.

Then run the free checks before making hosted calls:

```bash
./commands.sh test
./commands.sh smoke
```

Run the reduced matrix:

```bash
export OPENAI_API_KEY="..."
./commands.sh openai-reduced --dry-run
./commands.sh openai-reduced
```

`--dry-run` validates the profile and prints output paths, concurrency, and call
ceilings without creating directories or calling OpenAI. Use it before every
costly configuration. A paid launch also refuses a dirty Git worktree or missing
lockfile, so commit the exact implementation first. The direct reduced-matrix
equivalent is:

```bash
./.venv/bin/python scripts/run_openai_matrix.py --profile openai_reduced
```

Run the embodied-commander protocol in stages:

```bash
export OPENAI_API_KEY="..."
./commands.sh commander-study --stage 30 --dry-run
./commands.sh commander-study --stage 30 --preflight
./commands.sh commander-study --stage 30

# One-seed intermediate comparison.
./commands.sh commander-study --stage 100 --dry-run
./commands.sh commander-study --stage 100 --preflight
./commands.sh commander-study --stage 100

# Run only after the 30-step automated gates pass and the traces receive
# semantic review.
./commands.sh commander-study --stage 200 --dry-run
./commands.sh commander-study --stage 200
```

Resume with the same stage and arm set, or summarize without API calls:

```bash
./commands.sh commander-study --stage 30 --resume outputs/alem_eval/RUN_NAME
./commands.sh commander-study --summarize outputs/alem_eval/RUN_NAME
```

After broadcast passes, append the star-routing ablation on a fresh run with
`--include-star`. Do not mix that exploratory third arm into the preregistered
two-arm contrast.

The canonical full protocol is intentionally explicit because of its potentially
large cost:

```bash
./commands.sh openai-full --dry-run
./commands.sh openai-full
```

Every launcher prints the resolved output directory before evaluation begins.
Operate on that directory with:

```bash
./commands.sh resume outputs/alem_eval/RUN_NAME
./commands.sh summarize outputs/alem_eval/RUN_NAME
./commands.sh visualize outputs/alem_eval/RUN_NAME
```

The visualize command prints every matching `*_debug.html` path and opens the
first one only when `xdg-open` and a graphical display are available. On a
headless machine, copy a printed self-contained HTML file to your workstation or
open it with your preferred browser.

Resume is episode-granular: completed episode records are skipped; an interrupted
episode restarts from its deterministic world seed. Never mix outputs from
different source revisions or resolved configurations in one run directory. The
matrix manifest stores a hash of the fully resolved profile and rejects a resume
after a profile/configuration mismatch. It also pins the source commit and UV
lock hash, so resume rejects a different implementation or dependency set.
The current paid-profile resume path checks `OPENAI_API_KEY`, a clean worktree,
the source commit, and the lockfile before entering its episode-skip loop. Those
requirements therefore still apply when every episode is already complete; use
`summarize` or `visualize` when no resume attempt is needed.

## Metrics

Report both metric tracks rather than silently choosing one. In each episode
JSON, the paper-facing normalized values are
`user_info["Team/reward_pct_of_max"]`,
`user_info["Team/normal_reward_pct_of_max"]`, and
`user_info["Team/coord_reward_pct_of_max"]`. Their aggregated local values are
under `user_info_means` in each difficulty's `summary_stats.json`.

The current repository/leaderboard headline is
`Team/achievement_pct`, the fraction of achievements unlocked by any team
member. Always report it with:

- `Team/coordination_achievement_pct`
- `Team/normal_achievement_pct`
- `action_parse_rate` (per episode and aggregated in `summary_stats.json`)
- episode length (`num_steps`), `done`, and `termination_reason`; provider length
  guards also set `early_stop_reason`
- per-agent and team reward
- deaths and survival
- cooperation events such as transfers, revivals, requests, handovers, and
  successful synchronized actions when present
- input, output, reasoning, and cached-token usage plus model-call latency and
  errors

`summary_stats.json` also records decision versus debrief call counts and
`termination_reason_counts`. Semantic action-format retries are included in the
decision call total. Usage from partial failed episodes remains included because
those calls were made, while reward/world metrics exclude failed episodes.
`provider_request_count`, `transport_error_count`, and error-type counts include
transport retries even when no response/usage object was returned.

When W&B is explicitly enabled, aggregate score keys use
`eval/{difficulty}/Team/...`; per-episode parse rate uses
`eval_ep/{difficulty}/action_parse_rate`. For current runs, the local summarizer
computes usage, action-parse rate, and terminal-cause counts across durable
attempt-ledger rows, including failed attempts that were later restarted. Reward
and world-state metrics use only the current valid completed episode JSON. Runs
created before the ledger existed fall back to their episode JSON. Every new
attempt records `termination_reason` as environment termination, environment
truncation, evaluator step cap, provider length guard, or error;
`early_stop_reason` adds detail for the provider length guard.

Do not average Easy, Medium, and Hard into one headline number. A low parse rate
means response-format failures may be limiting the achievement score.

## Run artifacts

Each run directory is self-describing and should include:

- the resolved configuration, Git revision, upstream revision, dependency-lock
  hash, model/request settings, and timestamps;
- the initial provenance fields in each per-difficulty `run_manifest.json` remain
  unchanged on resume; the manifest file itself is updated to append each
  invocation's source/lock/config hashes to `resume_history`, and each invocation
  saves a separate resolved configuration snapshot;
- episode JSON and CSV results plus aggregate summary statistics;
- append-only debug JSONL containing observations, prompt messages, raw model
  output, parsed fields, chosen actions, communication, scratchpad, response
  metadata, usage, latency, and normalized errors;
- episode JSON records `termination_reason` and complete logical-call usage,
  including semantic retries and post-episode debriefs;
- symbolic trajectories and exact pre-step JAX-state bundles where supported;
- self-contained debug HTML; profiles with `save_images: true` also produce
  first-episode GIF/video artifacts for each difficulty. The `fake_smoke`
  profile intentionally disables image capture.

Prompts and outputs can contain sensitive mission or model-generated text. Keep
run directories local unless they have been reviewed and scrubbed. API keys must
never be written to artifacts.

## Interpretation limits

- LLM results are stochastic and should retain all episode-level observations,
  not only aggregate means.
- The symbolic RL and text LLM interfaces operate in the same world but are
  separate evaluation tracks.
- Neither baseline profile evaluates large-scale collaboration, process failure,
  negotiated roles, or Byzantine behavior. Those are staged research additions
  described in [FuturePlans.md](FuturePlans.md).
