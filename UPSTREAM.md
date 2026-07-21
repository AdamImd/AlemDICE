# Upstream provenance

AlemDICE is a research fork of [Alem](https://github.com/alem-world/alem-env),
retaining Alem's MIT license, attribution, and Git history.

## Pinned starting point

| Field | Value |
| --- | --- |
| Upstream repository | `https://github.com/alem-world/alem-env.git` |
| Local remote | `upstream` |
| Upstream branch audited | `main` |
| Imported commit | `b1344e46cb2cd3e0ea7474ee1973712b5eb2fde1` |
| Commit date | 2026-07-17 |
| Alem public-release tag | `v0.1.0` (`07e0cbbe9fa7d2d07a5d4283f59f0d43275ce0c5`) |

The imported commit is 25 commits after `v0.1.0`. It contains fixes and additions
to the text wrapper, prompt builder, evaluation harness, examples, tests, and
packaging. The LLM baseline is excluded from the published Python package, so a
source fork is required to preserve and extend the experiment harness.

## Reproduction boundary

The `upstream_main_full` experiment recreates the **current-main Alem LLM
protocol** at the pinned commit: the same environment family, seed schedule,
three roles, prompt mode, tagged response format, coordination settings, and
metrics. AlemDICE changes the hosted-model transport to the OpenAI Responses API
and adds provenance and usage logging around it.

Consequently:

- Results from this fork are current-main reproductions, not bit-identical
  reproductions of the code published at `v0.1.0`.
- Hosted LLM sampling and provider infrastructure can remain nondeterministic
  even when the world seed and request seed are fixed.
- The OpenAI request wire format differs from the upstream Chat Completions
  client, while the model-visible tagged-text contract is kept compatible.
- A paper-release comparison must explicitly check out `v0.1.0`, install that
  revision's dependencies, and report that revision separately. Do not combine
  its scores with current-main scores.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the exact baseline and reduced protocols.

## Updating from Alem

Upstream changes are never pulled implicitly. To inspect a candidate update:

```bash
git fetch upstream
git log --oneline b1344e46..upstream/main
git diff --stat b1344e46..upstream/main
```

Import an update only on a dedicated branch. Record the new full commit hash
here, regenerate the UV lock, and rerun the environment, parser, and fixed-seed
regression tests before merging it. Preserve the upstream license and citation
when redistributing this fork.
