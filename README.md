# Seed Miner

Automated generation of small, self-contained data-analysis project seeds, using
preference-weighted sampling and a human review gate.

## Overview

A scheduled GitHub Actions workflow runs once each weekday. On every run the
system selects one `{domain, technique}` pair from a fixed matrix, asks the
Claude API to produce a small standalone analysis project for that pair, commits
the result to `seeds/`, and appends a record to an append-only log.

Each generated seed is reviewed manually and marked as promoted or rejected.
Those decisions feed back into the sampler: domains and techniques that produce
promoted seeds are sampled more often, while the sampler continues to explore
the rest of the matrix.

Seeds are produced by an automated process and committed under a dedicated bot
identity (`seed-miner-bot`). The generated code is not executed by the system;
it is written to the repository for later human review.

The intent is breadth. Most seeds are expected to be discarded. The value lies
in the small number of pairings where applying a technique to an unfamiliar
domain requires a genuine adaptation of that technique, which can then be
developed further in a separate repository.

## How it works

Each run of `runner/generate.py` performs the following steps:

1. Re-derive tag weights from the log if the last derivation is at least seven
   days old.
2. Score every valid matrix cell and sample one cell in proportion to its score.
3. Collect recent seed titles to discourage repetition.
4. Call the Claude API with the generation prompt for the selected cell.
5. Validate the response and write the returned files to `seeds/<date>_<slug>/`.
6. Append one record to `state/log.jsonl`.
7. Regenerate `LOG.md` and write the commit message to
   `state/last_commit_msg.txt`.

The workflow then commits and pushes the result.

## Repository layout

| Path | Description |
|------|-------------|
| `runner/generate.py` | Entry point: sample a cell, call the API, write the seed, update the log |
| `runner/weights.py` | Derives domain and technique weights from the log |
| `runner/prompt.py` | Generation prompt and prompt construction |
| `runner/store.py` | Shared paths and all filesystem access |
| `runner/review.py` | Command-line interface for promoting and rejecting seeds |
| `state/matrix.json` | Domain and technique lists, obvious pairings, excluded cells |
| `state/log.jsonl` | Append-only record of every generated seed |
| `state/weights.json` | Current tag weights, derived from the log |
| `seeds/` | One directory per generated seed |
| `LOG.md` | Generated review surface |
| `.github/workflows/generate.yml` | Scheduled workflow |

## Sampling and weighting

The matrix contains 12 domains and 8 techniques. Seven combinations are excluded
as unworkable (text-analysis techniques applied to domains with no meaningful
free text), leaving 89 valid cells.

**Per-run sampling.** Each valid cell is scored as:

```
score = w_domain × w_technique × novelty_bonus × obviousness_penalty
```

- `novelty_bonus` favours domains and techniques absent from the last ten runs.
- `obviousness_penalty` reduces the weight of the single most conventional
  technique for each domain, which is declared in `state/matrix.json`.

A cell is then drawn at random in proportion to its score rather than by taking
the maximum, so that exploration continues.

**Weight derivation.** At most once a week, each domain and technique is scored
as an exponentially weighted, Laplace-smoothed promotion rate over the log.
Recent decisions dominate (decay 0.9 per week), unreviewed seeds count as weak
negatives after 14 days, unseen tags receive a neutral prior, and every score is
floored so that no tag is permanently excluded. See
[`runner/weights.py`](runner/weights.py) for the formula.

## Installation

Requires Python 3.12 or later.

```bash
pip install -r requirements.txt
```

The runner depends only on the Anthropic SDK. Packages that a generated seed may
import (pandas, numpy, scikit-learn, and similar) are not required to generate
seeds and should be installed only when running a seed.

## Usage

Exercise the full pipeline without an API key or any cost:

```bash
python runner/generate.py --dry-run
```

Perform a real run:

```bash
export ANTHROPIC_API_KEY=...
python runner/generate.py
```

Re-derive weights without generating a seed:

```bash
python runner/weights.py
```

### Configuration

All settings are optional environment variables.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | API credential. Required for real runs. |
| `SEED_MINER_MODEL` | `claude-haiku-4-5` | Model used for generation |
| `SEED_MINER_MAX_TOKENS` | `8000` | Upper bound on output tokens per run |
| `SEED_MINER_DATE` | current UTC date | Overrides the run date, for testing |
| `SEED_MINER_SEED` | — | Seeds the random number generator, for reproducible sampling |

## Reviewing seeds

Review is a required manual step; without it the weights receive no signal.
`LOG.md` lists every seed grouped by week, together with the model's own
assessment of whether the seed is substantive.

```bash
python runner/review.py list
python runner/review.py promote 2026-07-17_disc-golf-network --note "reason"
python runner/review.py reject  2026-07-18_thrift-pricing-anomaly --note "reason"
```

Promoting or rejecting updates `state/log.jsonl`, regenerates `LOG.md`, and
re-derives `state/weights.json`. `LOG.md` is generated and should not be edited
directly.

A promotion records a judgement; it does not itself produce finished work. A
promoted seed is intended to be developed further in its own repository.

## Scheduled execution

The workflow in `.github/workflows/generate.yml` runs at 15:00 UTC, Monday to
Friday, and can also be triggered manually. It requires one repository secret:

| Secret | Purpose |
|--------|---------|
| `ANTHROPIC_API_KEY` | Passed to the generation step as an environment variable |

The repository's Actions setting for workflow permissions must allow read and
write access so that the workflow can push its commit.

To trigger a run manually:

```bash
gh workflow run generate.yml
```

## Security

**Credential handling.** The API key is stored as a GitHub Actions secret and
injected as an environment variable into the generation step alone. It is never
written to the repository or to the log, and `.env` is excluded by
`.gitignore`. GitHub masks secret values in workflow output.

**Generated code is untrusted input.** Seeds are model output committed without
review. The runner writes them to disk and never executes them, and the workflow
does not run them either. Read a seed before running it, and prefer an isolated
environment such as a dedicated virtual environment or container.

**Workflow trigger surface.** The workflow responds only to `schedule` and
`workflow_dispatch`. It has no `pull_request` or comment trigger, so a fork or an
external contributor cannot cause it to execute with repository credentials.
`workflow_dispatch` requires write access to the repository.

**Token scope.** The workflow requests `contents: write`, the minimum required
to commit the generated seed. No other permission is granted.

**Filesystem confinement.** File paths returned by the model are normalised,
stripped of leading separators, and rejected if they resolve outside the seed
directory. All paths are validated before any file is written, so a seed
containing an invalid path is rejected without leaving partial output on disk.

**Model input.** Titles from earlier seeds are included in later prompts to
discourage repetition. That text is whitespace-collapsed and truncated so it
cannot introduce structure into the prompt. The model is given no tools and no
repository access; its output is treated purely as data.

**Failure behaviour.** Malformed or truncated model output aborts the run before
any file or log entry is written, and the workflow's commit step does not run
after a failed generation step. A failed run therefore commits nothing.

## Cost

At the default model and token ceiling a run costs on the order of a few cents,
which is under one US dollar per month at a weekday cadence. Two controls are
recommended in the Anthropic Console: a prepaid balance and a low monthly spend
limit. Automatic top-up should remain disabled so that a scheduling or retry
fault stops at the limit rather than continuing to bill.

## Limitations

The system searches a deliberately wide space and most output is expected to be
discarded. A realistic expectation is roughly one seed worth developing out of
every 30 to 50 generated.

The weighting is a sampling heuristic, not a trained model. With few reviewed
seeds the signal is weak, and the smoothing prior keeps early weights close to
neutral by design.

The approach depends on the review step being carried out. If seeds are never
reviewed, every tag decays toward its floor, the log accumulates unreviewed
entries, and the output has no filter applied to it.
