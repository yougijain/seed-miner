# Seed Miner 🌱

**This repository is an automated experiment, and it says so on purpose.**

Every weekday morning a GitHub Action asks Claude to generate one small,
self-contained data-analyst project — a "seed" — from an under-mined
domain × technique pairing. It logs the seed, commits it, and steers future
generation toward the domain/technique combinations a human (me) has promoted.

Most seeds die. That's the design. The point is the handful that don't: the
occasional pairing where applying a familiar technique to an unfamiliar domain
*forces a genuine modification* to the technique, and that modification turns out
to be interesting. Those get forked out into their own real repositories where
actual human work happens.

## What this is (and isn't)

- **Is:** a wide-net idea explorer with a preference-weighted (bandit-style)
  sampler and a human review gate.
- **Isn't:** machine learning in the training sense — there aren't enough
  labeled events for that. And it isn't a disguised commit-graph padder: the
  automation is stated openly here and in every generated seed's README, and the
  commits are made by a clearly-labeled bot identity.

The honest one-line description: *"It tracks which domain/technique combinations I
promote and biases generation toward those, while forcing continued exploration."*
That's it. A few dozen lines of arithmetic, no black box.

## How it works

```
   .github/workflows/generate.yml   cron: weekday mornings
                │
                ▼
   runner/generate.py
     1. weekly?  → runner/weights.py re-derives tag scores from the log
     2. sample a {domain, technique} cell   (weighted, not argmax → keeps exploring)
     3. fetch recent seeds to avoid repeats
     4. call Claude (Haiku 4.5) with the Section-5 generation prompt
     5. write seeds/<date>_<slug>/{main.py, README.md}
     6. append one line to state/log.jsonl
     7. regenerate LOG.md  and  state/last_commit_msg.txt
                │
                ▼
   git commit (as seed-miner-bot) + push
```

### The moving parts

| Path | Role |
|------|------|
| `runner/generate.py` | Main loop: sample → call Claude → write → log |
| `runner/weights.py`  | Re-derives tag scores from the log (the self-steering part) |
| `runner/prompt.py`   | The runtime generation prompt handed to the model |
| `runner/store.py`    | Shared paths + JSON/log/`LOG.md` I/O |
| `runner/review.py`   | Human review CLI: promote / reject a seed |
| `state/matrix.json`  | The domain × technique grid + obvious-pairing/skip metadata |
| `state/log.jsonl`    | Source of truth — one line per generated seed |
| `state/weights.json` | Current tag scores, re-derived weekly (derived from the log) |
| `seeds/`             | One dated, slugged folder per seed |
| `LOG.md`             | Human-readable weekly review surface (generated) |

## The self-steering part (in one paragraph)

Each run scores every valid matrix cell as
`w_domain × w_technique × novelty_bonus × non_obvious_bonus` and samples one cell
*proportional* to that score (not the maximum — exploration must continue).
Weekly, each domain and technique tag is rescored as an exponentially-weighted,
Laplace-smoothed promote rate: recent promotions dominate (decay ≈ 0.9/week),
seeds I never reviewed count as weak negatives after 14 days, and every tag is
floored so nothing is permanently frozen out. See [runner/weights.py](runner/weights.py).

## Reviewing (the non-negotiable part)

This whole thing is worthless if the weekly review doesn't happen. Skim
[LOG.md](LOG.md) — each seed carries the model's own honest `self:` assessment of
whether it has legs or is filler — then promote the good ones:

```bash
python runner/review.py list
python runner/review.py promote 2026-07-17_disc-golf-network --note "betweenness-on-throwing-lines angle is real"
python runner/review.py reject  2026-07-18_thrift-pricing-anomaly --note "just IQR on a toy dataset"
```

Promotion updates `state/log.jsonl`, regenerates `LOG.md`, and re-derives the
weights so the next run leans toward what you liked. **Promotion is a signal, not
the work** — the actual value comes from forking a promoted seed into its own
real repo and diverging from it by hand.

## Running it yourself

```bash
pip install -r requirements.txt

# Try the full pipeline with a stubbed seed — no API key, no cost:
python runner/generate.py --dry-run

# A real run (needs ANTHROPIC_API_KEY):
export ANTHROPIC_API_KEY=sk-ant-...
python runner/generate.py
```

Config via environment (all optional): `SEED_MINER_MODEL` (default
`claude-haiku-4-5`), `SEED_MINER_MAX_TOKENS` (default `8000`), `SEED_MINER_DATE`
(override "today", for testing), `SEED_MINER_SEED` (seed the RNG for reproducible
sampling).

## Cost & safety

Default model is Haiku 4.5 with a hard `max_tokens` ceiling: roughly 3¢ per run,
comfortably under $1/month at a weekday cadence. Fund the Anthropic Console with a
small prepaid balance, set a low monthly spend cap, and **do not enable
auto-reload** — a runaway cron/retry misfire should hit the ceiling and stop, not
top up a card. If generation returns malformed output the run fails loudly and
commits nothing, rather than committing a broken state.

## Honest failure mode

If nothing ever gets promoted, this reduces to *"I automated making filler nobody
looks at"* — which is worse than no project at all. The entire thing rests on the
weekly review actually happening and something eventually hitting. Base rate for
this kind of tail-end search is roughly one promotable seed per 30–50. If you
won't commit to the review, build a curated shortlist instead.
