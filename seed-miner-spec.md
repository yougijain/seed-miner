# Seed Miner — Build Spec

An automated project-seed generator with human curation and preference-weighted sampling. It generates one small data-analyst project per weekday, logs it, and steers future generation toward the domain/technique combinations you promote. Most seeds die. The point is the handful that don't.

This doc is the spec to build from *and* it contains the runtime generation prompt (Section 5). Hand the whole thing to Claude Code.

---

## 1. What this is (and isn't)

- **Is:** a wide-net idea explorer with a bandit-style preference tracker and a human review gate.
- **Isn't:** ML in the training sense (not enough labeled events for that), and not a disguised commit-graph padder. The automation is stated openly in every README.

Non-negotiable for the story to hold: the weekly review actually happens, and promoted repos show real human divergence from the seed.

---

## 2. Repo layout (monorepo / seed farm)

```
seed-miner/
├── .github/workflows/generate.yml   # scheduler
├── runner/
│   ├── generate.py                  # main loop: sample → call Claude → write → commit
│   ├── weights.py                   # re-derive tag scores from log
│   └── prompt.py                    # holds the generation prompt (Section 5)
├── state/
│   ├── matrix.json                  # domain × technique grid (Section 4)
│   ├── log.jsonl                    # one line per generated seed (Section 3)
│   └── weights.json                 # current tag scores, re-derived weekly
├── seeds/
│   └── 2026-07-17_disc-golf-network/  # one folder per seed, dated + slugged
│       ├── main.py
│       └── README.md
├── LOG.md                           # human-readable review surface
└── README.md                        # states plainly what this repo is
```

**Why monorepo, not one-repo-per-seed:** 400 dead repos on your profile reads as noise. A single seed farm reads as a deliberate experiment. Promoted diamonds get forked *out* into their own properly-named repos where your real commits go.

---

## 3. State: `log.jsonl` schema

One JSON object per line, appended each run. This is the source of truth; `weights.json` is derived from it.

```json
{
  "id": "2026-07-17_disc-golf-network",
  "date": "2026-07-17",
  "domain": "disc_golf",
  "technique": "graph_network_analysis",
  "obviousness": "non_obvious",
  "title": "Course-layout centrality analyzer",
  "one_line": "Treats a disc golf course as a graph; ranks holes by throwing-line betweenness.",
  "promoted": null,
  "review_note": null
}
```

- `promoted`: `null` until reviewed, then `true` / `false`. This is your only required review action.
- `review_note`: optional one-liner on *why* — this becomes gold if a promoted repo needs a story later.
- `obviousness`: `obvious` / `non_obvious` — set at generation, used by weighting to keep de-prioritizing safe pairings.

---

## 4. The matrix: `matrix.json`

Rich-but-under-mined domains × transferable techniques. Rich = passionate subculture, real recurring decisions, real data exhaust, almost no software people in it.

```json
{
  "domains": [
    "competitive_birdwatching", "community_theater_box_office",
    "beekeeping_colony_health", "disc_golf", "thrift_store_pricing",
    "amateur_radio_contest_logs", "little_league_scheduling",
    "community_garden_plots", "local_trail_conditions",
    "board_game_cafe_inventory", "farmers_market_vendor_sales",
    "youth_chess_tournament_pairings"
  ],
  "techniques": [
    "anomaly_detection", "forecasting", "matching_recommendation",
    "clustering_segmentation", "graph_network_analysis",
    "optimization_scheduling", "nlp_on_logs", "causal_inference_observational"
  ]
}
```

96 cells. Most are boring. Some are nonsensical (nlp_on_logs × a domain with no text — skip). The value lives in cells where the technique is a real but non-obvious fit for that domain's actual pain point, and applying it *forces a modification* to the standard technique. That modification is the original contribution.

---

## 5. The generation prompt (runtime — fed to the API each run)

`generate.py` samples a `{domain, technique}` cell (weighting in Section 6), fetches the last ~30 log entries to avoid repeats, and injects both into this prompt. Use Haiku 4.5 by default; Sonnet if you want the code to read less generic.

```
You are generating ONE small, self-contained data-analyst project seed for a public "seed farm" repo. This is exploratory scratch work — most seeds will be discarded. Your job is to make THIS cell of the domain×technique matrix produce something a curious person wouldn't have deliberately picked, but that has a real chance of being interesting.

CELL FOR THIS RUN:
- Domain: {domain}
- Technique: {technique}
- Target obviousness: NON-OBVIOUS. Do not produce the safe, tutorial-grade pairing. Find the angle where applying {technique} to {domain} forces a genuine modification because the technique wasn't built for this domain's data shape. The seams are the point.

AVOID REPEATING (last ~30 seeds — do not overlap in concept, dataset, or framing):
{recent_titles_and_onelines}

HARD CONSTRAINTS:
- Total code under 200 lines across all files.
- Runs standalone. stdlib + at most one common package (pandas / numpy / networkx / scikit-learn / plotly — pick what fits). No API keys, no paid data.
- If real public data isn't plausibly available for this domain, generate a small synthetic dataset inline that has the RIGHT STRUCTURE to make the technique meaningful (i.e. the anomaly/pattern/signal actually exists in the data — don't fake a result on noise).
- The project must answer ONE narrow, concrete question. State it in the README.
- Be honest in the README about what's real vs synthetic and what the limitation is.

QUALITY BAR — the seed is worth generating only if a reviewer could look at it and say "huh, that modification to {technique} is actually the interesting bit." If the honest answer is "this is just {technique} on a toy dataset," say so in a `"self_assessment"` field rather than dressing it up.

OUTPUT — strict JSON, no markdown fences, no preamble:
{
  "title": "short descriptive title",
  "slug": "kebab-case-slug",
  "one_line": "one sentence: what question it answers + the non-obvious angle",
  "obviousness": "obvious | non_obvious",
  "self_assessment": "one honest sentence on whether this has legs or is filler",
  "commit_message": "conventional-commit style",
  "files": [
    { "path": "main.py", "content": "..." },
    { "path": "README.md", "content": "states the question, real-vs-synthetic, the limitation, and that this is an auto-generated seed" }
  ]
}
Return only the JSON object.
```

Note the `self_assessment` field — it makes the model's own honesty a logged signal you can review against, instead of letting it oversell every seed.

---

## 6. Weighting: `weights.py` (the self-steering part)

This is a preference-weighted sampler, not a trained model. Say that plainly if anyone asks.

**Per-run sampling** (in `generate.py`):
1. Load `weights.json` (tag scores for each domain + each technique).
2. Score each valid matrix cell = `w_domain × w_technique × novelty_bonus × non_obvious_bonus`.
   - `novelty_bonus`: higher if that domain/technique hasn't appeared in the last 10 runs.
   - `non_obvious_bonus`: fixed multiplier that down-weights the single most-obvious technique for each domain (hardcode the obvious pairing per domain in `matrix.json`).
3. Sample a cell proportional to score. Not argmax — you want continued exploration, not collapse onto one favorite.

**Weekly re-derivation** (`weights.py`, run once/week or start-of-run if 7 days passed):
- For each tag (domain or technique), score = exponentially-weighted promote rate:
  `score = Σ(promoted_i × decay^age_i) / Σ(decay^age_i)`, with `decay ≈ 0.9` per week so recent signal dominates.
- Unreviewed seeds (`promoted == null`) count as weak implicit negatives after 14 days (they didn't earn a look).
- Floor every tag at a small minimum so nothing gets permanently frozen out — keeps exploration alive.

That's it. A few lines of arithmetic. The honest description: "it tracks which domain/technique combinations I promote and biases generation toward those, while forcing continued exploration." True, buildable, interview-safe.

---

## 7. Scheduler: `generate.yml`

```yaml
name: generate-seed
on:
  schedule:
    - cron: '0 15 * * 1-5'   # ~weekday mornings; weekends skipped = less robotic
  workflow_dispatch: {}         # manual trigger for testing
jobs:
  seed:
    runs-on: ubuntu-latest
    permissions:
      contents: write            # built-in GITHUB_TOKEN pushes to THIS repo; no PAT needed
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install anthropic pandas numpy networkx scikit-learn plotly
      - run: python runner/generate.py
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - run: |
          git config user.name  "seed-miner-bot"
          git config user.email "seed-miner-bot@users.noreply.github.com"
          git add -A && git commit -m "$(cat state/last_commit_msg.txt)" && git push
```

- Bot commit identity is a **feature** here, not a tell — it makes the automation visible, which is the whole transparency story.
- No timestamp jitter, no fake-human variance. Let it look like what it is.
- `GITHUB_TOKEN` (built-in) is enough since it only pushes to its own repo. Only add a PAT if you later auto-fork promoted diamonds elsewhere.

---

## 8. Review surface: `LOG.md`

Regenerated from `log.jsonl` each run so you skim one file weekly:

```
## Week of 2026-07-14
- [ ] disc-golf-network — Course-layout centrality analyzer
      self: "the betweenness-on-throwing-lines angle is real, worth a look"
- [ ] thrift-pricing-anomaly — Mispriced-item flagger
      self: "just IQR on a toy dataset, probably filler"
```

Checking a box = promote. Promotion = fork the seed into its own real repo and do human work on it. That divergence is where your actual commits live and where the diamond becomes credible.

---

## 9. Cost + safety caps

- Default model Haiku 4.5. ~3¢/run → **under $1/month** at weekday cadence.
- Set a hard `max_tokens` ceiling in the API call so a runaway generation can't balloon.
- Fund Console with ~$10-20 prepaid. Set a **monthly spend cap of ~$5**. Do NOT enable auto-reload — a bug should hit the ceiling and stop, not top up your card.
- The realistic overage risk is a cron/retry misfire (24×), which caps out at annoying, not catastrophic.

---

## 10. Honest failure mode

If nothing ever gets promoted, this reduces to "I automated making filler nobody looks at" — worse than no project. The entire thing rests on you doing the weekly review and something eventually hitting. Base rate for this kind of tail-end search is roughly 1 promotable seed per 30-50. If you won't commit to the review, build a curated shortlist instead — lower ceiling, but it doesn't depend on discipline you might not have.
