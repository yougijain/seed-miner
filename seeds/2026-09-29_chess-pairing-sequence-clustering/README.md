# Youth Chess Tournament Pairing Fairness: Sequence Clustering Collapse Detection

## Question

**Can clustering on opponent-rating *sequences* (not static player features) detect structural unfairness in youth chess tournaments by identifying when player cohorts fracture into fundamentally different difficulty curves?**

## Why This Angle Matters

Standard fairness audits check each pairing individually or aggregate across all pairings. But structural unfairness is **temporal and path-dependent**: a player paired strong → weak → strong faces different cumulative difficulty than weak → strong → strong, even if each individual round is locally balanced.

By clustering on **opponent-rating sequences across rounds**, we force the algorithm to detect when:
- Elite players systematically face easier curves (strong → mid → weak)
- Junior players systematically face harder curves (elite → strong → elite)
- The tournament structure biases certain skill groups despite per-round fairness

This is invisible to:
- Cross-sectional player clustering (rating, age, win %)
- Per-round pairing fairness checks
- Standard machine-learning fairness audits (which assume i.i.d. treatment)

## Data: Synthetic, Intentionally Unfair Tournament

**Why synthetic?** Public youth chess pairing records don't expose full opponent sequences in machine-readable form; synthetic data lets us inject a known unfairness signal and verify the technique detects it.

**Structure:**
- 24 players across 4 skill groups (elite, strong, mid, junior)
- 5 tournament rounds
- Elite players: systematically paired against mid/strong opponents (avg ~1600 rating)
- Junior players: systematically paired against elite/strong opponents (avg ~1900 rating)
- Mid/strong: balanced pairings

**Signal:** Opponent-rating sequences encode the unfairness; clustering on these sequences will fracture into cohorts with significantly different average opponent difficulty.

## The Technique: Sequence Clustering Collapse

1. **Extract** opponent rating for each player in each round → shape (n_players, n_rounds)
2. **Normalize** via zscore (opponent difficulty relative to player's own pairings)
3. **Cluster** using Ward linkage on normalized sequences
4. **Detect unfairness** if clustering produces multiple cohorts with significantly different average opponent ratings

**Key insight:** If pairings were fair, all players should have similar opponent-difficulty curves (after controlling for their own rating). Unfairness causes the manifold to shatter.

## Limitations

- **Synthetic signal:** Real tournament unfairness is subtle and confounded by skill variance, draw luck, and rating inflation. This seed proves the *method*, not that real tournaments are unfair.
- **Assumes Euclidean structure:** Ward linkage on zscore-normalized sequences assumes opponent difficulty varies smoothly; discrete pairing quirks (e.g., one-off blunders) won't be detected.
- **Small scale:** 24 players, 5 rounds is a toy tournament. Real fairness analysis requires many more players and rounds to disentangle signal from noise.
- **No ground truth:** Without labeled "fair" vs. "unfair" tournaments, we can't validate the threshold for "collapse = unfairness."

## What's Real vs. Synthetic

| Aspect | Status |
|--------|--------|
| Technique (sequence clustering) | Real |
| Data | Synthetic |
| Unfairness signal | Injected (artificial) |
| Interpretation (cohort fracturing → bias) | Plausible but unvalidated |

## How to Run

```bash
python main.py
```

Expected output: 2 or 3 cohorts with significantly different average opponent ratings, suggesting structural unfairness.

---

**This is an auto-generated seed from a data-analyst project farm.** Modify, extend, or discard as needed.
