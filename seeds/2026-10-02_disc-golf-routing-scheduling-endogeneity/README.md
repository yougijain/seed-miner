# Disc Golf Round Scheduling Under Course-Routing Endogeneity

## Question

Can bin-packing optimization detect structural unfairness in player groupings when the grouping constraint (similar observed scores) is endogenously determined by an unobserved confounder (which 9-hole course routing each player chose)? Specifically, do bin-packing *failure patterns* (which skill distances the optimizer accepts to form groups) reveal the existence of routing bias?

## Why This Is Non-Obvious

Standard bin-packing treats capacity constraints as exogenous. Here, the "capacity" (acceptable score spread within a group) is endogenously set by two competing forces:
- **True skill differences** (what we want to minimize)
- **Routing effects** (unobserved course difficulty bias, which we don't control for)

When routing confounds the observed scores, the bin-packing optimizer will sacrifice skill homogeneity (true constraint) to maintain score homogeneity (measured constraint). By measuring *when* and *how much* the optimizer violates skill-based expectations, we reverse-engineer evidence of the confounder.

## Data

**Synthetic.** 60 players, each with:
- Observed round score (influenced by true skill + unobserved routing choice)
- True skill (the ideal grouping criterion, unmeasured)
- Routing choice: A or B (A adds +8 penalty, B adds +3)

The confounder is realistic: in disc golf, the same player's score varies by ~5–10 strokes depending on which half of the course they play first, but tournament directors don't always know this routing choice.

## Method

1. Bin-pack players by observed score (the standard optimization).
2. For each group, measure:
   - Routing imbalance: how skewed is the A/B split?
   - Packing quality: score spread within group
   - True skill spread: what the spread would be if routing didn't exist
3. Correlate routing imbalance with packing quality.

**If correlation is high**, the optimizer is accepting routing-imbalanced groups to keep observed scores tight. This is evidence of endogeneity.

## Limitation

With a small synthetic dataset, the signal is somewhat artificial. In real data, you'd need:
- Repeated rounds from same player (to estimate routing effect per player),
- A sample of players who played both routings (to measure causal routing effect),
- Or external metadata on course difficulty.

This seed demonstrates the *diagnostic principle*, not a production-ready solution.

---

**Note:** This is an auto-generated seed from a data-analyst project farm. It explores the seam where optimization_scheduling meets the domain-specific confounder structure of disc golf.
