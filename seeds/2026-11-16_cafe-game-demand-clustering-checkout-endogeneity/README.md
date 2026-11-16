# Board Game Cafe Demand Clustering Under Checkout-Duration Endogeneity

## Question
Can K-means clustering segment board games into true demand tiers when the **same game's checkout duration artificially constrains session-availability windows**, causing demand signals to be confounded with operational feasibility?

## The Non-Obvious Angle
Standard clustering of cafe inventory treats "observed checkout events" as a demand signal. But in a cafe with fixed tables and opening hours, a game's observed checkout count is partly determined by how long it occupies a table:

- **Gloomhaven** (2hr game, inherently high-demand): only 10 windows/week → fewer observed checkouts
- **Ticket to Ride** (45min game, low-demand): 20 windows/week → more observed checkouts

Naive clustering on observed count will misclassify game tiers because **duration endogenously reduces the availability windows**, which confounds the demand signal.

## Methodology
We test three clustering approaches:

1. **Naive**: cluster on observed_checkouts + player_count + complexity + footprint
2. **Confound-Adjusted**: cluster on (observed_checkouts / weekly_windows) + player_count + complexity
3. **Duration-Explicit**: cluster on observed_checkouts + player_count + complexity + checkout_duration_min

We measure:
- **Silhouette score** (internal cluster cohesion)
- **Purity** (match to ground-truth demand tiers)

## Data
**Synthetic**, generated with known ground truth:
- 30 games across 3 true demand tiers (low, medium, high)
- Checkout duration sampled from tier-conditional distributions
- Observed checkouts = f(true_demand, duration, windows) + noise
- The confound is **real and baked into the data**

## Expected Result
If duration is a true confounder, **Adjusted or Explicit clustering should achieve higher purity** than Naive, forcing the analyst to either:
- Normalize demand by opportunity (windows), or
- Explicitly include duration as a control variable

This is a genuinely non-obvious application of clustering: **the problem isn't grouping games, it's untangling a measurement confound that standard distance metrics can't see.**

## Limitations
- **Synthetic data**: real cafe data would require logging checkout events + table occupancy timings
- **Known ground truth**: in practice, true demand tiers are latent
- **Linear confounding**: assumes windows scales linearly with duration; real availability may be nonlinear

## Run
```bash
python main.py
```

---
*Auto-generated seed from [data-analyst-seed-farm](https://github.com/[repo])*
