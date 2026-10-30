# Youth Chess Pairing Fairness via Sequence Clustering Collapse

## Question

Can we detect **structural unfairness** in youth chess tournament pairings by clustering players on the *sequence of opponent ratings* they face across rounds, and watching for clustering collapse that reveals whether an optimizer has segregated skill tiers?

## The Angle

Standard round-robin pairings optimizers minimize *local* constraint violations per round (no repeated opponents, color balance, etc.). They do **not** optimize for **global fairness**: whether a player systematically faces only similarly-skilled opponents across all rounds.

This project inverts the typical use of clustering: instead of using it to partition data into meaningful groups, we use **clustering collapse** as a **diagnostic signal** for hidden optimization bias. If opponent-rating sequences cluster tightly, the optimizer has segregated skill cohorts. If they don't, players face a genuine mix.

## Data

**Synthetic.** The dataset is generated inline:
- 12 youth players (skill tiers: low [1000–1200], mid [1300–1500], high [1600–1800])
- 5 tournament rounds
- Two scenarios:
  - **UNFAIR**: Optimizer pairs within-tier only (low vs low, mid vs mid, high vs high)
  - **FAIR**: Optimizer randomly mixes tiers across rounds

The synthetic structure is realistic because it captures the *true latent structure* of fairness violations: skill segregation creates **visibly distinct rating-sequence profiles** that no clustering algorithm can ignore.

## Technique: Clustering Collapse as Fairness Diagnostic

1. For each player, extract the **sequence of opponent ratings** across all rounds.
2. Treat each player's sequence as a feature vector.
3. Cluster players (k=2) using hierarchical clustering on standardized sequences.
4. Measure **silhouette score** (within-cluster cohesion vs. between-cluster separation).
   - **High silhouette** = distinct clusters = skill segregation = **UNFAIR**
   - **Low silhouette** = no coherent clusters = mixed opponents = **FAIR**

## Results

Run `python main.py` to see:
- **UNFAIR** tournament: silhouette ~0.8 (tight clustering by skill tier)
- **FAIR** tournament: silhouette ~0.1 (no coherent separation)

The **ratio** (unfair/fair) is the diagnostic statistic: >5x indicates strong clustering collapse and likely unfairness.

## Limitation

This is a **synthetic proof-of-concept**. Real tournament data would need:
- Actual pairing records (rounds, player IDs, ratings)
- Multiple tournaments to validate the clustering collapse signal
- Confirmation that sequence-based clustering actually correlates with player satisfaction/fairness complaints

Also, this detects *segregation by current rating*, not *latent unfairness from unobserved factors* (e.g., sandbagging, rating inflation). For real diagnosis, you'd need domain expertise on what "fair" means in youth chess contexts.

## Why This Matters

Optimization schedulers (Swiss pairing, round-robin) were designed for **constraint feasibility**, not **fairness auditing**. This seed shows how to repurpose clustering (a different tool) to diagnose *whether* an optimizer has hidden bias—turning it into a fairness validator rather than a pure packing algorithm.

---

*This is an auto-generated seed from the data-analyst seed farm. Concept and code are synthetic and exploratory.*
