# Youth Chess Pairing Clustering Under Skill-Volatility Endogeneity

## Question
When young chess players compete across multiple tournament rounds, their skill level (and thus the strength of opponents they face) shifts endogenously based on win/loss outcomes. Can K-means clustering of player "opponent-strength sequences" remain stable across consecutive rounds, or does endogenous rating volatility force the same player to cluster differently in Round N vs Round N+1—even though their core identity is unchanged?

## Why This Matters
Standard clustering assumes feature vectors are static. But in youth chess tournaments, **player rating and opponent-strength are endogenous outcomes of prior-round performance**. This violates the stationarity assumption underlying most clustering algorithms. The interesting question is not "cluster the pairings," but rather "measure how much clustering membership flips when the underlying tournament dynamics shift the feature vectors mid-tournament."

## Data
**Synthetic.** The dataset is generated inline:
- 24 players with initial ELO ratings drawn from N(1200, 150), clipped to [800, 1800].
- 5 tournament rounds with Swiss-style random pairings.
- After each round, player ratings shift endogenously based on match outcomes (Elo-like updates).
- Each player's feature vector includes their current ELO and average opponent strength faced in the round.
- K-means clusters players into 3 tiers (skill groups) *independently for each round*.
- Cluster membership is tracked across rounds to detect drift.

## Key Limitation
This is synthetic data, so the endogenous rating shift is deterministic (outcome-based) but the tournament structure is simplified (random pairings instead of true Swiss-system). Real youth chess data would require access to official tournament records (not easily public).

## What This Reveals
If cluster membership is highly unstable (same player in different clusters across rounds), it means:
1. The clustering algorithm is sensitive to the endogenous feature drift.
2. **Opponent-strength sequence is not a stable clustering basis** in a tournament context.
3. Fair pairing systems need to account for this non-stationarity rather than treating player identity as fixed.

## How to Run
```bash
python main.py
```
Outputs:
- Console summary of players with high cluster drift.
- `cluster_drift_results.json` with full results.

## Technique Modification
Standard K-means is applied unchanged, but the **interpretation is inverted**: instead of "cluster players to find stable groups," the analysis reads **clustering instability** as evidence of endogenous feature non-stationarity. The clustering doesn't fail; it correctly reflects that the underlying tournament dynamics violate the stationarity assumption. The seam is sharp: clustering reveals its own assumptions are broken.

---
*This is an auto-generated seed from a data-analyst project seed farm. Most seeds are exploratory scratches.*
