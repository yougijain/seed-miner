# Youth Chess Pairing Fairness via Round-Sequence Clustering Collapse

## Question

Can we detect structural unfairness in youth chess tournament pairings by clustering players on their *sequences of opponent ratings across rounds*—and measuring when clustering fragmentation reveals latent rating-group imbalance?

## The Non-Obvious Angle

Standard fairness audits cluster on player attributes (rating, age, gender). This project inverts the logic:

- **Feature engineering twist:** Instead of clustering players directly, cluster on the *sequences of opponent ratings each player faced*. This encodes pairing strategy as a multivariate time series.
- **Failure is the signal:** In fair tournaments, opponent-sequence clustering should recover coherent difficulty-based groups. When clustering *fragments* (many isolated players, low purity vs. ground-truth rating groups), it signals that pairing strategy has segregated rating tiers.
- **Why standard clustering struggles:** Opponent sequences are irregular (players skip rounds, drop out), so clustering algorithms designed for fixed-dimension feature spaces must either pad/aggregate sequences (losing temporal structure) or treat each player as near-independent, causing fragmentation.

## Real vs. Synthetic

**Synthetic.** The dataset is generated inline to demonstrate two scenarios:
1. **Fair pairings:** Cross-group matchups distributed evenly across rounds → clustering recovers both rating groups naturally.
2. **Unfair pairings:** Low-rated players face mostly low-rated opponents; high-rated face mixed → clustering fragments because low-rated sequences form an isolated, self-reinforcing cluster.

No real tournament data is used. A real deployment would require:
- Actual pairing records (round, player_id, opponent_id, opponent_rating).
- Ground-truth rating assignments (to validate purity).
- Domain expert confirmation that fragmentation actually indicates bias vs. legitimate structural features of the tournament (e.g., all young beginners in one section).

## Limitation

Clustering fragmentation is *correlated* with unfairness, not *causal*. A single-age-group tournament *should* cluster into a single cohort. This analysis assumes mixed rating tournaments should form balanced clusters; real fairness assessment requires:
1. Domain context (is the tournament stratified by design?).
2. Direct fairness metrics (cross-group pairing rate, rating variance within rounds).
3. Causal inference to isolate algorithmic bias from genuine structural constraints.

This is an exploratory seed; take results as anomaly flags, not proof.

## Auto-Generated Seed Notice

This project was auto-generated as part of a data-analyst project seed farm to explore non-obvious applications of clustering to domain-specific data shapes. The code is example-grade and meant to spark ideas, not production use.
