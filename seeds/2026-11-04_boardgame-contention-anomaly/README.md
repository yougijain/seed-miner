# Board Game Checkout Contention Detection via Bin-Packing Anomalies

## Question
Which board games create structural checkout bottlenecks that block the cafe from scheduling other games efficiently?

## Approach
Instead of detecting *rare checkout events*, we invert the anomaly-detection target: we flag games whose checkout patterns correlate most strongly with **packing infeasibility**.

For each game, we compute a "contention signature":
```
contention_score = (peak-hour checkout rate) × (packing failure rate | game checked out)
```

Then run Isolation Forest on these scores to flag games that are statistical outliers in their ability to break table-slot bin-packing.

## Why Non-Obvious
Standard anomaly detection on cafes would flag rare checkout patterns. Here, the technique works *backward* from scheduling failure: we're not looking for unusual games, we're looking for games whose *ordinariness during peak hours* has outsized impact on cafe capacity. The anomaly is in the game's contention signature, not its checkout frequency.

## Data
**Synthetic**: Generated 1 week (168 hours) of hourly checkout snapshots with embedded contention structure:
- 10 games with varying checkout rates
- Hidden bottleneck: `Heavy_Hitter` and `Power_Grid` are long-table games that reduce effective packing capacity when both in play during peak hours (16:00–22:00)
- Real data would come from cafe POS logs + table-assignment logs + manual notes on scheduling friction

## Limitation
Synthetic data is toy-scale; real validation would require:
1. Cafe POS system with game checkout timestamps
2. Table-slot assignment records (what games are paired at each table each hour)
3. Manual or automated logging of "failed to fit a new group" events

Without step 3, we're inferring packing feasibility from checkout counts alone, which is a simplification.

---

*This is an auto-generated seed from a data-analyst project farm. Treat as exploratory scratch work.*