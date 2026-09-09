# Trail Difficulty Matching via Temporal Co-Absence Graphs

## Question

Can bipartite matching infer latent trail difficulty tiers by treating simultaneous closure windows (weather, maintenance) as co-occurrence signal, rather than learning from explicit hiker behavior?

## The Non-Obvious Angle

Standard trail recommendation matching learns from *co-hiking*: if users hike Trail A and Trail B in sequence or within a time window, they're similar. This seed inverts the signal:

**Co-absence as affinity**: When two trails are closed at the same time (due to weather, maintenance cascades, or infrastructure scheduling), we treat that co-absence as evidence they belong to the same difficulty/maintenance tier. The bipartite matching algorithm then tries to pair trails that co-close most frequently, under the hypothesis that management logic (not user behavior) reveals the true difficulty structure.

This forces the matching algorithm to operate on a fundamentally different feature space: temporal overlap of unavailability instead of presence patterns.

## Data

**Synthetic**. Generated 10 trails with 365 days of closure history (2024). Ground truth: three latent difficulty tiers (Easy, Medium, Hard). Closures cluster within tiers via simulated weather and maintenance events. No real-world hiking data needed.

## What's Real vs. What's Synthetic

- **Real**: Bipartite matching algorithm (Hungarian, `scipy.optimize.linear_sum_assignment`).
- **Real**: The idea that infrastructure management (closures) might reveal structural similarity.
- **Synthetic**: The 365-day closure dataset and ground truth groupings.
- **Limitation**: This is a proof-of-concept only. Real validation would require actual trail closure logs + independent ground-truth difficulty labels from park management or trail surveys. A single year of data is not enough to learn robust patterns; seasonal/annual cycles matter.

## Results

Run `python main.py` to execute. Output:
- Bipartite matching pairs (left partition: trails 0-4; right: 5-9).
- Co-absence affinity scores for each pair.
- Accuracy: fraction of matched pairs in the same latent difficulty tier.
- Expected baseline: ~33% (random guessing across 3 tiers); synthetic data should exceed this significantly if the signal is real.

## Limitations & Next Steps

1. **Single-year data**: Real trail systems need multi-year closure logs to filter noise from true maintenance patterns.
2. **No user behavior**: A real evaluation would compare co-absence-inferred tiers to user difficulty ratings.
3. **Causality unclear**: Closure co-occurrence might be confounded by location (nearby trails close together) rather than difficulty.
4. **Matching is greedy**: This uses bipartite matching for structure, but doesn't model the full co-closure network; clustering or block-model inference might be more natural.

## Notes

This is an auto-generated exploratory seed for a data-analyst project farm. It's meant to be scratched and discarded if not interesting.
