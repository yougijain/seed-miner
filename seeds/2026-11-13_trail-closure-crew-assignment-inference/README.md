# Trail Closure Temporal Clustering: Inferring Hidden Maintenance Crew Structure via Inverse Causal Inference

## Question

Can we recover the hidden assignment of trail segments to maintenance crews **only** from the temporal clustering of observed closure events, without direct crew-assignment data?

## Non-Obvious Angle: Inversion of Causal Direction

Standard causal inference asks: *"Does treatment cause outcome?"*

This project inverts it: *"Does observed outcome co-occurrence reveal unobserved treatment?"*

Normally, we'd use causal inference to isolate a treatment effect (e.g., crew maintenance → closure duration).
Here, we treat closure-timing clustering as a **noisy backdoor confounder signal** and use clustering + correlation metrics to recover the hidden confounder itself (crew identity).

## Data

**Synthetic.** Generated 12 trail segments assigned to 3 invisible maintenance crews (4 segments per crew). Each crew works in a distinct seasonal window:
- Crew A: days 50–150
- Crew B: days 200–300  
- Crew C: days 320–365

Closures within a crew are temporally clustered (noisy version of crew's work season); closures between crews are independent.

## Method

1. **Compute closure-timing correlation** between all segment pairs (fraction of closures in segment *i* within 30 days of closures in segment *j*).
2. **Hierarchical clustering** on the distance matrix (1 − correlation).
3. **Compare inferred clusters** to true (hidden) crew assignments via Adjusted Rand Index and NMI.

## Limitation

This works **only** when crew closures are strictly temporally clustered and crews have non-overlapping seasons. In real data:
- Seasonal variation, weather, emergency repairs, and crew reallocation destroy clean clustering.
- Requires strong assumption that closure timing *is* the dominant signal of crew identity.
- Cannot distinguish "same crew" from "different crews working same season."

## What's Real

The causal-inference inversion is real: using outcome clustering to infer latent confounders. The data is synthetic because real trail-closure APIs (e.g., AllTrails, USDA) don't publish crew assignments.

---

*Auto-generated seed for data-analyst project farm.*
