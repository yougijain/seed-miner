# Garden Plot Clustering Under Harvest-Window Endogeneity

## Question
Does K-means clustering of community garden plots remain stable when endogenous harvest-window scheduling constraints reshape the effective feature space?

## Background
Standard clustering treats plot features (soil pH, sun hours, water access) as stable. But in real community gardens, plots near shared infrastructure (compost bins, water stations, group harvest days) face temporal coupling: gardeners are pressured to harvest in narrow 5-7 day windows. This creates an endogenous confound: plots that *should* cluster by agronomics may re-cluster by logistics.

## Method
1. Generate 20 synthetic plots with stable agronomic features (soil pH, sun hours, water access).
2. Assign harvest windows: plots near shared stations get tightly coupled windows (±5 days); others are free (±15 days).
3. Cluster Scenario A: K-means on agronomic features only → baseline clusters.
4. Cluster Scenario B: K-means on agronomic features + harvest day → constrained clusters.
5. Measure cluster label flips; focus on plots near stations (where endogeneity is strongest).

## Findings
The code identifies plots whose cluster membership changes when harvest-day is included, revealing that the endogenous constraint (shared logistics) reshapes clustering even though true agronomic drivers are unchanged. This is NOT noise; it's systematic: plots near shared stations are more likely to flip.

## Data
**Synthetic.** Structure is real (endogenous confound exists in real community gardens), but values are generated to exhibit the effect cleanly.

## Limitation
To prove causality (harvest logistics cause re-clustering), you'd need instrumental variables or quasi-experimental design (e.g., staggered station opening dates). This seed only demonstrates that the phenomenon occurs in data with the right structure.

## Angle
The non-obvious bit: clustering is usually presented as unsupervised dimension reduction. Here, clustering becomes a *diagnostic tool* that exposes endogenous confounding—not by failing, but by producing different answers when you add a confounded feature. Reversing the typical use of clustering failure into a signal of unobserved structural constraints.

---
*Auto-generated seed. For exploratory analysis only.*
