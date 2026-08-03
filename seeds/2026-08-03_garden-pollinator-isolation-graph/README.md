# Pollinator Isolation in Garden Networks

## Question
Can we detect when a community garden plot's flowering schedule creates isolation in the pollinator-visitation network—where graph centrality collapses not because plots are physically removed, but because flowering *timing* desynchronizes the edge structure?

## Method
**Temporal Graph Reconstruction**: Standard graph analysis assumes a static network. Here, we rebuild the pollinator-sharing network *each week* using only plots that are actively flowering. Edges represent potential pollinator transfer (co-flowering = shared pollinators). This forces the graph structure to change dynamically based on phenology, not plot removal.

When plots flower at different times, they become **isolated components** in their non-overlapping weeks, even though they're physically adjacent. This is the non-obvious angle: isolation is *temporal*, not spatial.

## Data
**Synthetic, by design**: 12 plots with assigned crops and flowering windows (1–8 weeks). The structure is realistic: tomatoes & basil flower early/mid, carrots & kale flower late, radishes flower very early and briefly. This creates natural timing desynchronization.

## Key Finding
- Plots 9 & 10 (carrots, kale) are isolated 6+ weeks because they bloom only weeks 6–8, missing the mid-season clusters.
- Plot 7 (radish) blooms only weeks 1–2, then becomes isolated.
- Network fragmentation peaks weeks 6–8: average degree drops, number of connected components rises.

## Limitation
This is a single synthetic dataset with hand-chosen windows. Real pollinator data (bee visit logs, flower phenology cameras) would reveal whether timing-driven isolation actually predicts pollinator diversity or yield. Without that ground truth, we've shown the *signal* but not its ecological meaning.

## Technique Tension
Standard centrality measures (betweenness, degree) assume the graph is fixed or nodes are deleted. Here, **edges vanish and reform** based on a time-indexed attribute (flowering start/end). This requires treating the network as a sequence of snapshots rather than a single structure, which breaks many traditional assumptions about "importance" and "bottleneck."

---
*Auto-generated seed. Proof-of-concept, not validated on real data.*
