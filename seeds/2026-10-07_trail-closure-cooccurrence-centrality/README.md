# Trail Network Centrality Under Partial Observability

## Question

Can we infer which trail segments are structurally critical (likely bottlenecks that, when closed, force closures of many downstream segments) using **only** sparse, noisy closure event logs—without explicit trail topology data?

## The Non-Obvious Angle

Standard graph network analysis assumes you know the network structure (nodes and edges) and computes centrality metrics on that fixed graph. This project **inverts** the problem:

1. We don't have an explicit trail map.
2. We only observe **closure events** (sets of segments simultaneously reported as unavailable).
3. We treat co-occurrence of segments in closure events as a noisy signal of topological dependency.
4. We reconstruct a co-occurrence graph from this sparse data, threshold it to reduce noise, then compute centrality.
5. **Hypothesis:** segments with high betweenness in the co-occurrence graph are likely structural bottlenecks—closing them cascades failures to many others.

The technique modification: graph centrality metrics are forced to operate on a network **inferred from partial, noisy observations of absence** rather than on ground truth. This mirrors real-world network reconstruction problems (protein interactions, ecological food webs) where you only see correlated disruptions, not the full topology.

## Data

**Synthetic.** Generated 50 closure events with latent "bridge" segments (2, 3, 5) that, when closed, typically trigger downstream segment unavailability. Noise is injected as random false-positive closures. This structure ensures the co-occurrence signal actually exists in the data.

## Limitations

- **No ground truth.** We cannot validate whether the inferred centrality rankings match the true trail network structure, since the trail topology is not provided in the closure logs.
- **Threshold sensitivity.** Results depend on the choice of co-occurrence weight threshold (here, 2). Tuning required for real data.
- **Confounding.** Co-occurrence may reflect management policy ("we close related segments together for safety") rather than topological necessity.
- **Sparsity.** With only 50 events on 12 segments, the co-occurrence graph may be disconnected or miss true dependencies if they rarely occur jointly.

## What This Seed Demonstrates

The value of graph centrality shifts when you must first reconstruct the graph from noisy, incomplete observations. Standard algorithms assume the graph is given; this project shows how the **inference step itself** becomes the methodological challenge, and centrality metrics become a diagnostic tool for validating the inferred structure.

---

*This is an auto-generated seed from the data-analyst project farm. It is exploratory and may not lead anywhere.*
