# Trail Closure Cascade Recovery via Flow-Network Bottleneck Detection

## Question
Which trail segments act as **structural chokepoints** in a trail network—i.e., segments whose closure causes disproportionate flow reduction because other segments depend on them to reroute hikers during cascading closures?

## Method
Instead of treating trails as a co-occurrence graph ("which trails close together?"), model the trail network as a **flow network** where each segment has capacity. Then:

1. Compute baseline max flow from trailhead to summit with all segments open.
2. Remove individual segments; measure flow reduction (segment criticality).
3. Remove *pairs* of segments; measure whether joint impact exceeds sum of individuals.
4. Positive "synergy" = hidden dependency: one segment was backing up flow that another reroutes.

The synergistic pairs form a **bottleneck dependency graph** revealing which segments must jointly be maintained (e.g., if seg_C and seg_G form a narrow reroute path, closing both cascades failures).

## Why This Is Non-Obvious for Graph-Network Analysis
Standard graph techniques (betweenness centrality, clustering) work on *static topology* or *co-occurrence*. This flips the analysis to **constraint propagation under removal**: a segment is critical not because it's central, but because its absence forces flow through bottleneck pairs elsewhere. The technique must detect *emergent dependencies* (synergies) that don't exist in the topology alone—they emerge from capacity constraints and the specific demands (flow) on the network.

## Data
**Synthetic.** A realistic 12-segment trail network with branching paths and varying capacities, plus synthetic closure co-occurrence records (real data would come from trail GPS + maintenance/weather closure logs). The synthetic data is constructed so that seg_C and seg_G form a genuine bottleneck reroute; closing both together cascades flow loss beyond individual effects.

## Limitations
- No real closure data (capacity/flow estimates are arbitrary).
- No temporal dynamics (real closures are staggered; this treats them as simultaneous).
- Assumes "capacity" as a proxy for trail width/difficulty; real hikers choose routes by preference, not just flow equilibrium.

---
*Auto-generated seed from domain×technique matrix: `local_trail_conditions` × `graph_network_analysis`.*
