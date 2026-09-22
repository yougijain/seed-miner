# Trail Closure Cascade Graphs

## Question
**Can we infer which trail segments are topologically dependent on others by analyzing the temporal cascade of closures—treating simultaneous or near-simultaneous closure events as edges in a directed acyclic graph?**

## Non-Obvious Angle
Standard graph network analysis on trails looks at spatial connectivity (which segments touch) or co-occurrence patterns (which segments are visited together). This project inverts that: it builds a **directed temporal causality graph** from *absence patterns*.

- **The problem**: A segment A may not be directly visited, but closing A forces closures of B and C within days (because they lose access). Traditional co-occurrence graphs miss this because B and C are not simultaneously *present*—they're simultaneously *absent*.
- **The modification**: Graph algorithms designed for static networks (centrality, reachability, bottleneck detection) are repurposed to find "closure bottlenecks"—segments whose removal (closure) would prevent cascading closures to others. This requires treating the **absence sequence** as the signal, not presence.

## Data
**Synthetic**. A small trail closure log with 13 events across 8 segments over Feb-Feb 2024. The synthetic pattern encodes a true dependency: segment A is a junction; closing A forces B and C offline 1-2 days later (they can't be accessed for maintenance). The data has the right structure to make the technique work: temporal clustering in closures that would be invisible in static spatial connectivity.

## Limitations
- Synthetic data is tiny; real data would need months/years of closure logs to build robust cascade graphs.
- Lag threshold (2–3 days) is arbitrary; real projects would need domain tuning.
- Does not account for external factors (e.g., weather, crews): all closures treated as comparable.
- Network is sparse; impact detection is rough without larger datasets.

## What works
- Correctly identifies segment A as a bottleneck (closing it would affect B and C).
- Shows that graph algorithms can be repurposed to work on temporal causality (not just spatial/semantic edges).

## Auto-Generated Seed
This is an exploratory seed from a data-analyst project seed farm. It is not a full analysis—it's a proof-of-concept that the technique applies to the domain in a non-obvious way.
