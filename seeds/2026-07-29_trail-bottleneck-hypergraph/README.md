# Trail Bottleneck Detection via Temporal Hypergraph Switching

## Research Question

Can we detect trail segments where hiker flow transitions from parallel/dispersed to sequential/congested by collapsing continuous GPS trajectories into time-windowed directed graphs and measuring positional in-degree skew?

## Non-Obvious Technical Angle

Standard graph network analysis assumes a pre-defined node set and edge list. Trail GPS data arrives as continuous **hyperedges** (each hiker's trajectory spans hundreds of GPS points). The technique doesn't apply directly—we must:

1. **Collapse hyperedges**: Bin time into windows and coarse-grain space (1 km position bins).
2. **Convert to digraph**: Treat hiker positions within a window as nodes; edges encode spatial overlap ("contact").
3. **Detect regime shift**: Bottlenecks manifest as high **in-degree skew**—many hikers forced into the same few positions, high variance in node degree distribution.

This forces us to detect bottlenecks not as structural graph properties (e.g., cutpoints) but as **behavioral transition signatures** in the temporal digraph sequence—a use case graph_network_analysis wasn't built for.

## Data

**Synthetic, embedded inline.** Simulates 200 hikers on a 15 km trail over 400 minutes with:
- Poisson arrival process.
- Constant speed hiking (0.02–0.08 km/min).
- **Artificial bottleneck**: km 6–8 has 70% speed reduction, forcing hikers into a narrow temporal/spatial band.

Bottleneck zone corresponds to t ≈ 150–250 min (hikers moving at reduced speed pile up).

## Limitation

- Speed reduction is synthetically imposed; real bottleneck detection would need field validation.
- In-degree skew is a proxy; ground truth (crowding, self-reported difficulty) unavailable.
- Coarse 1 km spatial binning and 20 min temporal windows are arbitrary; sensitivity analysis omitted.
- Results show *correlation*, not causation (e.g., terrain steepness not modeled).

## Seed Status

Auto-generated exploratory seed for public seed farm. Most seeds are discarded; this one survives only if the temporal hypergraph collapse is genuinely novel to you.
