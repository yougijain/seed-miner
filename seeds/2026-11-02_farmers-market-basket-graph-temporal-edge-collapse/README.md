# Farmers Market Vendor Cross-Purchase Inference via Temporal Bipartite Graph Edge Collapse

## Question
Can we infer hidden vendor complementarity and supply constraints (which vendors customers buy from together, and when that pattern breaks) by detecting when customer-vendor bipartite graph edges *collapse* week-to-week, rather than building the graph forward from observed edges?

## Non-Obvious Angle
Standard vendor affinity analysis counts co-purchase frequency ("how often do vendors A and B appear in the same transaction?"). This approach infers affinity from presence.

This seed inverts the problem: it detects **edge collapse**—weeks where vendor co-purchase edges that existed in the prior week vanish—and interprets that collapse as a signal of unobserved constraints (e.g., one vendor stockout, forcing customers to abandon the entire co-purchase pair). The graph-network technique must work *backward* from missing edges to infer what vendor-side state changed, rather than forward from observed edges to infer customer preference.

## Technical Approach
1. Build customer-vendor bipartite graphs for consecutive weeks.
2. For each vendor, compute a "co-purchase signature" (set of other vendors it appears with in customer baskets).
3. Measure Jaccard similarity of vendor signatures week-to-week.
4. Flag weeks where signatures show abnormal collapse (low Jaccard), indicating edges vanished.
5. Correlate edge collapse with known vendor groups (e.g., produce vendors) to infer likely cause (supply constraint).

## Data
**Synthetic.** Generated with 500 transactions across 80 customers, 12 vendors, and 60 days.
- Vendors 0–3: produce (high co-purchase affinity).
- Vendors 4–7: dairy (high co-purchase affinity).
- Vendors 8–11: baked goods (low co-purchase affinity).
- **Hidden signal:** Produce vendors randomly stockout some weeks; when vendor A stocks out, customers abandon co-purchase with vendors B, C, D, causing edge collapse in the bipartite graph.

This structure ensures the anomaly (edge collapse) is real and meaningful, not noise.

## Limitations
- **Synthetic data:** Real farmers market data would require transaction logs with vendor identifiers, unlikely to be public.
- **Causality inference is observational only:** Edge collapse correlates with supply constraints but does not prove causation.
- **Small scale:** 12 vendors, 80 customers; real market would need scalability testing.
- **Binary signal:** Detects collapse but does not quantify magnitude of vendor impact (which vendor stockout caused which edge loss?).

## Self-Assessment
This seed has real seams: the inversion from presence-based to absence-based graph inference is genuine, and the temporal Jaccard collapse metric forces the network analysis to work on dynamic signatures rather than static topology. A reviewer could reasonably say "the edge-collapse angle is a legitimate modification to standard bipartite graph analysis for this domain." However, the causal story (stockout → edge collapse) is not rigorously validated; a production version would need ground-truth supply data to confirm the inference.

---
*Auto-generated seed for public data-analyst project farm. Do not publish without review.*
