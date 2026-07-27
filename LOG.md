<!-- AUTO-GENERATED from state/log.jsonl on every run. Do NOT edit by hand. -->
<!-- To promote/reject a seed:  python runner/review.py promote <id> --note "why" -->

# Review log

Checking a box = promoted. `[x]` promoted · `[~]` rejected · `[ ]` unreviewed. Promote with `python runner/review.py promote <id>`.


## Week of 2026-07-27

- [ ] `2026-07-27_beekeeper-log-coherence-decay` — Beekeeper Log Variance Decay: Inferring Inspection Fatigue from NLP Coherence Drift
      self: "This has legs: NLP on logs assumes static vocabulary/style, but beekeepers' logs naturally drift in precision, detail, and reference consistency as fatigue or attention shifts. That drift itself—measured as token-sequence entropy or semantic coherence within rolling windows—becomes a signal. The insight is that the *noise floor* of the writing is more stable and informative than the surface content."

## Week of 2026-07-20

- [ ] `2026-07-23_farmers-market-vendor-adjacency-causal` — Vendor Layout Spillover: Does Adjacency Confound the Market-Day Effect on Sales?
      self: "Has legs: the tension between observational causal inference (built for independent units) and spatial-network data (where treatment and confounders are entangled by geography) forces a genuine methodological compromise that exposes the limits of covariate adjustment."
- [ ] `2026-07-24_brood-temporal-hypergraph-collapse` — Temporal Hypergraph Collapse: Detecting Brood-Pattern Asymmetries via Time-Windowed Subgraph Isomorphism
      self: "The seam is real: standard network analysis treats contacts as undirected snapshots, but brood disease propagation is causal (nurse→brood, not vice versa) and time-indexed; we must engineer directed lagged edges from temporal contact sequences. This forces explicit handling of the temporal asymmetry rather than applying off-the-shelf centrality."
- [ ] `2026-07-26_thrift-clustering-under-description-noise` — Thrift Price Clusters Under Measurement Error: Can k-means recover categories when item descriptions are noisy proxies for condition?
      self: "This has legs because the core tension (clustering assumes latent clean structure, but thrift pricing is driven by messy grading systems) forces a methodological compromise: either you admit the cluster labels don't align with ground truth, or you ask what structure survives *despite* noise—which is honest and uncommon in tutorial examples."
