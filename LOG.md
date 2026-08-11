<!-- AUTO-GENERATED from state/log.jsonl on every run. Do NOT edit by hand. -->
<!-- To promote/reject a seed:  python runner/review.py promote <id> --note "why" -->

# Review log

Checking a box = promoted. `[x]` promoted · `[~]` rejected · `[ ]` unreviewed. Promote with `python runner/review.py promote <id>`.


## Week of 2026-08-10

- [ ] `2026-08-10_boardgame-session-forecast-inventory-coupling` — Multiplayer Session Duration Forecasting Under Inventory Coupling
      self: "Has real teeth: standard forecasting assumes regressors are exogenous, but inventory-constrained domains have inventory *levels driven by the quantity being forecast*, creating simultaneity bias that requires structural detection or differencing, not just lag-augmentation."
- [ ] `2026-08-11_ll-schedule-timing-anomaly` — Game-Slot Imbalance Detection in Little League Schedules via Isolation Forest on Fixture Timing Skew
      self: "This has real legs: isolation forest is built to find points in feature space that don't cluster with peers, but little-league fairness requires detecting when a *team's aggregated exposure* (to prime time slots + strong opponents) deviates from the population norm—a shape that isolation forest can surface without needing a fairness-specific loss function, revealing hidden biases that slot-counting methods miss."

## Week of 2026-08-03

- [ ] `2026-08-03_garden-pollinator-isolation-graph` — Pollinator Isolation in Garden Networks: Detecting Crop-Specific Connectivity Collapse
      self: "The genuine tension: standard node/edge removal doesn't model garden dynamics; we must redefine edges as time-windowed flowering overlaps, forcing the network to be *reconstructed* at each week. This breaks traditional centrality assumptions and reveals when scheduling creates de-facto isolation despite physical proximity."
- [ ] `2026-08-04_garden-forecast-rotation-structural-break` — Plot Yield Forecast Under Crop Rotation Structural Break
      self: "This has real legs: the tension is genuine—standard forecasting assumes stationarity, but crop rotation *intentionally breaks* the feature-to-target relationship. The forecaster must detect this structural break, not just fit trend. Most garden forecasts ignore rotation; this one treats rotation-induced non-stationarity as the core problem."
- [ ] `2026-08-05_qso-duration-clustering-scheduling-constraint` — QSO Duration Clustering Under Scheduling Constraint Collapse
      self: "This has legs: the tension is real—optimization schedulers assume unimodal task durations, but amateur radio logs exhibit persistent bimodal structure (short contestQSOs vs. long rag-chew conversations). The interesting bit is that *forcing* a bin-packing solver to minimize idle time on a mixed-mode duration stream makes the solver a mode-detector: it will cluster durations not to recover ground truth but to pack efficiently, revealing which operators were actually ragchewin in a contest-logged session."
- [ ] `2026-08-07_theater-temporal-demand-clustering` — Temporal Clustering Collapse in Community Theater Demand: Do Purchase Sequences Reveal Hidden Show-Type Segments When Arrival Timing Is the Primary Feature?
      self: "This has teeth: the insight is that community theater demand is *sequential*—families buy together days in advance, while walk-ups and subscription renewals arrive minutes apart—and clustering on temporal spacing rather than ticket type forces the algorithm to discover audience *behavior modes* hidden in the timing microstructure. The seam is real: standard clustering assumes features, but here the feature engineering itself is the discovery."

## Week of 2026-07-27

- [ ] `2026-07-27_beekeeper-log-coherence-decay` — Beekeeper Log Variance Decay: Inferring Inspection Fatigue from NLP Coherence Drift
      self: "This has legs: NLP on logs assumes static vocabulary/style, but beekeepers' logs naturally drift in precision, detail, and reference consistency as fatigue or attention shifts. That drift itself—measured as token-sequence entropy or semantic coherence within rolling windows—becomes a signal. The insight is that the *noise floor* of the writing is more stable and informative than the surface content."
- [ ] `2026-07-28_farmers-market-vendor-anomaly-composition` — Vendor Revenue Stability via Isolation Forest on Compositional Sales Data
      self: "This has legs: the core insight is that anomaly detection on compositional data (proportions summing to 1) breaks standard distance metrics, and forcing iso-forest to work on the simplex requires either log-ratio transformation or a custom metric—revealing why anomalies in *budget allocation* across categories (not just total revenue) signal real vendor stress."
- [ ] `2026-07-29_trail-bottleneck-hypergraph` — Trail Bottleneck Detection via Temporal Hypergraph Switching
      self: "This has genuine friction: standard graph_network_analysis assumes nodes and edges are pre-defined, but trail data arrives as continuous trajectories (hyperedges spanning many GPS points). The insight is that bottlenecks aren't structural—they're behavioral regime shifts in the *temporal* edge-formation process, requiring us to bin time windows and detect when the digraph transitions from high out-degree (parallel dispersal) to low out-degree (forced serialization). That modification is the technique, not window dressing."
- [ ] `2026-07-30_theater-copurchase-demand-reversal` — Demand Reversal Detection in Community Theater Box Office via Isolation Forest on Ticket-Category Co-Purchase Anomalies
      self: "This forces isolation forest to operate on a compositional simplex (ticket-category mix per transaction) where the anomaly isn't high/low volume but structural *change in co-purchase ratios*, requiring the detector to see that same-day transactions suddenly decouple from historical pairing patterns—real theater data would show this during cast changes, venue reconfiguration, or dynamic pricing, making it a genuine domain friction point."
- [ ] `2026-07-31_beekeeper-entailment-drift-nlp` — Entailment Drift in Beekeeper Inspection Logs: Detecting Colony Decline via NLP Premise-Hypothesis Collapse
      self: "This forces NLP beyond sentiment/classification: logs are treated as mini-arguments where premise (observations) should entail hypothesis (health status), and that relationship degrades under colony stress—a genuine mismatch between log structure (causal narratives) and standard NLP tasks (labels or bags-of-words)."

## Week of 2026-07-20

- [ ] `2026-07-23_farmers-market-vendor-adjacency-causal` — Vendor Layout Spillover: Does Adjacency Confound the Market-Day Effect on Sales?
      self: "Has legs: the tension between observational causal inference (built for independent units) and spatial-network data (where treatment and confounders are entangled by geography) forces a genuine methodological compromise that exposes the limits of covariate adjustment."
- [ ] `2026-07-24_brood-temporal-hypergraph-collapse` — Temporal Hypergraph Collapse: Detecting Brood-Pattern Asymmetries via Time-Windowed Subgraph Isomorphism
      self: "The seam is real: standard network analysis treats contacts as undirected snapshots, but brood disease propagation is causal (nurse→brood, not vice versa) and time-indexed; we must engineer directed lagged edges from temporal contact sequences. This forces explicit handling of the temporal asymmetry rather than applying off-the-shelf centrality."
- [ ] `2026-07-26_thrift-clustering-under-description-noise` — Thrift Price Clusters Under Measurement Error: Can k-means recover categories when item descriptions are noisy proxies for condition?
      self: "This has legs because the core tension (clustering assumes latent clean structure, but thrift pricing is driven by messy grading systems) forces a methodological compromise: either you admit the cluster labels don't align with ground truth, or you ask what structure survives *despite* noise—which is honest and uncommon in tutorial examples."
