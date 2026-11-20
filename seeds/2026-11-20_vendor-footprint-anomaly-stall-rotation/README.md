# Vendor Temporal Footprint Clustering Under Stall-Rotation Endogeneity

## Question

Can isolation-forest anomaly detection on vendor **co-absence patterns** (rather than sales/visits) reveal hidden operational constraints (shared equipment, power, water) that force vendors into incompatible spatial/temporal configurations at farmers markets?

## The Non-Obvious Angle

Standard anomaly detection on farmers market data targets rare sales events or unusual purchase volumes. This seed **inverts** the technique:

1. Instead of detecting *rare vendor visits*, detect *rare co-presences*.
2. Build a co-absence matrix: for each vendor pair, count how many weeks they were NOT simultaneously present.
3. Extract statistical features from each vendor's co-absence distribution with all others.
4. Run isolation forest on these co-absence features to find vendors with unusual temporal footprints.

**The seam**: Stall rotation (the market rotates which vendors can attend each week) creates endogenous unavailability that confounds naive anomaly detection. By working backward from *absence patterns*, we can infer operational dependencies that wouldn't appear in raw visit/sales data.

## Real vs. Synthetic

**Synthetic dataset** (generated inline): 
- 12 weeks, 20 vendors, 8 stalls per week.
- Ground-truth constraints embedded:
  - Vendors 0,1,2 share equipment → can't all be present simultaneously.
  - Vendors 3,4,5 share water line → asymmetric co-absence.
  - Vendor 15 has power constraint → only specific stalls work.

**Why synthetic**: Public farmers market vendor attendance data is rare and usually lacks operational metadata. This synthetic structure ensures the signal (vendor operational conflicts) actually exists in the data.

## Limitation

- **Weak signal in noise**: Without strong ground-truth constraints, co-absence patterns can be dominated by random scheduling variation or market capacity, not operational conflicts.
- **Requires domain knowledge to interpret**: Anomalies must be validated against real vendor interviews or stall-assignment logs.
- **Not a causal inference**: This finds correlated absence patterns, not proof of causation.
- **Scalability**: Matrix construction is O(V²) in vendors; feasible up to ~100 vendors.

## How to Run

```bash
python main.py
```

Outputs anomalous vendors and their co-absence feature profiles.

---

*This is an auto-generated exploratory seed. Most seeds are discarded. This one trades off interpretability for a genuinely inverted anomaly-detection angle.*
