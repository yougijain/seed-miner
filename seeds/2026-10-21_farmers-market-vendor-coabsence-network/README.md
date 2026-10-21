# Vendor Substitutability via Customer Co-Absence Graphs

## Research Question
**Which farmers-market vendors are functionally substitutable (interchangeable) from the customer's perspective?**

## The Non-Obvious Angle

Standard vendor clustering uses product attributes ("lettuce stand" vs. "herb grower"). This approach builds a **co-visitation graph** where edges represent **customer visits spanning multiple vendors**. The key inversion: substitutability is *revealed through transaction bundles*, not vendor metadata.

The technique then detects substitutability clusters by:
1. Weighting edges by co-visit frequency (customers voting with their feet)
2. Using community detection on co-visitation patterns (standard modularity clustering)
3. **Measuring network robustness under vendor removal** — high-substitutability vendors cause dramatic cohesion loss when absent

## Why This Matters

- **Market structure**: reveals implicit vendor groupings customers actually recognize
- **Pricing power**: vendors in sparse clusters can raise prices; substitutable ones cannot
- **Layout optimization**: physically group substitutable vendors to concentrate traffic
- **Vendor churn risk**: substitutable vendors are more at risk if a competitor enters

## Data

**Synthetic** (see `main.py`). 

Structure: customer visits with per-visit vendor lists. Two clusters:
- **Produce group** (lettuce, tomato, herbs) — cross-visit frequently
- **Dairy group** (cheese, bread, butter) — cross-visit frequently
- **Cross-group** visits are rare → reveals substitutability boundaries

Real data would come from:
- POS transaction logs (if vendors share a system)
- Customer survey data ("who did you visit?")
- Receipt analysis ("which vendors on the same receipt?")

Synthetic data exists to demonstrate the *core inversion*: graph structure is inferred from transaction bundles, not vendor similarity.

## Limitations

1. **No causality**: co-visitation ≠ substitutability; customers might visit both intentionally (complementary preference).
2. **Synthetic data**: no real customer behavior; result is a proof-of-concept.
3. **Temporal collapse**: treats all visits equally; ignores seasonal or event-driven clustering.
4. **Single-attribute clustering**: doesn't integrate price, quality, distance as competing substitutability drivers.

## Interpretation

- High modularity clusters = clear substitutability groups
- High node-removal cohesion loss = hub/critical vendors (less substitutable; single-vendor reliance)
- Low removal impact = niche vendors (not interchangeable)

---

*Auto-generated seed for data-analyst exploration. Most seeds are discarded. Honest assessment: the co-absence graph inversion is real; the synthetic data is toy-scale.*
