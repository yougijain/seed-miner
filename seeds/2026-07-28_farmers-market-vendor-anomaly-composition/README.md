# Farmers Market Vendor Anomaly Detection via Compositional Data

## Question

Can isolation forest detect vendors experiencing revenue-allocation instability (budget crisis across product categories) by applying additive log-ratio (ALR) transformation to enforce the compositional constraint that category shares sum to 1?

## The Non-Obvious Angle

Standard anomaly detection (including iso-forest) uses Euclidean distance, which is invalid on the simplex (proportions summing to 1). A vendor with composition [0.5, 0.2, 0.2, 0.05, 0.05] and [0.51, 0.19, 0.19, 0.06, 0.05] are nearly identical, but raw Euclidean distance treats them as comparable to [0.1, 0.1, 0.1, 0.1, 0.6]—both distance 0.01, which is nonsense on the simplex.

This seed applies iso-forest to the ALR-transformed space (log-ratios relative to a reference category), which projects the simplex into ℝ^(k-1) while preserving distance semantics. The result: detecting when vendors' typical budget splits *destabilize*, not just when total revenue dips.

## Data

**Synthetic.** 40 vendors × 8 weeks × 5 product categories (produce, dairy, prepared, flowers, eggs).
- Baseline: each vendor has a stable compositional preference, sampled from a Dirichlet.
- Signal: vendors 35–39 undergo sudden allocation shock at week 4 (e.g., dairy supplier fails, forcing rapid reallocation).
- Noise: normal weekly variation (Dirichlet concentration ≈ 10).

## Method

1. Compute ALR features: `log(category_i / reference)` for each category except eggs.
2. Standardize ALR features.
3. Fit iso-forest with 12.5% contamination.
4. Inspect which vendor IDs and weeks are flagged.

## Limitation

This is proof-of-concept on synthetic data where the signal is strong and clean. Real farmers market data (if available) would have:
- Seasonal trends (spring vs. fall allocation shifts).
- Confounders (holidays, weather, batch failures).
- Vendor entry/exit and day-to-day sampling variation.

The ALR transform itself is standard, but applying it to force iso-forest into compositional space is the non-trivial design choice here.

## Running

```bash
python main.py
```

Expects to flag vendors 35–39 as anomalous in weeks 4–7, demonstrating that anomaly detection on the simplex can detect allocation instability without relying on total revenue thresholds.
