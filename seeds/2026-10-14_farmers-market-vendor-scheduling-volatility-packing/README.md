# Farmers Market Vendor Scheduling via Demand Volatility Bin-Packing Inversion

## Question
**Can bin-packing constraint violations reveal hidden vendor dependencies (shared equipment, weather sensitivity, customer overlap) without explicit co-registration data?**

## Non-Obvious Angle
Standard bin-packing assumes fixed weights. Here, vendor stall demand is **volatile and correlated**. When you assign vendors to fixed stalls and their demand patterns cause repeated packing violations, those violations encode information about which vendors are "too close" due to hidden dependencies. By inverting the constraint-violation pattern, you can infer affinity without ever seeing vendor metadata.

## Data: Synthetic
**Why synthetic:** Real farmers market demand data is proprietary and difficult to access in structured form. The synthetic dataset is deliberately constructed to contain the *right anomaly structure*:
- 8 vendors, 26 weeks of demand observations.
- V0-V1: Correlated weather sensitivity (flowers).
- V2-V3: Shared equipment (cold storage).
- V4-V5: Customer cannibalization (both sell prepared lunch).
- Underlying correlation injected as demand shocks.

## What the Code Does
1. Generates vendor weekly demand with hidden correlations.
2. Performs greedy bin-packing for fixed stall assignments (initial, week 0).
3. Simulates 26 weeks of demand; counts stall-capacity constraint violations.
4. **Inverts violations:** For each pair of vendors assigned to the same stall, counts how many weeks their combined demand exceeds capacity.
5. Ranks vendor pairs by co-violation frequency; compares to ground-truth correlations.

## Expected Result
**Yes:** The top co-violation pairs should be (V0, V1), (V2, V3), (V4, V5)—matching ground truth. This proves the technique works.

## Limitation
This is a proof-of-concept on synthetic data with **known, simple correlations**. Real farmers markets have more complex dependencies (cross-vendor customer flows, equipment sharing networks, time-of-harvest variability). The technique would need:
- Real demand time series (collected weekly via vendor surveys or transaction logs).
- Multiple seasons to detect seasonal co-patterns.
- Validation against actual stall-rotation decisions made by market managers.

## Technique Insight
**Bin-packing under volatility becomes an inference tool:** Standard optimization assumes you know weights. Here, weight *uncertainty* (demand volatility) and its *structure* (correlation) become the signal. The optimizer doesn't solve scheduling; it reveals latent structure by failing in informative ways.

---
*This is an auto-generated seed from a data-analyst project farm.*
