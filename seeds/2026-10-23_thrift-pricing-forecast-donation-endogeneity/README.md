# Thrift Store Pricing Forecast Collapse Under Donation-Velocity Endogeneity

## Research Question

**Can ARIMA-style forecasting detect hidden thrift-store pricing regimes by measuring where prediction intervals collapse when donation volume endogenously triggers unobserved inventory-management rule shifts?**

## The Non-Obvious Angle

Standard time-series forecasting assumes stationarity: the underlying data-generating process doesn't change. But thrift stores often endogenously *adjust their pricing logic* in response to donation inflows (e.g., lower prices when storage is full). This breaks the stationarity assumption not due to randomness, but due to **strategic behavior that violates the technique's core assumption**.

The interesting bit: use forecast *failure patterns* (widening prediction intervals, regime switches in residuals) as a **diagnostic for when pricing logic itself changed**, rather than treating wide intervals as noise.

## Data

**Synthetic**, generated inline. Structure:
- 120 weeks of weekly data
- **Donations**: exogenous trend + noise, ~50–70 items/week
- **Pricing rule**: endogenous regime switch
  - Low-donation weeks: avg price = $8.00
  - High-donation weeks (>70th percentile): avg price = $5.50 (storage pressure)
  - Both regimes: ±$0.40 noise

This structure is realistic: real thrift stores do lower prices when donations surge to manage inventory turnover.

## Method

1. **Train** a simple linear trend model on weeks 0–79 (mostly low-donation regime)
2. **Forecast** weeks 80–119 (mixed, includes donation spike ~week 95+)
3. **Measure** prediction-interval width (±2σ prediction error) as a rolling diagnostic
4. **Detect** where PI collapses (width anomalies) coincide with regime switches

The key insight: when the underlying pricing rule changes, forecast error increases systematically, widening the interval. This is *not random noise*—it's evidence of unobserved state change.

## Limitations

- **Synthetic data**: real thrift pricing is more complex (item category, season, brand, etc.)
- **Simple model**: using linear regression instead of full ARIMA (same principle applies)
- **No ground truth**: we know the regime in this seed, but in reality you'd infer it from the forecast pattern alone
- **Small sample**: 120 weeks is toy-scale; real forecasting would use years of data

## What This Demonstrates

1. Forecasting failures can signal **hidden behavioral/operational regime changes**
2. The technique isn't broken—it's working correctly by failing when assumptions are violated
3. Using **prediction-interval width as a diagnostic** converts forecasting into an anomaly detector for endogenous rule shifts

---

*This is an auto-generated seed from the data-analyst project farm. Code is <200 lines, runs standalone with no APIs or paid data.*
