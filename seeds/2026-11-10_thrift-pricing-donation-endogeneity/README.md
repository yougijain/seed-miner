# Thrift Store Pricing Regime Detection via Donation Velocity Endogeneity

## Question
Can anomaly detection on thrift-store daily pricing reveal hidden inventory-management regime switches that are *endogenously triggered by donation-volume shocks*?

## Why Non-Obvious
Standard anomaly detection (Isolation Forest, LOF, etc.) assumes anomalies are **independent** deviations from a stationary process. But thrift-store pricing is driven by an unobserved rule:

**When donation volume crashes, management switches to a clearance/aggressive-discounting regime.**

This means:
- Anomalies cluster temporally (violation of independence assumption)
- The anomaly-generating process is non-stationary by design
- Detecting *individual* anomalies fails; detecting **anomaly clustering patterns** succeeds

The seam: anomaly detection must be **inverted**—instead of using individual anomaly scores, cluster the anomaly *density* over time to infer hidden regime transitions.

## Data
**Synthetic** (structured to make the signal real):
- 180 days of thrift-store pricing simulations
- 5 hidden donation-volume shocks trigger regime switches
- Normal regime: higher avg price (~7.50), lower variance
- Clearance regime: lower avg price (~3.00), higher variance
- Signal: anomaly *clustering* peaks at regime transitions

## Limitation
This is synthetic data tuned to show the phenomenon. Real thrift pricing is influenced by:
- Item category (furniture vs. books)
- Store location and donor demographics
- Seasonal donation cycles
- Day-of-week effects

Real data would require category-specific regime detection and multivariate donation tracking.

## How It Works
1. Fit Isolation Forest on (donation_volume, avg_daily_price)
2. Compute anomaly density in sliding 7-day windows
3. Find peaks in anomaly density → these mark regime transitions
4. Compare detected regimes to true regime labels

## Key Output
- Detected regime transitions via anomaly clustering
- Accuracy of recovered regimes vs. ground truth
- Shows why standard anomaly *scores* fail but anomaly *patterns* succeed

---
*Auto-generated data-analyst project seed. Exploratory only.*
