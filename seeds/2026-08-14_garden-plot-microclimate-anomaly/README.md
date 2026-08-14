# Garden Plot Microclimate Drift: Anomaly Detection via Temporal Heteroskedasticity

## Question
Can we detect problematic garden plots by identifying plots where *yield variance is itself becoming unstable* over the growing season, rather than simply looking for low average yield?

## Angle (Non-Obvious Twist)
Standard anomaly detection on garden plots would flag plots with unusually low or high mean yield. This seed flips the signal: it detects plots where the **within-season variance pattern changes**, suggesting microclimate instability, soil degradation, or emerging pests. This forces Isolation Forest to work on a derived feature (rolling variance statistics) rather than raw measurements, and exposes the algorithm's core weakness: it assumes points are independent, but garden plots are spatially embedded—neighbors affect each other.

## Data
**Synthetic.** 40 plots over 12 weeks. Ground truth:
- Plots 1–35: stable yield + constant noise (normal)
- Plots 36–38: destabilizing variance (grows mid-season, indicates soil/microclimate drift)
- Plots 39–40: sudden variance spike mid-season (pest outbreak or disease)

The synthetic data is structured to make the signal real: variance instability actually exists and should be detectable.

## Method
1. Compute rolling 3-week variance for each plot's weekly yields.
2. Aggregate into features: mean variance, variance-of-variance, coefficient of variation, max variance.
3. Apply Isolation Forest (contamination=0.12) to flag plots with anomalous variance patterns.

## Key Limitation
**Isolation Forest violates a core assumption of this domain:** plots are NOT independent. Adjacent plots share water, pests, and soil conditions. A true solution would use spatial-aware anomaly detection (e.g., Local Outlier Factor with spatial kernels, or Markov random fields), but ISO Forest is simpler and exposes why naive techniques fail here.

## Files
- `main.py`: Data generation, feature engineering, anomaly detection, reporting.
- `README.md`: This file.

---
*Auto-generated seed. Not production-ready. Intended as exploratory scratch work for discovery of real project angles.*
