# Plot Yield Forecasting Under Temporal Planting Endogeneity

## Question
Can ARIMA forecasting residual patterns reveal which gardeners endogenously adapt their planting dates in response to prior-season yields, rather than following exogenous (calendar-driven) planting schedules?

## Why This Matters for Forecasting
Standard ARIMA forecasting assumes:
1. Time series is stationary (or can be differenced to stationarity)
2. Regressors (planting date) are exogenous to the outcome (yield)

When gardeners *adapt* planting date based on prior yields, the regressor becomes endogenous. This violates ARIMA's exogeneity assumption and should create a **systematic, predictable residual pattern** that ARIMA cannot explain away—a signal that forecasting is failing for a specific reason (adaptive behavior).

## The Non-Obvious Angle
Rather than forecasting yields directly, we use **ARIMA residual clustering** as a diagnostic: gardeners with high systematic residual trends (residuals that lean one way or show autocorrelation) are more likely to be endogenous actors, because the model is consistently missing a feedback loop. We then validate by checking whether the "high-residual-trend" cluster contains more true endogenous gardeners.

## Data
**Synthetic.** 20 gardeners × 36 months (3 years):
- 8 gardeners adapt planting date to prior-season yield (±5 days per 10-unit deviation from baseline 50 yield)
- 12 gardeners plant on fixed calendar
- Yield influenced by planting date + random noise
- No real garden or weather data used

## Limitation
This is a proof-of-concept showing how forecast residuals can encode behavior endogeneity. In real data, you would need:
- Actual planting-date records (often unavailable in community gardens)
- Confounding variables (weather, soil, pest pressure) that also affect both planting timing and yield
- Larger sample size to separate signal from noise

The synthetic setup is *designed* to make endogeneity detectable; real-world signal may be much weaker.

## How to Run
```bash
python main.py
```
Output shows cluster profiles and the % of endogenous gardeners in each cluster. If clustering works, one cluster should have ~70%+ endogenous gardeners, the other ~0%.

---
*Auto-generated seed; data is synthetic to enforce the endogeneity signal.*
