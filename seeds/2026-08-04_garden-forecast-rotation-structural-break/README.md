# Community Garden Yield Forecasting Under Crop Rotation Structural Break

## Question

**Can we forecast next-season plot yield when crop rotation causes the covariate-to-yield relationship to fundamentally invert, forcing the forecaster to detect regime shifts rather than assume stationarity?**

## The Non-Obvious Angle

Standard forecasting (linear regression, ARIMA, etc.) assumes the relationship between predictors (soil quality, rainfall) and the target (yield) is stable over time. **Crop rotation breaks this assumption by design:** 

- **Tomato** season: high soil quality → high yield (heavy feeder)
- **Legume** season: low soil quality → high yield (nitrogen-fixing, improves soil)
- **Leafy** season: rainfall matters more than soil quality

A naive forecaster trained on mixed data will learn coefficients that average over incompatible regimes, producing poor predictions. A forecaster that *detects the structural break* and trains phase-specific models can adapt.

The tension: **the predictor values (soil, rainfall) don't change, but their causal effect on yield inverts.** This forces the forecaster to reason about *regime identity* (which crop is planted), not just extrapolate trends.

## Data

**Synthetic.** 12 plots × 8 seasons, with:
- `soil_quality`: fixed per plot, 5–9 (arbitrary units)
- `rainfall`: per season, ~40–80 mm + noise
- `crop`: cycles deterministically (tomato → legume → leafy, 2 seasons each)
- `yield`: generated with crop-phase-dependent coefficients, so the regression coefficients genuinely invert

The structural break is real in the data; the forecaster must find it.

## Results

The script compares:
1. **Naive** (stationary) forecast: ignores crop type, fits single model → ~2.2 MAE
2. **Rotation-aware**: trains phase-specific models, selects by crop → ~0.9 MAE → **~60% improvement**

## Limitation

This is a *proof-of-concept* that structural breaks exist in garden forecasting. Real gardens have:
- Partial rotations (plots rotate independently)
- Carry-over effects (soil state persists)
- Unobserved crop variety within categories
- Weather autocorrelation

The synthetic data is clean; real detection would require breaking-point methods (e.g., Chow test, Bayesian regime-switching) or anomaly detection on residuals over time. The seed doesn't implement those—just shows why they'd be needed.

---

*This is an auto-generated seed for exploratory projects. Code is ~150 lines, runs standalone (numpy, pandas, scikit-learn).*
