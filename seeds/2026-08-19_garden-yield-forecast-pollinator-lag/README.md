# Garden Plot Yield Forecasting Under Variable Pollinator Lag

## Question

Can ARIMA forecast garden plot yields when the key causal driver (neighbor bloom timing) affects yield through a **variable, domain-specific lag**? How does time-varying cross-plot dependency break standard forecasting assumptions?

## The Non-Obvious Angle

Standard forecasting techniques (ARIMA, exponential smoothing, etc.) assume:
- Exogenous covariates have **fixed lead-lag structure** (e.g., weather affects yield 2 weeks later, always).
- The causal relationship is **stationary** across time.

But community gardens violate both:
1. **Pollinator networks are dynamic.** Plot B's bloom affects Plot A's yield only when pollinators actively visit both—which depends on seasonal availability, weather, and competing resources. The lag is not fixed.
2. **Crop rotation and staggered planting create regime shifts.** Early-season plots rely on wild pollinators (long lag); mid-season plots benefit from established hives (short lag); late-season plots may not benefit at all.
3. **Network effects are cross-plot and nonlinear.** ARIMA treats each plot independently; it can't natively model "Plot A's yield depends on Plot B's timing."

## The Tension

The domain forces the technique to **detect and adapt** rather than assume. We apply rolling cross-correlation to identify lag *within* the ARIMA fitting pipeline, converting a forecasting problem into a structural-change detection problem first.

## Data

**Synthetic.** Generated inline in `main.py`:
- 24 weeks of data across 3 plots.
- Plot A & B are neighbors; Plot B's bloom drives Plot A's yield with **variable lag**: lag=1 (weeks 1–8), lag=2 (weeks 9–16), lag=3 (weeks 17–24).
- Plot C is independent (control).
- Bloom intensity and yield include trend and noise to mimic real agricultural data.

Why synthetic? Real community garden data with precise bloom timestamps + yield measurements + pollinator logs is rare. The synthetic structure *encodes the real problem*: variable cross-plot lag.

## Limitation

This is a **single mechanism proof-of-concept**, not a production forecaster. Real gardens have:
- Multiple pollinator types with different lag profiles.
- Weather confounders (rain suppresses pollinators).
- Unmeasured plots affecting the network.
- Nonlinear, threshold-based effects (pollinator density matters, not just timing).

The code demonstrates the *adaptation required* to use ARIMA on cross-plot dependencies, not a complete solution.

## How to Run

```bash
python main.py
```

Output shows:
1. Detected lags via rolling cross-correlation (Step 1).
2. Naive ARIMA forecast (Step 2, ignores lag structure).
3. Lag-aware ARIMA forecast using exogenous bloom covariate (Step 3).
4. AIC comparison showing whether incorporating lag improves fit.

## Auto-Generated Seed

This project is an auto-generated exploratory seed. Treat as scratch work. The intent is to flag the tension between forecasting assumptions and garden-network structure.
