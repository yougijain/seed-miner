# QSO Propagation Mode Inference via ARIMA Prediction Mismatch Clustering

## Question
Can ARIMA forecasting errors on QSO (amateur radio contact) counts reveal hidden operator mode-switching (rapid contest-style vs. slow ragchew-style QSOs) when propagation conditions endogenously trigger unobserved behavioral changes that break the stationarity assumption?

## The Non-Obvious Angle

Standard forecasting treats time-series prediction as a *technique goal*. Here, we reverse that: **prediction failure becomes the diagnosis tool**.

Operators in amateur radio contests switch between two modes:
- **Contest mode**: rapid exchanges (~30 QSOs/hour), triggered by good band propagation
- **Ragchew mode**: long conversations (~8 QSOs/hour), fallback when bands are poor

Neither the operator's mode nor the underlying propagation quality is logged. Only QSO counts are observed.

ARIMA assumes stationarity: it expects mean, variance, and autocorrelation to remain constant. When an operator switches modes, the underlying distribution shifts *without warning*, violating this assumption. This produces **clusters of large forecast residuals** at transition points.

The key: we use ARIMA not to forecast (it fails), but to detect *where it fails*. Residual clustering separates high-error regions (mode switches) from low-error regions (within-mode stability).

## Data

**Synthetic** (no real QSO logs are publicly time-stamped at minute/hour granularity with mode metadata).

The dataset is generated to have:
- A hidden "band quality" signal (sinusoidal + noise)
- Mode-switching tied to band quality (unobserved)
- QSO counts drawn from mode-specific distributions (Poisson λ=30 or λ=8)
- 200 hours of simulated activity with ~5-10 mode transitions

**Real-world limitation**: In practice, inferring mode-switching from residuals alone requires validation (e.g., operator surveys, or hand-coding a sample). This seed assumes the residual signature is strong enough to separate modes; in real data, other confounders (time-of-day, seasonal propagation, operator skill drift) could add noise.

## Output

The script fits ARIMA(1,1,1) and clusters absolute residuals into 2 groups (high, low). It reports:
- How many true mode transitions are *caught* by the high-residual cluster
- Model AIC and residual statistics
- A sample window showing QSO counts, forecasts, residuals, and inferred clusters

## Limitations

1. **Synthetic data**: Signal is clear and separable. Real logs would have more confounding.
2. **Assumes binary modes**: Many contests have >2 distinct behaviors (mobile, DX-hunting, etc.).
3. **Requires cluster validation**: Residual clustering alone doesn't prove mode-switching; external labels are needed.
4. **ARIMA order fixed**: Tuning ARIMA(p,d,q) could improve fit and muddy the residual signal.

---

*This is an auto-generated seed from a data-analyst project farm. The goal is to identify non-obvious pairings of technique × domain where the seams show.*
