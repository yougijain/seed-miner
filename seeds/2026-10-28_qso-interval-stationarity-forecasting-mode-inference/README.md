# QSO Interval Stationarity Collapse as Propagation Mode Detector

## Question
Can we detect hidden operator behavior mode switches (contest-sprint vs. ragchew) in amateur radio contest logs by watching where ARIMA forecasting of inter-QSO intervals breaks down, without explicit mode labels?

## The Non-Obvious Angle
Standard forecasting applies *forward*: given past data, predict future values. Here, we flip it: **forecasting *failure* is the signal**. When inter-QSO intervals shift from one regime (e.g., long ragchew pauses, ~180s) to another (e.g., short contest sprints, ~35s), the underlying time series becomes non-stationary. An ARIMA model fit on stationary data will produce wide prediction intervals and large residuals when it encounters data from a different regime.

The technique doesn't forecast the mode; it detects mode switches via the *collapse of forecasting stability*. This is what makes the seam real: ARIMA was built for stationary series, but radio propagation and operator behavior create hidden regime shifts that invalidate stationarity assumptions in ways that standard forecasting diagnostics can surface.

## Dataset
**Synthetic.** Generated 200 QSOs with:
- QSOs 0–80: Ragchew mode (long pauses, ~180s intervals, low variance).
- QSOs 80–120: Transition zone (propagation degrades, intervals drift upward).
- QSOs 120–200: Contest-sprint mode (short intervals, ~35s, higher variance).

True mode labels are included for validation but not used by the forecasting algorithm.

## Method
1. **Rolling ARIMA model**: Fit a simple AR(5) model on a 20-QSO sliding window; forecast next interval and compute prediction interval width.
2. **Stationarity proxy**: Compute rolling standard deviation of forecast residuals. High residual variance indicates the model is struggling—a sign of regime change.
3. **Shock detection**: Flag QSOs where residual std exceeds threshold (~35s), signaling forecasting breakdown.
4. **Interpretation**: Shocks cluster near the true mode-transition boundary (QSO ~110), correctly inferring the hidden regime switch.

## Limitations
- **Synthetic data**: Real QSO logs have propagation dynamics, CQ pileups, and operator fatigue that we simplified.
- **No causal structure**: We detect *that* a regime shift occurred, not *why* (propagation, fatigue, activity levels, etc.).
- **Threshold tuning**: The residual std threshold is manual; a real implementation would use data-driven selection.
- **Single operator**: Contest logs often have many operators; this treats a log as a single stream.

## Real-World Applicability
Amateur radio contest logs are publicly archived (e.g., ARRL, CQ WW). Applying this to real logs could:
- Detect when a single operator switches activity modes mid-contest (rest vs. sprint).
- Infer propagation regime changes by seeing where forecasting breaks for *all* operators simultaneously.
- Flag anomalies (equipment failure, operator fatigue) as sudden stationarity shocks.

## Status
**Auto-generated seed**. This explores a genuine tension: forecasting is built for stationary data, but radio contests are inherently non-stationary. The seam—where ARIMA's assumptions break—is where the insight lives.
