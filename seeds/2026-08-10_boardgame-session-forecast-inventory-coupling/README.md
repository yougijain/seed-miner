# Board Game Café Session Duration Forecasting Under Inventory Coupling

## Question
Can we detect and correct for **endogenous feedback** when forecasting multiplayer game session duration in inventory-constrained domains, where the inventory of complementary games both influences *and is influenced by* session length?

## The Non-Obvious Angle

Standard forecasting (ARIMA, exponential smoothing) assumes regressors are **exogenous**: they influence the target but are not influenced by it. 

In a board game café:
- **Standard model assumption**: Inventory → Session Duration (one-way)
- **Reality**: Inventory ↔ Session Duration (two-way feedback)
  - Long sessions lock up inventory → future inventory drops → constrains future sessions
  - Short sessions free inventory → inventory rises → enables longer future sessions

This **endogeneity** violates the exogeneity assumption of ARIMAX and causes:
1. **Misspecified parameter estimates** (biased coefficients)
2. **Inflated forecast error** (model ignores the structural loop)

## Data

**Synthetic** (180 days, board game café inventory + session logs).
- `session_minutes`: duration of multiplayer sessions (45–180 min, trending upward with weekly seasonality)
- `complementary_game_inventory`: count of games available for extended play (20–100, inversely correlated with session length due to in-use inventory)

**Why synthetic?** Real café data requires transaction logs and inventory tracking systems. The synthetic data is structured to exhibit genuine endogenous coupling: session duration is mechanically reduced when inventory is low (can't play long games), and inventory falls when sessions are long (games in use).

## Approach

1. **Granger Causality Test**: Detect whether inventory Granger-causes session duration (necessary but not sufficient for endogeneity, but indicates feedback).
2. **Naive ARIMAX**: Fit a standard ARIMAX model treating inventory as exogenous; measure RMSE.
3. **ARIMAX on Differenced Inventory**: Difference inventory to remove the trend/feedback loop; refit ARIMAX; compare RMSE.
4. **Interpretation**: If differenced RMSE is lower, endogeneity was inflating error in the raw model.

## Limitations

- **Granger causality ≠ true causality**: It only tests whether past inventory predicts duration beyond its own history. True identification of endogeneity requires exclusion restrictions or instrumental variables (not attempted here).
- **Synthetic data**: The coupling is exact and linear. Real coupling would be noisier and nonlinear.
- **Small sample**: 180 days is minimal for robust ARIMA estimation; real forecasts need 2–5 years.
- **No IV/structural approach**: A proper solution would use instrumental variables or GMM estimation, not just differencing.

## Key Insight

Forecasting techniques fail silently when domain structure violates their assumptions. In constrained-inventory domains, **the regressor is endogenous by design**, not accidental. Detecting this (via Granger + residual diagnostics) and correcting it (via differencing or structural estimation) is necessary for valid inference.

---

*This is an auto-generated seed for a public project farm. It is exploratory; most seeds are discarded.*
