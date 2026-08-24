# Board Game Café Rental Duration Forecasting Under Inventory Feedback Loops

## Question

Can ARIMA forecast per-game rental duration when inventory availability of complementary games creates **endogenous feedback**—where rental duration directly causes next-day inventory (short rentals → faster replenishment), which in turn influences future rental decisions—forcing the forecaster to detect and model circular causality instead of treating covariates as exogenous?

## Why This Matters

Standard time-series forecasting assumes covariates (like inventory) are **exogenous**: independent causes that drive the outcome. But in a board game café, game availability is a **consequence** of recent rental behavior, not an independent driver. When we observe that "heavy inventory availability correlates with longer rental durations," it's not that availability *causes* longer rentals—it's that *short recent rentals freed up copies faster*, creating the appearance of causality in the wrong direction.

This breaks the assumption underlying ARIMA with exogenous covariates (ARIMAX), producing biased coefficient estimates and misleading confidence intervals.

## Data

**Synthetic** (180 days of rental data for a single popular game).

- `rental_duration_min`: How long the game was rented (in minutes).
- `inventory_available`: Number of copies on shelf at day start (0–5).

The data is generated with an explicit feedback loop:
1. Duration on day *t* is depressed by low inventory (scarcity → shorter rentals).
2. Inventory on day *t+1* depends on day *t* duration: shorter rentals → faster throughput → inventory replenishes.

**Why synthetic:** This structure is hard to ground-truth in real café data (inventory logs rarely link back to rental duration causally), and the feedback loop must be precisely controlled to demonstrate the forecasting bias.

## How Forecasting Breaks

1. **Naive ARIMA** (ignoring inventory entirely) misses a real driver and may have high residual autocorrelation.
2. **ARIMAX** (treating inventory as exogenous) produces a biased coefficient for inventory and invalid standard errors, because the true causal arrow runs from duration to inventory, not the reverse.
3. **Detection:** ACF will decay slowly (spurious autocorrelation from endogeneity), and the exogenous coefficient will likely reverse sign or collapse under instrumental-variable or lagged-difference specification.

## What the Code Does

- Generates 180 days of synthetic rental + inventory data with known feedback structure.
- Fits both naive ARIMA and ARIMAX models.
- Compares their forecast divergence.
- Plots ACF/PACF to expose endogeneity signature (slow ACF decay).
- Documents where standard forecasting assumptions fail.

## Limitation

This is a **proof-of-concept** that endogenous covariates bias forecasting in this domain. Real café data would require:
- Simultaneous rental logs + inventory snapshots (rare in practice).
- Deconfounding via instrumental variables or Granger causality tests.
- Validation against held-out rental windows.

The synthetic data is designed to make the feedback loop obvious; a real dataset might have messier causal structure and multiple confounders (day-of-week effects, game popularity, seasonality).

---

**Auto-generated seed** for data-analyst project exploration. This is scratch work; most seeds are discarded.
