import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic: 3 garden plots over 24 weeks.
# Plot A & B are neighbors; B's bloom timing affects A's yield with a VARIABLE lag.
# Plot C is isolated.
n_weeks = 24

# Plot B: bloom intensity (0-1)
plot_b_bloom = np.sin(np.linspace(0, 4*np.pi, n_weeks)) * 0.4 + 0.5
plot_b_bloom = np.clip(plot_b_bloom, 0, 1)

# Plot A: yield depends on its own trend + Plot B's bloom with LAG.
# The lag varies: weeks 1-8 lag=1, weeks 9-16 lag=2, weeks 17-24 lag=3.
plot_a_yield = np.zeros(n_weeks)
plot_a_yield[0] = 5.0
for t in range(1, n_weeks):
    lag = 1 if t < 8 else (2 if t < 16 else 3)
    bloom_contrib = plot_b_bloom[t - lag] * 3.0 if t >= lag else 0
    trend = 0.1 * t
    noise = np.random.normal(0, 0.3)
    plot_a_yield[t] = plot_a_yield[t-1] * 0.9 + bloom_contrib + trend + noise

# Plot C: independent trend, no neighbor effect.
plot_c_yield = 3.0 + 0.05 * np.arange(n_weeks) + np.random.normal(0, 0.4, n_weeks)

df = pd.DataFrame({
    'week': np.arange(n_weeks),
    'plot_a_yield': plot_a_yield,
    'plot_b_bloom': plot_b_bloom,
    'plot_c_yield': plot_c_yield
})

print("="*70)
print("GARDEN PLOT YIELD FORECAST WITH VARIABLE POLLINATOR LAG")
print("="*70)
print(f"\nData shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# Step 1: Detect variable lag via rolling cross-correlation.
print("\n" + "="*70)
print("STEP 1: Detect Variable Lag via Rolling Cross-Correlation")
print("="*70)

window = 7
lags_detected = []

for start in range(0, n_weeks - window, 3):
    end = start + window
    a_window = df['plot_a_yield'].iloc[start:end].values
    b_window = df['plot_b_bloom'].iloc[start:end].values
    
    max_corr = -np.inf
    best_lag = 0
    
    for lag in range(1, 4):
        if start + lag < n_weeks:
            b_lagged = df['plot_b_bloom'].iloc[start+lag:end+lag].values
            if len(b_lagged) == len(a_window):
                corr = np.corrcoef(a_window, b_lagged)[0, 1]
                if corr > max_corr:
                    max_corr = corr
                    best_lag = lag
    
    lags_detected.append((start, best_lag, max_corr))
    if max_corr > 0.3:
        print(f"Weeks {start:2d}–{end:2d}: lag={best_lag}, corr={max_corr:.3f}")

# Step 2: Fit ARIMA to Plot A (ignoring the lag structure; see the forecast error).
print("\n" + "="*70)
print("STEP 2: Naive ARIMA (Ignoring Variable Lag)")
print("="*70)

try:
    model_naive = ARIMA(df['plot_a_yield'], order=(1, 1, 1))
    result_naive = model_naive.fit()
    forecast_naive = result_naive.get_forecast(steps=5)
    print(f"\nNaive forecast (next 5 weeks): {forecast_naive.predicted_mean.values}")
    print(f"AIC: {result_naive.aic:.2f}")
except Exception as e:
    print(f"Naive ARIMA fit failed: {e}")
    forecast_naive = None

# Step 3: Lag-aware forecast using lagged bloom as exogenous variable.
print("\n" + "="*70)
print("STEP 3: ARIMA with Exogenous Lag-Adjusted Bloom Covariate")
print("="*70)

# Create lagged bloom feature using the detected lag pattern.
df['bloom_lagged'] = df['plot_b_bloom'].shift(2)  # Use mode lag=2 for simplicity.
df_clean = df.dropna()

try:
    model_exog = ARIMA(df_clean['plot_a_yield'], exog=df_clean[['bloom_lagged']], order=(1, 1, 1))
    result_exog = model_exog.fit()
    
    # Forecast: assume future bloom follows its own pattern.
    future_bloom_lagged = df['plot_b_bloom'].iloc[-2:-1].values  # Use recent lagged bloom.
    forecast_exog = result_exog.get_forecast(steps=5, exog=np.ones((5, 1)) * future_bloom_lagged[0])
    print(f"\nLag-aware forecast (next 5 weeks): {forecast_exog.predicted_mean.values}")
    print(f"AIC: {result_exog.aic:.2f}")
except Exception as e:
    print(f"Lag-aware ARIMA fit failed: {e}")
    forecast_exog = None

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
Key insight: The naive ARIMA (Step 2) misses the structure because
plot A's yield is driven by plot B's bloom with a VARIABLE lag (1→2→3 weeks).
This violates ARIMA's assumption of stationary lag structure.

The lag-adjusted ARIMA (Step 3) incorporates the detected lag,
but requires domain knowledge or rolling cross-correlation to identify it first.

This is the seam: forecasting techniques assume fixed causal structure,
but garden plots (via pollinator networks) have time-varying, cross-plot
dependencies that force us to *detect and adapt* the lag rather than estimate it.
""")

print("\nAIC Comparison:")
if forecast_naive and forecast_exog:
    print(f"  Naive ARIMA AIC: {result_naive.aic:.2f}")
    print(f"  Lag-aware ARIMA AIC: {result_exog.aic:.2f}")
    print(f"  Improvement: {result_naive.aic - result_exog.aic:.2f} points (lower is better)")
