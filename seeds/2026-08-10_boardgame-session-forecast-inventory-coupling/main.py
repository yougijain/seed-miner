import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Generate synthetic board game cafe data with inventory-duration coupling
n_days = 180
date_range = pd.date_range('2024-01-01', periods=n_days, freq='D')

# Base session duration (minutes)
base_duration = 90
trend = np.linspace(0, 20, n_days)
seasonal = 15 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
noise = np.random.normal(0, 8, n_days)
session_duration = base_duration + trend + seasonal + noise
session_duration = np.clip(session_duration, 45, 180)

# Inventory of complementary games (games that extend sessions)
# Inventory DEPENDS on how many games are in use (inverse relationship)
# When sessions are long, inventory drops; when sessions are short, inventory rises
inventory_coupling = 100 - (session_duration - 60) * 0.5
inventory_coupling = np.clip(inventory_coupling, 20, 100)
# Add independent shocks (cafe restocking, new arrivals)
inventory_noise = np.random.normal(0, 5, n_days)
inventory = inventory_coupling + inventory_noise
inventory = np.clip(inventory, 20, 100)

# Session duration is also pushed DOWN when inventory is low (can't play extended games)
session_duration = session_duration - (100 - inventory) * 0.15 + np.random.normal(0, 5, n_days)
session_duration = np.clip(session_duration, 45, 180)

df = pd.DataFrame({
    'date': date_range,
    'session_minutes': session_duration,
    'complementary_game_inventory': inventory
})

# Test 1: Granger causality (does inventory Granger-cause duration?)
print("=" * 70)
print("TEST 1: Granger Causality (inventory → session duration)")
print("=" * 70)
data_for_granger = df[['session_minutes', 'complementary_game_inventory']].values
try:
    gc_result = grangercausalitytests(data_for_granger, maxlag=7, verbose=True)
    print("\nInterpretation: p-value < 0.05 suggests inventory Granger-causes duration.")
    print("If true, standard ARIMA ignoring inventory will be misspecified.\n")
except Exception as e:
    print(f"Granger test error: {e}\n")

# Test 2: Naive ARIMA (ignoring inventory covariate)
print("=" * 70)
print("TEST 2: Naive ARIMA(1,1,1) on duration only")
print("=" * 70)
train_size = int(0.8 * len(df))
train_duration = df['session_minutes'][:train_size]
test_duration = df['session_minutes'][train_size:]

try:
    arima_naive = ARIMA(train_duration, order=(1, 1, 1))
    arima_naive_fit = arima_naive.fit()
    naive_pred = arima_naive_fit.get_forecast(steps=len(test_duration)).predicted_mean
    naive_rmse = np.sqrt(np.mean((test_duration.values - naive_pred.values) ** 2))
    print(f"Naive ARIMA RMSE: {naive_rmse:.2f} minutes")
    print(f"Test set mean duration: {test_duration.mean():.2f} minutes")
    print(f"Naive model explains {100*(1 - naive_rmse/test_duration.std()):.1f}% of variation.\n")
except Exception as e:
    print(f"Naive ARIMA error: {e}\n")

# Test 3: ARIMAX (treating inventory as exogenous)
# This is the workaround that works IF inventory is truly exogenous,
# but fails structurally if inventory is endogenous.
print("=" * 70)
print("TEST 3: ARIMAX with inventory as exogenous regressor")
print("=" * 70)
train_exog = df['complementary_game_inventory'][:train_size].values.reshape(-1, 1)
test_exog = df['complementary_game_inventory'][train_size:].values.reshape(-1, 1)

try:
    arima_exog = ARIMA(train_duration, exog=train_exog, order=(1, 1, 1))
    arima_exog_fit = arima_exog.fit()
    exog_pred = arima_exog_fit.get_forecast(steps=len(test_duration), exog=test_exog).predicted_mean
    exog_rmse = np.sqrt(np.mean((test_duration.values - exog_pred.values) ** 2))
    print(f"ARIMAX RMSE (inventory as exog): {exog_rmse:.2f} minutes")
    print(f"Improvement over naive: {100*(naive_rmse - exog_rmse)/naive_rmse:.1f}%")
    print(f"WARNING: This assumes inventory is exogenous. Granger test above")
    print(f"suggests it may be endogenous (duration → inventory feedback).")
    print(f"If endogenous, ARIMAX estimates are structurally biased.\n")
except Exception as e:
    print(f"ARIMAX error: {e}\n")

# Test 4: Differencing inventory to break feedback loop
print("=" * 70)
print("TEST 4: ARIMAX with differenced inventory (remove trend/feedback)")
print("=" * 70)
inv_diff = np.diff(df['complementary_game_inventory'].values)
train_exog_diff = inv_diff[:train_size-1].reshape(-1, 1)
test_exog_diff = inv_diff[train_size-1:].reshape(-1, 1)

try:
    arima_diff = ARIMA(train_duration[:-1], exog=train_exog_diff, order=(1, 1, 1))
    arima_diff_fit = arima_diff.fit()
    diff_pred = arima_diff_fit.get_forecast(steps=len(test_duration)-1, exog=test_exog_diff).predicted_mean
    diff_rmse = np.sqrt(np.mean((test_duration[:-1].values - diff_pred.values) ** 2))
    print(f"ARIMAX RMSE (differenced inventory): {diff_rmse:.2f} minutes")
    print(f"Hypothesis: differencing breaks endogenous feedback.")
    print(f"If differenced < raw exog, endogeneity was inflating error.\n")
except Exception as e:
    print(f"Differenced ARIMAX error: {e}\n")

print("=" * 70)
print("SUMMARY: The non-obvious angle")
print("=" * 70)
print("Standard forecasting treats regressors as exogenous inputs.")
print("In inventory-constrained domains, the quantity being forecast")
print("(session duration) directly influences the regressor (inventory).")
print("Naive ARIMAX fails because it assumes one-way causality.")
print("Granger causality + differencing can detect and mitigate this.")
