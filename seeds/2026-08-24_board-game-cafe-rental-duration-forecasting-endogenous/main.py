import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic: 180 days of rental data for a popular 4-player game.
# Rental duration in minutes. Inventory availability (0-5 copies) is endogenous:
# - When game is heavily rented, fewer copies available → next renters see scarcity → rent shorter (avoid queue)
# - Short rentals free up copies faster → more availability → next renters rent longer (less pressure)

days = 180
trend = np.linspace(45, 55, days)  # Baseline 45–55 min
base_duration = trend + np.random.normal(0, 3, days)

# Endogenous inventory feedback.
inventory = np.zeros(days)
inventory[0] = 3
rental_duration = np.zeros(days)

for t in range(days):
    # Duration is depressed when inventory is low (scarcity signal).
    rental_duration[t] = base_duration[t] - 5 * (1 - inventory[t] / 5.0) + np.random.normal(0, 2)
    rental_duration[t] = max(20, rental_duration[t])  # Floor at 20 min.
    
    # Inventory next day depends on *this day's rental duration*:
    # Short rentals → more throughput → inventory replenishes faster.
    # Long rentals → less throughput → inventory depletes.
    replenish_rate = 2 + (60 - rental_duration[t]) / 30  # Faster replenish if short rentals.
    if t < days - 1:
        inventory[t + 1] = np.clip(inventory[t] - np.random.poisson(1.5) + np.random.poisson(replenish_rate), 0, 5)

df = pd.DataFrame({
    'day': np.arange(1, days + 1),
    'rental_duration_min': np.round(rental_duration, 1),
    'inventory_available': inventory.astype(int)
})

print("Data (first 10 rows):")
print(df.head(10))
print(f"\nCorrelation (rental_duration vs inventory_available): {df['rental_duration_min'].corr(df['inventory_available']):.3f}")
print("^ Positive correlation signals endogeneity: availability does NOT independently cause duration.")
print("  Rather, duration causes next-day availability.\n")

# Naive ARIMA (ignoring inventory as exogenous).
model_naive = ARIMA(df['rental_duration_min'], order=(1, 0, 1))
result_naive = model_naive.fit()
print("Naive ARIMA(1,0,1) — ignoring feedback:")
print(result_naive.summary().tables[1])
print()

# ARIMA with inventory as exogenous covariate (assumes it is independent — it is not).
model_exog = ARIMA(df['rental_duration_min'], exog=df[['inventory_available']], order=(1, 0, 1))
result_exog = model_exog.fit()
print("ARIMA(1,0,1) with inventory as exogenous covariate:")
print(result_exog.summary().tables[1])
print("^ The exogenous covariate coefficient may appear significant, but it is BIASED:")
print("  inventory is caused by duration, not the reverse. Standard errors are invalid.\n")

# Inspect ACF/PACF to detect endogeneity signature.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['rental_duration_min'], lags=20, ax=axes[0])
axes[0].set_title('ACF: Rental Duration\n(Endogenous feedback may inflate autocorrelation)')
plot_pacf(df['rental_duration_min'], lags=20, ax=axes[1], method='ywm')
axes[1].set_title('PACF: Rental Duration')
plt.tight_layout()
plt.savefig('acf_pacf.png', dpi=100)
print("ACF/PACF plots saved to acf_pacf.png")
print("^ If feedback is strong, ACF will decay slowly (spurious autocorrelation from endogeneity).\n")

# Demonstrate the problem: forecast one step ahead using both models.
forecast_horizon = 10
forecast_naive = result_naive.get_forecast(steps=forecast_horizon).predicted_mean
forecast_exog = result_exog.get_forecast(steps=forecast_horizon, exog=df[['inventory_available']].tail(forecast_horizon)).predicted_mean

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['day'].tail(50), df['rental_duration_min'].tail(50), 'o-', label='Observed', linewidth=1.5)
forecast_days = np.arange(days + 1, days + forecast_horizon + 1)
ax.plot(forecast_days, forecast_naive, 's--', label='Naive ARIMA forecast', linewidth=1.5, alpha=0.7)
ax.plot(forecast_days, forecast_exog, '^--', label='ARIMA+exog forecast', linewidth=1.5, alpha=0.7)
ax.axvline(x=days, color='red', linestyle=':', alpha=0.5, label='Forecast origin')
ax.set_xlabel('Day')
ax.set_ylabel('Rental Duration (min)')
ax.set_title('Forecasts Diverge When Endogenous Feedback Is Ignored')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_comparison.png', dpi=100)
print("Forecast comparison saved to forecast_comparison.png\n")

print("KEY INSIGHT:")
print("Standard ARIMA treats inventory as exogenous (independent), but it is endogenous.")
print("The positive correlation between duration and future availability is NOT a causal")
print("relationship where availability lengthens rentals. Rather, short rentals cause")
print("inventory to replenish, which appears to lengthen future rentals.")
print("\nForecasting duration without modeling this feedback loop produces biased forecasts.")
print("Detection: high ACF, unexplained autocorrelation, and exogenous covariate")
print("          coefficient that reverses under lagged-difference or IV specification.")
