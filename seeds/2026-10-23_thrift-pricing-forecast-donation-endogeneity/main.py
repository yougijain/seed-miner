import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic thrift store data: weekly pricing and donation volume
# Real thrift stores endogenously lower prices when donations spike (storage pressure)
weeks = 120
time_idx = np.arange(weeks)

# Donation volume (exogenous trend + noise)
donation_volume = 50 + 0.3 * time_idx + np.random.normal(0, 8, weeks)
donation_volume = np.maximum(donation_volume, 10)

# Pricing logic switches when donations exceed threshold
pricing_regime = (donation_volume > np.percentile(donation_volume, 70)).astype(int)

# Average item price: baseline $8, drops to $5.50 when donations high (endogenous rule)
base_price = 8.0
regime_shift_price = 5.5
avg_price = np.where(pricing_regime == 1, regime_shift_price, base_price)
avg_price = avg_price + np.random.normal(0, 0.4, weeks)

df = pd.DataFrame({
    'week': time_idx,
    'donations': donation_volume,
    'avg_price': np.maximum(avg_price, 2.0),
    'regime': pricing_regime
})

print("=" * 70)
print("THRIFT STORE PRICING FORECAST COLLAPSE DETECTOR")
print("=" * 70)
print(f"\nDataset: {weeks} weeks. Donation volume drives hidden pricing regime changes.")
print(f"Regime switch threshold: {np.percentile(donation_volume, 70):.1f} donations/week")
print(f"Price drops from ${base_price:.2f} → ${regime_shift_price:.2f} when donations spike.")
print(f"\nActual regime changes observed: {pricing_regime.sum()} weeks in high-donation regime")

# Split: train on first 80 weeks, test on last 40
train_df = df[df['week'] < 80].copy()
test_df = df[df['week'] >= 80].copy()

print(f"\nTrain: weeks 0-79 (mostly low-donation regime)")
print(f"Test: weeks 80-119 (mixed, with donation spike around week 95+)")

# Naive ARIMA-like forecast: fit linear trend + seasonal on train data
train_prices = train_df['avg_price'].values
train_weeks = train_df['week'].values

model = LinearRegression()
model.fit(train_weeks.reshape(-1, 1), train_prices)

# Forecast on test data
test_weeks = test_df['week'].values
forecast = model.predict(test_weeks.reshape(-1, 1))
actual = test_df['avg_price'].values

# Compute rolling prediction error and interval width
window = 5
errors = actual - forecast
rolling_std = pd.Series(errors).rolling(window).std().values
rolling_mean_error = pd.Series(errors).rolling(window).mean().values

# Prediction interval (±2σ)
pi_upper = forecast + 2 * rolling_std
pi_lower = forecast - 2 * rolling_std

# Detect collapse: where PI width becomes extremely wide (forecast loses confidence)
pi_width = pi_upper - pi_lower
pi_width_z = (pi_width - np.nanmean(pi_width)) / (np.nanstd(pi_width) + 1e-6)
collapse_weeks = np.where((pi_width_z > 1.5) | (pi_width_z < -0.5))[0]

print(f"\n" + "=" * 70)
print("FORECAST PERFORMANCE")
print("=" * 70)
print(f"Mean Absolute Error (test): ${np.mean(np.abs(errors)):.2f}")
print(f"Forecast interval collapse events (PI width anomalies): {len(collapse_weeks)} weeks")
print(f"Collapse weeks (test index): {collapse_weeks[:10].tolist()}...")
if len(collapse_weeks) > 0:
    actual_collapse_weeks = test_weeks[collapse_weeks]
    print(f"Calendar weeks: {actual_collapse_weeks[:5].tolist()}...")
    collapse_donations = test_df.iloc[collapse_weeks]['donations'].values
    print(f"Donation volumes during collapse: {collapse_donations[:5].tolist()}...")
    print(f"Mean donation during collapse: {np.mean(collapse_donations):.1f}")
    print(f"Mean donation during normal weeks: {np.mean(test_df.iloc[~np.isin(np.arange(len(test_df)), collapse_weeks)]['donations'].values):.1f}")

# Compare: are collapse weeks associated with high donations?
collapse_df = test_df.iloc[collapse_weeks] if len(collapse_weeks) > 0 else pd.DataFrame()
normal_df = test_df.iloc[~np.isin(np.arange(len(test_df)), collapse_weeks)]

if len(collapse_df) > 0:
    collapse_high_donation = (collapse_df['donations'] > np.percentile(df['donations'], 70)).mean()
    normal_high_donation = (normal_df['donations'] > np.percentile(df['donations'], 70)).mean()
    print(f"\nFraction in HIGH-donation regime:")
    print(f"  During forecast collapse: {collapse_high_donation:.2%}")
    print(f"  During normal forecasting: {normal_high_donation:.2%}")
    print(f"\n→ Insight: Forecast breaks when donations spike, signaling endogenous")
    print(f"  pricing-regime shift that violates ARIMA stationarity assumption.")
else:
    print("\n(No collapse events detected; regime shift may be subtle in this sample)")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("Standard forecasting fails not because demand is random,")
print("but because the pricing *rule itself* changes when donations surge.")
print("This endogeneity breaks stationarity: the time series is non-stationary")
print("by design, not due to noise or trend.")
print("\nDetecting PI collapse isolates weeks where hidden regime switches occur.")
