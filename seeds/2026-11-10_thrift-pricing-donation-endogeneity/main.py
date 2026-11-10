import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic thrift store pricing data: donation velocity drives unobserved pricing regime switches
days = 180
data = []

# Simulate two hidden regimes: Normal (high donation) and Clearance (low donation)
donation_vol = np.random.poisson(50, days).astype(float)
pricing_regime = np.ones(days)  # 1=normal, 0=clearance

# Donation shocks trigger regime switches
shock_days = np.random.choice(range(20, days-20), 5, replace=False)
for shock_day in shock_days:
    donation_vol[shock_day:shock_day+14] *= np.random.uniform(0.2, 0.4)  # volume crash
    pricing_regime[shock_day:shock_day+14] = 0  # unobserved regime flip to clearance

# Generate item prices under each regime
for day in range(days):
    regime = pricing_regime[day]
    n_items = int(donation_vol[day])
    
    if regime == 1:  # Normal regime: higher avg price, lower variance
        prices = np.random.gamma(shape=3.5, scale=2.2, size=n_items)
    else:  # Clearance regime: lower avg price, higher variance (aggressive discounting)
        prices = np.random.gamma(shape=2.0, scale=1.5, size=n_items)
    
    avg_price = np.mean(prices) if prices.size > 0 else 0
    price_var = np.var(prices) if prices.size > 0 else 0
    
    data.append({
        'day': day,
        'donation_volume': donation_vol[day],
        'avg_daily_price': avg_price,
        'daily_price_variance': price_var,
        'true_regime': regime
    })

df = pd.DataFrame(data)

# Apply Isolation Forest on feature space: [donation_volume, avg_daily_price]
features = df[['donation_volume', 'avg_daily_price']].values
if_model = IsolationForest(contamination=0.15, random_state=42)
anomaly_scores = if_model.fit_predict(features)
df['anomaly'] = anomaly_scores == -1

# Key insight: cluster anomalies by regime
df['anomaly_regime'] = df.groupby((df['true_regime'] != df['true_regime'].shift()).cumsum())['anomaly'].transform('mean')

# Detect regime switches by measuring *anomaly clustering* in sliding windows
window = 7
df['anomaly_density'] = df['anomaly'].rolling(window=window, center=True).mean()

# Find peaks in anomaly density: these mark regime transitions
peaks, _ = find_peaks(df['anomaly_density'].fillna(0), height=0.3, distance=10)

print("=" * 70)
print("THRIFT STORE PRICING REGIME DETECTION VIA DONATION ENDOGENEITY")
print("=" * 70)
print(f"\nDataset: {len(df)} days of thrift store pricing data")
print(f"True regime switches: {len(shock_days)} donation-shock triggered events")
print(f"\nIsolation Forest Anomalies Detected: {df['anomaly'].sum()}")
print(f"\nDetected Regime Transition Events (via anomaly clustering):")
for peak in peaks:
    print(f"  Day {int(df.iloc[peak]['day'])}: anomaly density = {df.iloc[peak]['anomaly_density']:.3f}")

print(f"\nKey Finding:")
print(f"  Anomaly detection alone treats pricing swings as isolated outliers.")
print(f"  But clustered anomalies in sliding windows reveal REGIME TRANSITIONS.")
print(f"  The technique fails to isolate single anomalies because pricing regime")
print(f"  changes are endogenous to unobserved donation-volume shocks.")
print(f"  Solution: cluster anomaly *density* over time to infer regimes.")

print(f"\nRegime Accuracy (detected vs true):")
df['detected_regime'] = (df['anomaly_density'] < 0.15).astype(int)
accuracy = (df['detected_regime'] == df['true_regime']).mean()
print(f"  Accuracy: {accuracy:.2%}")
print(f"\nThis demonstrates the seam: standard anomaly detection assumes")
print(f"independence, but endogenous regime-switching violates that assumption,")
print(f"forcing anomaly *patterns* (not individual scores) to carry the signal.")
