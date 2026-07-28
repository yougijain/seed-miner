import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic dataset: 40 vendors, 8 weeks, 5 product categories
# Each vendor's revenue is split across produce, dairy, prepared, flowers, eggs
weeks = 8
vendors = 40
categories = ['produce', 'dairy', 'prepared', 'flowers', 'eggs']

data = []
for v in range(vendors):
    # Baseline composition for this vendor (Dirichlet-sampled)
    base_comp = np.random.dirichlet([2, 1.5, 1, 0.5, 1])
    
    for w in range(weeks):
        # Normal case: small weekly noise
        if not (v >= 35 and w >= 4):  # vendors 35-39 become unstable at week 4
            comp = np.random.dirichlet(base_comp * 10)  # concentration ~10
        else:
            # Anomaly: wild swings in allocation (seller losing category)
            comp = np.random.dirichlet([0.5, 5, 0.3, 3, 0.2])
        
        total_revenue = np.random.normal(300, 50)
        row = {'vendor_id': v, 'week': w, 'total_revenue': max(50, total_revenue)}
        for cat, prop in zip(categories, comp):
            row[cat] = prop
        data.append(row)

df = pd.DataFrame(data)

# Additive log-ratio (alr) transform: log-ratio each category to reference (eggs)
for cat in categories[:-1]:
    df[f'alr_{cat}'] = np.log(df[cat] / df['eggs'])

alr_features = [f'alr_{cat}' for cat in categories[:-1]]
X = df[alr_features].values

# Standardize for iso-forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest: contamination tuned to expect ~12.5% anomalies
iso_forest = IsolationForest(contamination=0.125, random_state=42)
df['anomaly'] = iso_forest.fit_predict(X_scaled)
df['anomaly_score'] = iso_forest.score_samples(X_scaled)

# Results
anomalies = df[df['anomaly'] == -1]
print(f"Detected {len(anomalies)} anomalies (rows)")
print(f"Vendors with anomalies: {sorted(anomalies['vendor_id'].unique())}")
print(f"\nExpected anomalous vendors (35-39 after week 3): {list(range(35, 40))}")

by_vendor = df.groupby('vendor_id').apply(lambda g: (g['anomaly'] == -1).sum())
print(f"\nAnomalies per vendor (top 10):")
print(by_vendor.nlargest(10))

print(f"\nSample anomalous row (unstable vendor):")
print(anomalies.iloc[0][['vendor_id', 'week', 'produce', 'dairy', 'prepared', 'flowers', 'eggs', 'anomaly_score']])
