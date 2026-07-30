import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic data: 200 performances, transaction-level ticket purchases
# Categories: orchestra, mezzanine, balcony (share must sum to 1 per transaction)
performances = 200
transactions_per_perf = np.random.randint(15, 40, performances)

data = []
for perf_idx in range(performances):
    perf_date = datetime(2024, 1, 1) + timedelta(days=perf_idx // 4)
    n_trans = transactions_per_perf[perf_idx]
    
    # Normal pattern: ~50% orchestra, ~30% mezzanine, ~20% balcony
    # (with realistic Dirichlet variation)
    if perf_idx < 180:  # Normal period
        alpha = np.array([50, 30, 20])
        mixes = np.random.dirichlet(alpha, n_trans)
    else:  # Anomaly period (last 20 perfs): sudden shift to balcony dominance
        # Simulates venue reconfiguration or unexpected casting change
        alpha = np.array([20, 20, 60])
        mixes = np.random.dirichlet(alpha, n_trans)
    
    for trans_idx, (orch, mezz, bal) in enumerate(mixes):
        data.append({
            'perf_idx': perf_idx,
            'perf_date': perf_date,
            'trans_id': f"P{perf_idx}_T{trans_idx}",
            'orch_frac': orch,
            'mezz_frac': mezz,
            'bal_frac': bal,
            'total_qty': np.random.randint(1, 8)  # unused in anomaly detection
        })

df = pd.DataFrame(data)

# Features for isolation forest: compositional ticket ratios
X = df[['orch_frac', 'mezz_frac', 'bal_frac']].values

# Fit isolation forest
iso_forest = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
anomalies = iso_forest.fit_predict(X)
df['anomaly'] = anomalies == -1
df['anomaly_score'] = iso_forest.score_samples(X)

# Aggregate to performance level: what fraction of transactions in each perf were anomalous
perf_agg = df.groupby('perf_idx').agg({
    'anomaly': ['sum', 'count'],
    'anomaly_score': 'mean',
    'orch_frac': 'mean',
    'mezz_frac': 'mean',
    'bal_frac': 'mean'
}).reset_index()
perf_agg.columns = ['perf_idx', 'anomaly_count', 'total_trans', 'mean_score', 'orch_mean', 'mezz_mean', 'bal_mean']
perf_agg['anomaly_rate'] = perf_agg['anomaly_count'] / perf_agg['total_trans']

# Results
print("=== COMMUNITY THEATER DEMAND REVERSAL DETECTION ===")
print(f"\nDataset: {len(df)} transactions across {performances} performances.")
print(f"Ground truth: Performances 0-179 = normal; 180-199 = demand shift (balcony surge).")
print(f"\nTop 10 anomalous performances (by anomaly rate):")
print(perf_agg.nlargest(10, 'anomaly_rate')[['perf_idx', 'anomaly_rate', 'orch_mean', 'mezz_mean', 'bal_mean']])

print(f"\nPerformances 180-199 stats (ground truth anomaly window):")
late_window = perf_agg[perf_agg['perf_idx'] >= 180]
print(f"  Mean anomaly rate: {late_window['anomaly_rate'].mean():.3f}")
print(f"  Mean balcony fraction: {late_window['bal_mean'].mean():.3f}")

print(f"\nPerformances 0-179 stats (baseline):")
baseline = perf_agg[perf_agg['perf_idx'] < 180]
print(f"  Mean anomaly rate: {baseline['anomaly_rate'].mean():.3f}")
print(f"  Mean balcony fraction: {baseline['bal_mean'].mean():.3f}")

print(f"\nSeparation: Anomaly detection flags {(late_window['anomaly_rate'].mean() / (baseline['anomaly_rate'].mean() + 1e-6)):.1f}x higher rate in demand-shift window.")
print(f"\nNote: Real theater box offices would show this structure in actual transaction logs;")
print(f"the challenge here is that isolation forest normally flags *extreme* univariate values,")
print(f"not compositional *rebalancing*—the algorithm had to detect a change in co-purchase geometry.")