import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import json

# Synthetic concession transaction times during intermissions for 12 shows over 4 weeks
np.random.seed(42)

shows = []
for show_id in range(1, 13):
    # Normal shows: ~40-50 transactions spread across 15-min intermission
    if show_id not in [5, 9, 11]:  # Anomaly shows
        arrivals = sorted(np.random.uniform(0, 15, np.random.randint(40, 51)))
    else:
        if show_id == 5:
            # Tech failure → audience held in seats, concessions compressed to last 2 min
            arrivals = sorted(np.random.uniform(13, 15, 8))
        elif show_id == 9:
            # Cast injury → show extended, intermission cut short, arrivals front-loaded
            arrivals = sorted(np.random.uniform(0, 5, 35))
        elif show_id == 11:
            # Early dismissal (fire alarm test): arrivals stopped mid-intermission
            arrivals = sorted(np.random.uniform(0, 8, 22))
    
    for i in range(len(arrivals)):
        shows.append({
            'show_id': show_id,
            'transaction_order': i + 1,
            'time_in_intermission': arrivals[i],
            'inter_arrival_interval': arrivals[i] - (arrivals[i-1] if i > 0 else 0)
        })

df = pd.DataFrame(shows)

# Aggregate per show: compute distribution statistics of inter-arrival intervals
show_stats = []
for show_id in df['show_id'].unique():
    subset = df[df['show_id'] == show_id].sort_values('time_in_intermission')
    intervals = subset['inter_arrival_interval'].values[1:]  # Skip first (no prior)
    
    show_stats.append({
        'show_id': show_id,
        'mean_interval': np.mean(intervals),
        'std_interval': np.std(intervals),
        'cv_interval': np.std(intervals) / (np.mean(intervals) + 1e-6),  # Coefficient of variation
        'min_interval': np.min(intervals),
        'max_interval': np.max(intervals),
        'transaction_count': len(subset),
        'interval_skew': (np.max(intervals) - np.mean(intervals)) / (np.std(intervals) + 1e-6)
    })

show_df = pd.DataFrame(show_stats)

# Anomaly detection: Isolation Forest on interval distribution features
# Key inversion: normal is *varied* arrivals (high std, high CV), anomalous is *uniform* or *clustered*
features = ['mean_interval', 'std_interval', 'cv_interval', 'transaction_count', 'interval_skew']
X = show_df[features].values

iso_forest = IsolationForest(contamination=0.25, random_state=42)
anomaly_scores = iso_forest.fit_predict(X)
anomalies = iso_forest.score_samples(X)  # Raw anomaly scores (lower = more anomalous)

show_df['anomaly_flag'] = anomaly_scores
show_df['anomaly_score'] = anomalies
show_df['expected_issue'] = show_df['show_id'].isin([5, 9, 11])

print("Show Intermission Concession Anomalies:")
print(show_df[['show_id', 'mean_interval', 'std_interval', 'cv_interval', 'transaction_count', 'anomaly_flag', 'anomaly_score', 'expected_issue']].to_string())
print()

# Evaluation
detected_anomalies = set(show_df[show_df['anomaly_flag'] == -1]['show_id'].values)
expected_anomalies = set([5, 9, 11])

print(f"Expected anomalies: {expected_anomalies}")
print(f"Detected anomalies: {detected_anomalies}")
print(f"True Positives: {detected_anomalies & expected_anomalies}")
print(f"False Positives: {detected_anomalies - expected_anomalies}")
print(f"False Negatives: {expected_anomalies - detected_anomalies}")
print()
print("Interpretation:")
for idx, row in show_df.iterrows():
    if row['anomaly_flag'] == -1:
        if row['cv_interval'] < 0.5:
            print(f"  Show {row['show_id']}: Collapsed interval variance (CV={row['cv_interval']:.2f}) → uniform clustering")
        elif row['transaction_count'] < 25:
            print(f"  Show {row['show_id']}: Reduced transaction volume → early exit")
        else:
            print(f"  Show {row['show_id']}: Skewed distribution → forced early surge")
