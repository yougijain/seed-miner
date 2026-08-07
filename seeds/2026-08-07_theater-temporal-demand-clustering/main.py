import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime, timedelta

np.random.seed(42)

# Generate synthetic theater box office data with realistic temporal structure
# Three hidden audience cohorts:
# 1. Planners: buy 5-14 days in advance, large groups, bunched arrivals (1-2 min apart)
# 2. Walk-ups: arrive day-of or day-before, single/pair tickets, sparse arrivals (10-60 min apart)
# 3. Subscribers: buy exactly 7 or 14 days before, tiny inter-arrival (seconds), renewal bursts

events = []
show_date = datetime(2025, 3, 15)

# Planners: clustered purchase bursts 5-14 days out
for _ in range(8):
    burst_time = show_date - timedelta(days=np.random.randint(5, 15))
    for _ in range(np.random.randint(2, 5)):
        burst_time += timedelta(minutes=np.random.exponential(1.5))  # tight clustering
        events.append({'timestamp': burst_time, 'cohort': 'planner', 'qty': np.random.randint(2, 6)})

# Walk-ups: sparse, last-minute
for _ in range(15):
    arrival_time = show_date - timedelta(hours=np.random.exponential(24))  # exponential tail toward show date
    arrival_time += timedelta(minutes=np.random.uniform(-30, 30))  # uniform jitter
    events.append({'timestamp': arrival_time, 'cohort': 'walkup', 'qty': np.random.randint(1, 3)})

# Subscribers: regular pulses at day 7 and day 14 before show
for day_offset in [7, 14]:
    sub_time = show_date - timedelta(days=day_offset)
    for _ in range(np.random.randint(3, 8)):
        sub_time += timedelta(seconds=np.random.exponential(5))  # very tight
        events.append({'timestamp': sub_time, 'cohort': 'subscriber', 'qty': 1})

df = pd.DataFrame(events).sort_values('timestamp').reset_index(drop=True)

# Compute inter-arrival times (feature for clustering)
df['hours_before_show'] = (show_date - df['timestamp']).dt.total_seconds() / 3600
df['inter_arrival_sec'] = df['timestamp'].diff().dt.total_seconds
df['inter_arrival_sec'].iloc[0] = 0
df['inter_arrival_min'] = df['inter_arrival_sec'] / 60

# Lag features: running statistics of recent inter-arrivals (capture burst structure)
window = 3
df['inter_arrival_lag_mean'] = df['inter_arrival_min'].rolling(window=window, min_periods=1).mean()
df['inter_arrival_lag_std'] = df['inter_arrival_min'].rolling(window=window, min_periods=1).std().fillna(0)

# Cluster on temporal microstructure only (not qty or explicit cohort)
features = df[['hours_before_show', 'inter_arrival_min', 'inter_arrival_lag_mean', 'inter_arrival_lag_std']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# DBSCAN is a natural fit for temporal bursts: tight epsilon captures clustering, sparse arrivals are noise/core points
dbscan = DBSCAN(eps=0.4, min_samples=2)
clusters = dbscan.fit_predict(features_scaled)
df['predicted_cohort'] = clusters

# Evaluation: map predicted clusters to true cohorts and compute purity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

true_labels = df['cohort'].astype('category').cat.codes.values
ari = adjusted_rand_score(true_labels, clusters)
nmi = normalized_mutual_info_score(true_labels, clusters)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Info: {nmi:.3f}")
print(f"\nCluster sizes:\n{pd.Series(clusters).value_counts().sort_index()}")
print(f"\nTrue cohort breakdown:\n{df['cohort'].value_counts()}")
print(f"\nCluster composition (true labels):")
for cluster_id in sorted(set(clusters)):
    if cluster_id == -1:
        mask = clusters == cluster_id
        print(f"  Cluster {cluster_id} (noise): {df[mask]['cohort'].value_counts().to_dict()}")
    else:
        mask = clusters == cluster_id
        print(f"  Cluster {cluster_id}: {df[mask]['cohort'].value_counts().to_dict()}")

# Sample output for inspection
print(f"\nSample rows (first 10):")
print(df[['timestamp', 'cohort', 'inter_arrival_min', 'inter_arrival_lag_mean', 'predicted_cohort']].head(10).to_string())

results = {
    'adjusted_rand_index': float(ari),
    'normalized_mutual_info': float(nmi),
    'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
    'noise_points': int((clusters == -1).sum()),
    'total_events': len(df)
}
print(f"\n{json.dumps(results, indent=2)}")
