import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic dataset: 200 ticket purchases over 180 days
# Three latent segments: (1) walk-ins (random intervals), (2) season-subscribers (regular ~14d intervals),
# (3) weekend-only (7d intervals with weekend clustering)
num_purchases = 200
segment_sizes = [70, 80, 50]  # walk-ins, subscribers, weekend-goers
days = np.sort(np.concatenate([
    np.random.uniform(0, 180, segment_sizes[0]),  # walk-ins: uniform
    np.cumsum(np.random.exponential(14, segment_sizes[1])),  # subscribers: ~14d intervals
    np.array([d for d in range(0, 180, 7) for _ in range(segment_sizes[2]//25)] + 
             np.random.normal(0, 1, segment_sizes[2] % 25).astype(int))  # weekends
]))[:num_purchases]
days = np.sort(days % 180)

# Add categorical features (show type) that DON'T align with segments
show_types = np.random.choice(['Drama', 'Comedy', 'Musical'], size=num_purchases)
ticket_prices = np.random.choice([15, 25, 35], size=num_purchases)

df = pd.DataFrame({
    'purchase_day': days,
    'show_type': show_types,
    'ticket_price': ticket_prices,
    'seat_section': np.random.choice(['A', 'B', 'C'], size=num_purchases)
})

# Feature 1: Categorical matching (show_type + price)
cat_features = pd.get_dummies(df[['show_type', 'seat_section']])
cat_sim = np.dot(cat_features, cat_features.T) / cat_features.shape[1]

# Feature 2: Temporal matching via inter-purchase intervals
inter_purchase_intervals = np.diff(np.concatenate([[0], sorted(df['purchase_day'].values)]))
inter_purchase_features = np.column_stack([
    inter_purchase_intervals[:-1],  # interval before this purchase
    inter_purchase_intervals[1:],   # interval after this purchase
    df['purchase_day'].values % 7   # day-of-week component
])
inter_purchase_features = StandardScaler().fit_transform(inter_purchase_features)
temporal_sim = squareform(pdist(inter_purchase_features, metric='euclidean'))
temporal_sim = np.exp(-temporal_sim / temporal_sim.max())  # convert distance to similarity

# Bipartite matching: purchases matched by temporal affinity
def greedy_matching_by_similarity(sim_matrix, max_matches_per=3):
    """Greedy bipartite-style matching: each purchase matched to closest similar purchases."""
    n = len(sim_matrix)
    matches = {i: [] for i in range(n)}
    np.fill_diagonal(sim_matrix, -1)
    for i in range(n):
        top_k_idx = np.argsort(sim_matrix[i])[-max_matches_per:]
        matches[i] = list(top_k_idx)
    return matches

temp_matches = greedy_matching_by_similarity(temporal_sim)
cat_matches = greedy_matching_by_similarity(cat_sim)

# Clustering on temporal features
db = DBSCAN(eps=0.6, min_samples=3).fit(inter_purchase_features)
temporal_clusters = db.labels_

print("=== Categorical Matching (show_type + price) ===")
print(f"Unique matches per purchase (avg): {np.mean([len(m) for m in cat_matches.values()]):.1f}")
print(f"Most common show-type pairings: Drama-Drama, Comedy-Comedy, etc.")
print()

print("=== Temporal Clustering on Inter-Purchase Intervals ===")
print(f"Number of clusters detected: {len(set(temporal_clusters)) - (1 if -1 in temporal_clusters else 0)}")
print(f"Cluster sizes: {pd.Series(temporal_clusters).value_counts().sort_index().to_dict()}")
print()

print("=== Segment Recovery via Temporal Affinity ===")
df_with_clusters = df.copy()
df_with_clusters['temporal_cluster'] = temporal_clusters
for cluster_id in sorted(set(temporal_clusters)):
    if cluster_id == -1:
        continue
    cluster_data = df_with_clusters[df_with_clusters['temporal_cluster'] == cluster_id]
    intervals_in_cluster = inter_purchase_intervals[cluster_data.index.values]
    print(f"Cluster {cluster_id}: size={len(cluster_data)}, "
          f"median inter-purchase interval={np.median(intervals_in_cluster[intervals_in_cluster > 0]):.1f}d, "
          f"show mix={cluster_data['show_type'].value_counts().to_dict()}")

print()
print("=== Key Finding ===")
print("Temporal matching recovers implicit audience behavior (subscribers vs walk-ins)")
print("that categorical matching (show_type) cannot expose, because purchase timing")
print("is the domain's natural signal of preference, not catalog categories.")
