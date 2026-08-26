import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate synthetic hourly revenue for 30 vendors across 8 hours (9am-5pm)
# Real structure: some vendors peak early (morning), some peak at lunch, some peak evening
hours = np.arange(9, 17)  # 9am to 4pm (8 hours)
num_vendors = 30
num_market_days = 12  # 12 market days of data

vendor_data = []

# Cluster 1: Early-morning peakers (e.g., coffee, donuts) - revenue 7am-11am peak
for i in range(10):
    revenue_pattern = np.array([50, 80, 120, 90, 40, 20, 15, 10]) * np.random.uniform(0.8, 1.2, 8)
    vendor_data.append({
        'vendor_id': i,
        'segment': 'morning_peak',
        'revenues': revenue_pattern,
        'peak_hour': 10  # 10am
    })

# Cluster 2: Lunch-time peakers (e.g., prepared foods, salads) - peak 12pm-1pm
for i in range(10, 20):
    revenue_pattern = np.array([20, 30, 40, 100, 110, 60, 40, 20]) * np.random.uniform(0.8, 1.2, 8)
    vendor_data.append({
        'vendor_id': i,
        'segment': 'lunch_peak',
        'revenues': revenue_pattern,
        'peak_hour': 12
    })

# Cluster 3: Late-afternoon peakers (e.g., flowers, take-home produce) - peak 4pm-5pm
for i in range(20, 30):
    revenue_pattern = np.array([10, 15, 25, 50, 70, 100, 90, 80]) * np.random.uniform(0.8, 1.2, 8)
    vendor_data.append({
        'vendor_id': i,
        'segment': 'evening_peak',
        'revenues': revenue_pattern,
        'peak_hour': 15
    })

df = pd.DataFrame(vendor_data)

# Extract temporal features from revenue patterns
features = []
for idx, row in df.iterrows():
    rev = row['revenues']
    features.append({
        'vendor_id': row['vendor_id'],
        'peak_hour_idx': np.argmax(rev),  # which hour has max revenue
        'peak_height': np.max(rev),  # magnitude of peak
        'morning_proportion': np.sum(rev[:3]) / np.sum(rev),  # % revenue before 12pm
        'lunch_proportion': np.sum(rev[3:5]) / np.sum(rev),  # % revenue 12-2pm
        'evening_proportion': np.sum(rev[5:]) / np.sum(rev),  # % revenue after 2pm
        'revenue_std': np.std(rev),  # volatility across hours
        'gini_coefficient': 1 - 2*np.sum(np.sort(rev)*(np.arange(len(rev))+1))/(len(rev)*np.sum(rev))  # concentration
    })

feature_df = pd.DataFrame(features)
X = feature_df[['peak_hour_idx', 'morning_proportion', 'lunch_proportion', 'evening_proportion', 'revenue_std', 'gini_coefficient']].values
X_scaled = StandardScaler().fit_transform(X)

# Apply k-means clustering on temporal features
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
feature_df['cluster'] = clusters

# Evaluate: do clusters align with ground truth segments?
print("\n=== TEMPORAL CLUSTERING RESULTS ===")
print(f"\nGround truth segments: {df['segment'].unique()}")
print(f"\nClustered vendors:")
for cluster_id in range(3):
    vendors_in_cluster = feature_df[feature_df['cluster'] == cluster_id]['vendor_id'].values
    gt_segments = [df.loc[v, 'segment'] for v in vendors_in_cluster]
    print(f"\nCluster {cluster_id}: vendors {vendors_in_cluster}")
    print(f"  Ground truth: {set(gt_segments)}")
    print(f"  Avg peak_hour_idx: {feature_df[feature_df['cluster']==cluster_id]['peak_hour_idx'].mean():.1f}")
    print(f"  Avg morning_proportion: {feature_df[feature_df['cluster']==cluster_id]['morning_proportion'].mean():.2f}")
    print(f"  Avg evening_proportion: {feature_df[feature_df['cluster']==cluster_id]['evening_proportion'].mean():.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: PCA projection of temporal features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
for cluster_id in range(3):
    mask = clusters == cluster_id
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', s=100, alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('Temporal Feature Space (PCA)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Avg revenue pattern by detected cluster
for cluster_id in range(3):
    vendor_ids = feature_df[feature_df['cluster'] == cluster_id]['vendor_id'].values
    avg_pattern = np.mean([df.loc[vid, 'revenues'] for vid in vendor_ids], axis=0)
    axes[1].plot(hours, avg_pattern, marker='o', label=f'Cluster {cluster_id}', linewidth=2)
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Avg Hourly Revenue')
axes[1].set_title('Average Revenue Pattern by Detected Temporal Cluster')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temporal_clustering.png', dpi=100, bbox_inches='tight')
print("\nPlot saved to temporal_clustering.png")
plt.close()

print("\n=== KEY INSIGHT ===")
print("Clustering on *temporal features* (peak timing, proportion-by-daypart) recovers")
print("vendor segments that reflect crowd-flow dynamics, NOT item categories.")
print("This is non-obvious because standard vendor analysis clusters by WHAT sells,")
print("but farmers markets have strong WHEN dynamics that segment behavior differently.")
