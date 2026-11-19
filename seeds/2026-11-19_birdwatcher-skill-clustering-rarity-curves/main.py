import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic dataset: 20 birdwatchers, 365 days of sighting records
# Novices: linear accumulation (~1-2 new species/week)
# Experienced: power-law discovery curve (fast early, saturates)

n_watchers = 20
n_days = 365
days = np.arange(1, n_days + 1)

watchers = []
sighting_curves = []

# Novice watchers (linear): cumulative = slope * days + noise
for i in range(10):
    slope = np.random.uniform(0.08, 0.12)  # ~30-45 species/year
    noise = np.random.normal(0, 1, n_days)
    cumulative = slope * days + np.cumsum(noise * 0.1)
    cumulative = np.maximum(cumulative, 0)
    sighting_curves.append(cumulative)
    watchers.append(f"novice_{i}")

# Experienced watchers (power-law): cumulative ≈ a * (days^b) where b < 1
for i in range(10):
    a = np.random.uniform(5, 8)
    b = np.random.uniform(0.55, 0.70)  # <1 = power-law saturation
    noise = np.random.normal(0, 0.5, n_days)
    cumulative = a * (days ** b) + np.cumsum(noise * 0.05)
    cumulative = np.maximum(cumulative, 0)
    sighting_curves.append(cumulative)
    watchers.append(f"expert_{i}")

sighting_curves = np.array(sighting_curves)

# Feature engineering: extract curve-shape features instead of raw counts
features = []
for curve in sighting_curves:
    # Feature 1: slope of log(cumulative) over time (power-law exponent proxy)
    log_curve = np.log(curve + 1)
    slope, _, r_val, _, _ = linregress(days, log_curve)
    
    # Feature 2: saturation point (normalized days when 90% of max is reached)
    max_val = np.max(curve)
    threshold_idx = np.where(curve >= 0.9 * max_val)[0]
    saturation_day = threshold_idx[0] / n_days if len(threshold_idx) > 0 else 1.0
    
    # Feature 3: curvature (second derivative at midpoint)
    mid_idx = n_days // 2
    if mid_idx > 1 and mid_idx < n_days - 1:
        d2 = (curve[mid_idx + 1] - 2 * curve[mid_idx] + curve[mid_idx - 1])
    else:
        d2 = 0
    
    # Feature 4: total species (raw count for reference)
    total_species = curve[-1]
    
    features.append([slope, saturation_day, d2, total_species])

features = np.array(features)

# Normalize features (crucial for K-means)
features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

# Apply K-means with k=2 (should separate novice from expert)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_norm)

# Results
df_results = pd.DataFrame({
    'watcher': watchers,
    'cluster': cluster_labels,
    'slope': features[:, 0],
    'saturation_day': features[:, 1],
    'total_species': features[:, 3]
})

print("\n=== Clustering Results (Skill Tiers) ===")
print(df_results.to_string(index=False))

print("\n=== Cluster Profiles ===")
for c in range(2):
    subset = df_results[df_results['cluster'] == c]
    print(f"\nCluster {c}:")
    print(f"  Mean slope (log-domain): {subset['slope'].mean():.4f}")
    print(f"  Mean saturation day: {subset['saturation_day'].mean():.3f}")
    print(f"  Mean total species: {subset['total_species'].mean():.1f}")
    print(f"  Members: {list(subset['watcher'])}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: sighting curves colored by cluster
for i, (curve, label, cluster) in enumerate(zip(sighting_curves, watchers, cluster_labels)):
    color = 'blue' if cluster == 0 else 'red'
    alpha = 0.7
    axes[0].plot(days, curve, color=color, alpha=alpha, linewidth=1.5)

axes[0].set_xlabel('Days')
axes[0].set_ylabel('Cumulative Species')
axes[0].set_title('Sighting Curves by Cluster\n(Blue=Cluster0, Red=Cluster1)')
axes[0].grid(True, alpha=0.3)

# Plot 2: 2D projection of features (slope vs saturation)
for cluster in range(2):
    mask = cluster_labels == cluster
    color = 'blue' if cluster == 0 else 'red'
    axes[1].scatter(features_norm[mask, 0], features_norm[mask, 1], 
                   label=f'Cluster {cluster}', s=100, color=color, alpha=0.7)

axes[1].set_xlabel('Slope (log-domain, normalized)')
axes[1].set_ylabel('Saturation Day (normalized)')
axes[1].set_title('Feature Space: Shape-Based Clustering')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('birdwatcher_clustering.png', dpi=100)
print("\n✓ Plot saved to birdwatcher_clustering.png")
