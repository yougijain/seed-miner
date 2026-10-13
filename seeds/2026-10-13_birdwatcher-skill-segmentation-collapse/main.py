import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate synthetic birdwatcher sighting data with realistic skill-dependent discovery curves
def generate_watcher_data(n_watchers=50, n_species=300, skill_levels=[0.3, 0.6, 0.9]):
    """
    Create cumulative species discovery over 12 outings.
    Novices: linear slow discovery (low exponent).
    Intermediate: faster discovery in common species, plateau on rare.
    Expert: aggressive rare-species targeting (power-law curvature).
    """
    data = []
    for watcher_id in range(n_watchers):
        # Assign latent skill level
        skill = np.random.choice(skill_levels)
        
        # Discovery curve: rarity-weighted sampling with skill-dependent exponent
        # Experts have steeper curves (find rare birds faster)
        rarity_exponent = 1.0 + 2.0 * skill  # 1.0 (novice) to 3.0 (expert)
        
        # Simulate 12 outings; cumulative species discovered
        cumulative = []
        discovered = set()
        for outing in range(1, 13):
            # Each outing: skill determines how many *new* species found
            # Exponent shapes discovery trajectory
            new_finds = int(np.random.poisson(lam=5 + 15 * (skill ** 1.5)))
            
            # Sample species weighted by rarity (power-law)
            species_ranks = np.arange(1, n_species + 1)
            rarity_weights = 1.0 / (species_ranks ** (rarity_exponent / 2))
            rarity_weights /= rarity_weights.sum()
            
            sampled = np.random.choice(n_species, size=min(new_finds, n_species), 
                                       replace=False, p=rarity_weights)
            discovered.update(sampled)
            cumulative.append(len(discovered))
        
        # Extract features from cumulative discovery curve
        cumulative = np.array(cumulative, dtype=float)
        
        # Feature 1: final count
        final_count = cumulative[-1]
        
        # Feature 2: curvature (second derivative) — measures how aggressively discovery accelerates
        diffs = np.diff(cumulative)
        second_diff = np.diff(diffs)
        curvature = np.mean(np.abs(second_diff[1:]))
        
        # Feature 3: rarity bias — ratio of discoveries in second half vs first half
        # (experts front-load rare birds; novices plateau)
        first_half = cumulative[5]
        second_half = cumulative[-1] - cumulative[5]
        rarity_bias = second_half / max(first_half, 1)
        
        data.append({
            'watcher_id': watcher_id,
            'skill_true': skill,
            'final_count': final_count,
            'curvature': curvature,
            'rarity_bias': rarity_bias,
            'cumulative_list': ','.join(map(str, cumulative))
        })
    
    return pd.DataFrame(data)

# Main analysis
df = generate_watcher_data()

# Prepare features for clustering
X = df[['final_count', 'curvature', 'rarity_bias']].values
X_scaled = StandardScaler().fit_transform(X)

# Attempt K-means segmentation across range of k
silhouette_scores = []
inertias = []

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    inertias.append(km.inertia_)

# Diagnose clustering quality
print("\n=== BIRDWATCHER SKILL SEGMENTATION ===")
print(f"N watchers: {len(df)}")
print(f"\nSilhouette scores by k:")
for k, score in enumerate(silhouette_scores, start=2):
    print(f"  k={k}: {score:.3f}")

best_k = np.argmax(silhouette_scores) + 2
print(f"\nBest k (silhouette): {best_k}")

# Re-fit with best k and examine cluster composition
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_best.fit_predict(X_scaled)

print(f"\nCluster composition (by true skill level):")
for cluster_id in range(best_k):
    cluster_df = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} (n={len(cluster_df)}):")
    print(f"  Mean final_count: {cluster_df['final_count'].mean():.1f}")
    print(f"  Mean curvature: {cluster_df['curvature'].mean():.2f}")
    print(f"  Mean rarity_bias: {cluster_df['rarity_bias'].mean():.2f}")
    print(f"  True skill composition: {cluster_df['skill_true'].value_counts().to_dict()}")

# Segmentation collapse diagnosis
print(f"\n=== SEGMENTATION COLLAPSE DIAGNOSIS ===")
if best_k < 3:
    print("⚠ COLLAPSE: Clustering cannot recover latent skill tiers.")
    print("  Reason: Expert discovery curves are too similar to intermediate curves")
    print("  in the (final_count, curvature, rarity_bias) space.")
    print("  Standard clustering feature space fails to capture power-law geometry.")
else:
    print("✓ Segmentation succeeded: skill tiers recovered.")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter: curvature vs rarity_bias, colored by cluster
axes[0].scatter(df['curvature'], df['rarity_bias'], c=df['cluster'], 
                cmap='tab10', s=60, alpha=0.6, edgecolors='k')
axes[0].set_xlabel('Curvature (discovery acceleration)')
axes[0].set_ylabel('Rarity Bias (second-half / first-half discovery)')
axes[0].set_title(f'Segmentation Result (k={best_k})')
axes[0].grid(alpha=0.3)

# Scatter: final_count vs curvature, colored by true skill
scatter = axes[1].scatter(df['final_count'], df['curvature'], 
                          c=df['skill_true'], cmap='YlOrRd', s=60, alpha=0.6, edgecolors='k')
axes[1].set_xlabel('Final Species Count')
axes[1].set_ylabel('Curvature')
axes[1].set_title('True Skill Levels (ground truth)')
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label('Skill Level')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('birdwatcher_segmentation.png', dpi=100)
print(f"\nPlot saved to birdwatcher_segmentation.png")
