import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic dataset: 30 board games with sessions, checkout duration, and demand signals
# The confound: high-demand short-duration games cluster differently when duration is ignored

games = []
for i in range(30):
    game_id = f"G{i:02d}"
    # True demand tier: 0=low, 1=mid, 2=high
    true_tier = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
    
    # Base demand: sessions per week
    base_demand = [2.5, 7.5, 13][true_tier] + np.random.normal(0, 1.5)
    base_demand = max(1, base_demand)
    
    # Checkout duration: *endogenously* longer games have fewer demand signals
    # but are truly high-demand (e.g., Gloomhaven: 2hr game, super popular)
    duration_base = [30, 70, 150][true_tier]  # min base by tier
    checkout_duration = duration_base + np.random.normal(0, 20)
    checkout_duration = max(15, checkout_duration)
    
    # Availability windows per week (how many times it could be booked)
    # Confound: long-duration games have fewer available windows
    weekly_windows = max(3, 20 - (checkout_duration / 20))
    weekly_windows += np.random.normal(0, 1)
    
    # Observed checkout events: true demand × windows (the confound)
    observed_checkouts = (base_demand / 10) * weekly_windows + np.random.normal(0, 1)
    observed_checkouts = max(0, observed_checkouts)
    
    # Other features: player count, complexity, table footprint
    player_count = np.random.randint(2, 7)
    complexity = np.random.uniform(1, 5)
    footprint_sq_ft = checkout_duration / 30 + np.random.normal(0, 1)  # longer games need more space
    footprint_sq_ft = max(1, footprint_sq_ft)
    
    games.append({
        'game_id': game_id,
        'true_demand_tier': true_tier,
        'observed_checkouts': observed_checkouts,
        'checkout_duration_min': checkout_duration,
        'weekly_windows': weekly_windows,
        'player_count': player_count,
        'complexity': complexity,
        'footprint_sq_ft': footprint_sq_ft,
    })

df = pd.DataFrame(games)

print("=" * 70)
print("BOARD GAME CAFE DEMAND CLUSTERING UNDER CHECKOUT-DURATION ENDOGENEITY")
print("=" * 70)
print(f"\nDataset: {len(df)} games")
print(f"True demand tiers (ground truth): {df['true_demand_tier'].value_counts().sort_index().to_dict()}")
print(f"\nFeature correlations with checkout_duration:")
print(df[['observed_checkouts', 'weekly_windows', 'footprint_sq_ft']].corrwith(df['checkout_duration_min']))

# Clustering Scenario 1: NAIVE (observed features, duration is confound)
features_naive = ['observed_checkouts', 'player_count', 'complexity', 'footprint_sq_ft']
X_naive = df[features_naive].values
X_naive_scaled = StandardScaler().fit_transform(X_naive)
kmeans_naive = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_naive'] = kmeans_naive.fit_predict(X_naive_scaled)
sil_naive = silhouette_score(X_naive_scaled, df['cluster_naive'])

# Clustering Scenario 2: CONFOUND-AWARE (remove duration effect from observed_checkouts)
# Adjust observed checkouts by dividing out the window-availability confound
df['adjusted_demand'] = df['observed_checkouts'] / (df['weekly_windows'] + 0.1)
features_adjusted = ['adjusted_demand', 'player_count', 'complexity']
X_adjusted = df[features_adjusted].values
X_adjusted_scaled = StandardScaler().fit_transform(X_adjusted)
kmeans_adjusted = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_adjusted'] = kmeans_adjusted.fit_predict(X_adjusted_scaled)
sil_adjusted = silhouette_score(X_adjusted_scaled, df['cluster_adjusted'])

# Clustering Scenario 3: DURATION-EXPLICIT (include duration as a feature for explicit control)
features_explicit = ['observed_checkouts', 'player_count', 'complexity', 'checkout_duration_min']
X_explicit = df[features_explicit].values
X_explicit_scaled = StandardScaler().fit_transform(X_explicit)
kmeans_explicit = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_explicit'] = kmeans_explicit.fit_predict(X_explicit_scaled)
sil_explicit = silhouette_score(X_explicit_scaled, df['cluster_explicit'])

print(f"\n" + "=" * 70)
print("CLUSTERING RESULTS")
print("=" * 70)
print(f"\n1. NAIVE (ignoring duration confound):")
print(f"   Silhouette Score: {sil_naive:.3f}")
print(f"   Cluster distribution: {pd.Series(df['cluster_naive']).value_counts().sort_index().to_dict()}")
print(f"\n2. CONFOUND-ADJUSTED (demand / windows):")
print(f"   Silhouette Score: {sil_adjusted:.3f}")
print(f"   Cluster distribution: {pd.Series(df['cluster_adjusted']).value_counts().sort_index().to_dict()}")
print(f"\n3. DURATION-EXPLICIT (duration as feature):")
print(f"   Silhouette Score: {sil_explicit:.3f}")
print(f"   Cluster distribution: {pd.Series(df['cluster_explicit']).value_counts().sort_index().to_dict()}")

# Cluster purity vs ground truth
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def cluster_purity(true_labels, pred_labels):
    """Compute cluster purity: fraction of samples with correct majority label."""
    cost_matrix = -np.array([[np.sum((true_labels == i) & (pred_labels == j)) 
                              for j in np.unique(pred_labels)] 
                             for i in np.unique(true_labels)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    correct = -cost_matrix[row_ind, col_ind].sum()
    return correct / len(true_labels)

purity_naive = cluster_purity(df['true_demand_tier'].values, df['cluster_naive'].values)
purity_adjusted = cluster_purity(df['true_demand_tier'].values, df['cluster_adjusted'].values)
purity_explicit = cluster_purity(df['true_demand_tier'].values, df['cluster_explicit'].values)

print(f"\nCLUSTER PURITY (vs ground-truth demand tiers):")
print(f"  Naive:        {purity_naive:.3f}")
print(f"  Adjusted:     {purity_adjusted:.3f}")
print(f"  Explicit:     {purity_explicit:.3f}")

print(f"\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print(f"""
The endogeneity problem: observed checkout count is confounded by checkout duration.
Long games (high duration) have fewer weekly windows, reducing observed count even if demand is high.
Naive clustering conflates duration with demand.

Adjusted approach normalizes by windows; Explicit approach treats duration as a control variable.
If Adjusted OR Explicit has higher purity than Naive, the confound is real and significant.
""")

df_out = df[['game_id', 'true_demand_tier', 'checkout_duration_min', 'observed_checkouts', 
             'cluster_naive', 'cluster_adjusted', 'cluster_explicit']].head(15)
print(f"\nSample results (first 15 games):")
print(df_out.to_string(index=False))
