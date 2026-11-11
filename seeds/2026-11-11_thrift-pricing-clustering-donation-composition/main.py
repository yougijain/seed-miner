import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic dataset: items with (category, condition, price) where price is set by
# an unobserved rule that depends on donation source.
# Donation source 1: estate sales → high condition → higher base prices
# Donation source 2: thrift-donor mix → medium condition → lower base prices

def generate_inventory(n_items, source_mix, base_price_multiplier=1.0):
    """Generate item prices influenced by donation source composition."""
    categories = np.random.choice(['books', 'clothing', 'furniture', 'electronics'], n_items)
    conditions = np.random.choice([1, 2, 3, 4, 5], n_items)
    
    # Base pricing rule: condition + category effect
    prices = conditions * 2 + np.random.normal(0, 1, n_items)
    
    # Donation source effect: high source_mix → shift toward high-condition donors
    # This changes the average condition *and* introduces price bias
    if source_mix > 0.5:
        # Estate-sale bias: condition boosted, price bumped
        condition_boost = np.random.binomial(1, source_mix, n_items) * 1.5
        prices += condition_boost * 3
    else:
        # Thrift-donor bias: prices compressed
        prices *= 0.8 + np.random.uniform(0, 0.2, n_items)
    
    prices = np.maximum(prices, 0.5)  # floor at 0.5
    return pd.DataFrame({
        'category': categories,
        'condition': conditions + (1.0 if source_mix > 0.5 else 0.0),  # composition shift
        'price': prices
    })

# Phase 1: stable donation source (source_mix=0.3, mostly thrift donors)
print("Phase 1: Stable donation source (thrift donors)")
inv_phase1 = generate_inventory(200, source_mix=0.3)
features_p1 = inv_phase1[['condition', 'price']].values
scaler = StandardScaler()
features_p1_scaled = scaler.fit_transform(features_p1)

# Cluster on price + condition
kmeans_p1 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_p1 = kmeans_p1.fit_predict(features_p1_scaled)
inv_phase1['cluster'] = labels_p1

print(f"Phase 1 cluster sizes: {np.bincount(labels_p1)}")
print(f"Phase 1 cluster centers (unscaled):\n{kmeans_p1.cluster_centers_}")
print()

# Phase 2: donation source shifts (source_mix=0.8, mostly estate sales)
print("Phase 2: Donation source shift (estate sales)")
inv_phase2 = generate_inventory(200, source_mix=0.8)
features_p2 = inv_phase2[['condition', 'price']].values

# Use SAME scaler fitted on phase 1 (as if we're analyzing streaming prices)
features_p2_scaled = scaler.transform(features_p2)

# Apply same K-means (no retraining) to detect clustering instability
labels_p2_same_model = kmeans_p1.predict(features_p2_scaled)
inv_phase2['cluster_same_model'] = labels_p2_same_model

print(f"Phase 2 cluster sizes (same model, no retrain): {np.bincount(labels_p2_same_model)}")
print()

# Now retrain on phase 2 to show how clustering *would* change
kmeans_p2 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_p2_retrained = kmeans_p2.fit_predict(features_p2_scaled)
inv_phase2['cluster_retrained'] = labels_p2_retrained

print(f"Phase 2 cluster sizes (retrained): {np.bincount(labels_p2_retrained)}")
print(f"Phase 2 cluster centers (retrained, unscaled):\n{kmeans_p2.cluster_centers_}")
print()

# Diagnostic: compute silhouette-like instability metric
# For each item type, check how often it switches clusters between same_model vs retrained
switches = (labels_p2_same_model != labels_p2_retrained).sum()
instability = switches / len(labels_p2_same_model)

print(f"Clustering instability (items reassigned): {instability:.2%}")
print(f"Interpretation: High instability suggests donation-source regime shift, not just noise.")
print()

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(inv_phase1['condition'], inv_phase1['price'], c=labels_p1, cmap='viridis', alpha=0.6)
axes[0].set_title('Phase 1: Stable Source (Thrift)')
axes[0].set_xlabel('Condition')
axes[0].set_ylabel('Price')

axes[1].scatter(inv_phase2['condition'], inv_phase2['price'], c=labels_p2_same_model, cmap='viridis', alpha=0.6)
axes[1].set_title('Phase 2: Same Model (No Retrain)')
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Price')

axes[2].scatter(inv_phase2['condition'], inv_phase2['price'], c=labels_p2_retrained, cmap='viridis', alpha=0.6)
axes[2].set_title('Phase 2: Retrained (New Regimes)')
axes[2].set_xlabel('Condition')
axes[2].set_ylabel('Price')

plt.tight_layout()
plt.savefig('clustering_instability.png', dpi=100)
print("Saved clustering_instability.png")
