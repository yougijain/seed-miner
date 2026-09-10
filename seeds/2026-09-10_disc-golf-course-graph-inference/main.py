import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import json

np.random.seed(42)

# Generate synthetic disc golf rounds.
# Course has 9 holes. Holes 3 and 6 are 'bottlenecks'—high variance, condition downstream scores.
def generate_rounds(n_rounds=200):
    rounds = []
    for _ in range(n_rounds):
        round_scores = []
        skill = np.random.uniform(0, 1)  # player skill
        for hole in range(9):
            base_score = 3  # par 3 for simplicity
            # Holes 3 and 6 are bottlenecks: high variance, depend on prior state
            if hole == 3:
                # Score depends on hole 2 outcome (conditioning)
                delta = (round_scores[2] - 3) * 0.5  # carry forward prior error
                noise = np.random.normal(0, 1.2)
                score = base_score + int(delta) + int(noise * (1 - skill))
            elif hole == 6:
                # Hole 6 depends on hole 5
                delta = (round_scores[5] - 3) * 0.5
                noise = np.random.normal(0, 1.2)
                score = base_score + int(delta) + int(noise * (1 - skill))
            else:
                # Non-bottleneck holes: independent
                noise = np.random.normal(0, 0.8)
                score = base_score + int(noise * (1 - skill))
            round_scores.append(max(2, min(6, score)))  # clamp to 2-6
        rounds.append(round_scores)
    return np.array(rounds)

rounds = generate_rounds(200)
print(f"Generated {len(rounds)} rounds, 9 holes each.")

# Compute pairwise distance between rounds using score sequences.
# Use Euclidean distance in 9D score space.
dists = squareform(pdist(rounds, metric='euclidean'))

# Cluster rounds at varying epsilon thresholds.
# As epsilon increases, clusters merge. The *pattern of merging* reveals hole dependencies.
eps_values = np.linspace(1.0, 8.0, 20)
cluster_counts = []
silhouette_scores = []

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
    labels = db.fit_predict(dists)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_counts.append(n_clusters)

# Detect the 'elbow' or discontinuity in clustering—where structure fragments.
# This indicates that holes that co-correlate (bottlenecks) are separating clusters.
cluster_diffs = np.diff(cluster_counts)
fragmentation_points = np.where(np.abs(cluster_diffs) > 2)[0]

print(f"\nCluster count by epsilon: {cluster_counts}")
print(f"Fragmentation detected at epsilon indices: {fragmentation_points}")

if len(fragmentation_points) > 0:
    critical_eps = eps_values[fragmentation_points[0]]
    print(f"\nCritical epsilon (clustering collapse): {critical_eps:.2f}")
    print("Hypothesis: holes 3 and 6 form dependency chains, causing clustering to fragment.")
    print("At low epsilon, all rounds cluster by global skill. As epsilon shrinks,")
    print("the bottleneck structure becomes visible: rounds separate by their hole-3 and hole-6 outcomes.")

# Verify by examining score variance at bottleneck holes.
print(f"\nHole-by-hole score variance:")
for hole in range(9):
    var = rounds[:, hole].var()
    label = " [BOTTLENECK]" if hole in [3, 6] else ""
    print(f"  Hole {hole}: {var:.3f}{label}")

# Export for visualization.
result = {
    "eps_values": eps_values.tolist(),
    "cluster_counts": cluster_counts,
    "fragmentation_points": fragmentation_points.tolist(),
    "hypothesis": "Bottleneck holes (3, 6) create score-sequence structure; clustering collapse reveals their network role."
}

with open("results.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nResults saved to results.json")
