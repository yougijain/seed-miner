import pandas as pd
import numpy as np
from itertools import combinations

np.random.seed(42)

# Generate synthetic little league schedule: 10 teams, 8 weeks, ~14 game slots per week
weeks = 8
teams = list(range(10))
slots_per_week = 14

schedule = []
for week in range(weeks):
    for slot in range(slots_per_week):
        if np.random.rand() < 0.4:  # ~40% occupancy
            t1, t2 = np.random.choice(teams, 2, replace=False)
            schedule.append({'week': week, 'slot': slot, 'team1': t1, 'team2': t2})

df_schedule = pd.DataFrame(schedule)

# Build rest-day vector for each team
rest_vectors = {}
for team in teams:
    plays = set()
    for _, row in df_schedule.iterrows():
        if row['team1'] == team or row['team2'] == team:
            plays.add(row['week'])
    rests = tuple(1 if w not in plays else 0 for w in range(weeks))
    rest_vectors[team] = rests

print("=== REST-DAY VECTORS (1=rest, 0=plays) ===")
for team, vec in sorted(rest_vectors.items()):
    print(f"Team {team}: {vec}")

# Compute pairwise rest-day congruence (Hamming similarity)
def congruence(v1, v2):
    return sum(a == b for a, b in zip(v1, v2)) / len(v1)

congruence_matrix = np.zeros((len(teams), len(teams)))
for i, t1 in enumerate(teams):
    for j, t2 in enumerate(teams):
        congruence_matrix[i][j] = congruence(rest_vectors[t1], rest_vectors[t2])

print("\n=== CONGRUENCE MATRIX ===")
print(pd.DataFrame(congruence_matrix, index=teams, columns=teams).round(2))

# Detect anomaly: if many pairs show suspiciously high congruence,
# schedule may have been constructed to deliberately balance rest days.
# Reverse matching insight: find the bipartition that minimizes congruence variance.

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering

# Treat congruence as 'cost' to avoid in a matching.
# High congruence = bad for fairness (implies teams were paired to have same rest pattern).
# Spectral clustering on congruence graph detects hidden team clusters with anomalous rest-pattern alignment.

dissimilarity = 1 - congruence_matrix
np.fill_diagonal(dissimilarity, 0)

sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
cluster_labels = sc.fit_predict(congruence_matrix)

print("\n=== SPECTRAL CLUSTERING ON REST CONGRUENCE ===")
for cluster_id in set(cluster_labels):
    teams_in_cluster = [t for t, c in zip(teams, cluster_labels) if c == cluster_id]
    print(f"Cluster {cluster_id}: {teams_in_cluster}")

# Fairness test: compute intra-cluster vs inter-cluster congruence
intra_congruence = []
inter_congruence = []

for i, t1 in enumerate(teams):
    for j, t2 in enumerate(teams):
        if i < j:
            cong = congruence_matrix[i][j]
            if cluster_labels[i] == cluster_labels[j]:
                intra_congruence.append(cong)
            else:
                inter_congruence.append(cong)

print(f"\n=== FAIRNESS DIAGNOSIS ===")
print(f"Mean intra-cluster congruence: {np.mean(intra_congruence):.3f}")
print(f"Mean inter-cluster congruence: {np.mean(inter_congruence):.3f}")
print(f"Gap (intra - inter): {np.mean(intra_congruence) - np.mean(inter_congruence):.3f}")

if np.mean(intra_congruence) - np.mean(inter_congruence) > 0.15:
    print("⚠️  ANOMALY DETECTED: Schedule may have been constructed to cluster")
    print("   teams with identical rest patterns, indicating possible unfairness.")
else:
    print("✓ Schedule appears fair: rest-day patterns are not suspiciously clustered.")
