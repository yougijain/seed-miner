import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import json

np.random.seed(42)

# Generate synthetic little league schedule: 8 teams, 10 game weeks
# Schedule assigns each team a binary rest pattern (1=rest, 0=play) across weeks
teams = [f'Team_{i}' for i in range(1, 9)]
weeks = list(range(1, 11))

# Fair schedule: rest days distributed uniformly
fair_rest = np.array([
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],  # even distribution
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
])

# Unfair schedule: cluster teams into two "blocks" with synchronized rest patterns
unfair_rest = np.vstack([
    # Block A (easier): rest on odd weeks
    np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0] for _ in range(4)]),
    # Block B (harder): rest on even weeks
    np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for _ in range(4)]),
])

def detect_rest_clustering(rest_matrix, schedule_name):
    """
    Detect anomalous rest-day synchronization via clustering of rest-pattern sequences.
    High clustering = unfair synchronization (anomaly in fairness context).
    """
    # Convert each team's rest sequence to a feature vector
    # Compute pairwise cosine-similarity of rest patterns
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity = cosine_similarity(rest_matrix)
    np.fill_diagonal(similarity, 0)  # ignore self-similarity
    
    # Flatten similarity matrix to use as "features" for anomaly detection
    # High inter-team similarity is anomalous (should be random/fair)
    # Convert to distance for DBSCAN
    distances = 1 - similarity
    
    # Cluster teams by their rest-pattern similarity
    # Anomaly: clusters form (indicates synchronized subgroups)
    clusterer = DBSCAN(eps=0.15, min_samples=2)
    labels = clusterer.fit_predict(rest_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Compute mean pairwise similarity within clusters (higher = more synchronized = more unfair)
    if n_clusters > 1:
        cluster_cohesion = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            if mask.sum() > 1:
                cluster_sim = similarity[np.ix_(mask, mask)].mean()
                cluster_cohesion.append(cluster_sim)
        fairness_score = np.mean(cluster_cohesion) if cluster_cohesion else 0.0
    else:
        fairness_score = similarity.mean()  # baseline uniformity
    
    print(f"\n{schedule_name}")
    print(f"  Clusters detected: {n_clusters}")
    print(f"  Noise points (fair teams): {n_noise}")
    print(f"  Mean rest-pattern similarity: {fairness_score:.3f}")
    print(f"  Interpretation: {'UNFAIR (synchronized blocks)' if fairness_score > 0.6 else 'FAIR (dispersed patterns)'}")
    print(f"  Team cluster assignment: {dict(zip(teams, labels))}")
    
    return {
        'schedule': schedule_name,
        'clusters': n_clusters,
        'fairness_score': fairness_score,
        'labels': labels.tolist()
    }

# Analyze both schedules
fair_result = detect_rest_clustering(fair_rest, "Fair Schedule")
unfair_result = detect_rest_clustering(unfair_rest, "Unfair Schedule (Clustered Blocks)")

# Export results
results = {'fair': fair_result, 'unfair': unfair_result}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("ANOMALY DETECTION INSIGHT:")
print("Fair schedules → low cluster cohesion (anomaly: isolated teams)")
print("Unfair schedules → high cluster cohesion (anomaly: synchronized blocks)")
print("Inversion: we detect fairness violations by finding EXCESSIVE synchronization.")
print("="*60)
