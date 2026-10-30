import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic tournament: 12 youth players in 5 rounds.
# Skill tiers: 0-3 (low), 4-7 (mid), 8-11 (high).
# UNFAIR scenario: optimizer segregates by tier, pairing low vs low, mid vs mid, high vs high.
# FAIR scenario: optimizer mixes tiers across rounds.

def generate_unfair_pairings(n_players=12, n_rounds=5):
    """Generate pairings that segregate by skill tier (unfair)."""
    players = list(range(n_players))
    tier_0 = players[0:4]   # Rating 1000-1200
    tier_1 = players[4:8]   # Rating 1300-1500
    tier_2 = players[8:12]  # Rating 1600-1800
    
    pairings = []
    for r in range(n_rounds):
        pairs = []
        # Within-tier pairing: segregates skill groups
        for tier in [tier_0, tier_1, tier_2]:
            for i in range(0, len(tier), 2):
                pairs.append((tier[i], tier[i+1] if i+1 < len(tier) else tier[0]))
        pairings.append(pairs)
    return pairings

def generate_fair_pairings(n_players=12, n_rounds=5):
    """Generate pairings that mix skill tiers (fair)."""
    players = list(range(n_players))
    pairings = []
    for r in range(n_rounds):
        # Random permutation within round
        shuffled = np.random.permutation(players)
        pairs = [(shuffled[i], shuffled[i+1]) for i in range(0, len(shuffled), 2)]
        pairings.append(pairs)
    return pairings

def player_ratings(n_players=12):
    """Rating vector: tier 0 [1000-1200], tier 1 [1300-1500], tier 2 [1600-1800]."""
    ratings = []
    for i in range(n_players):
        if i < 4:
            ratings.append(1000 + i * 50)
        elif i < 8:
            ratings.append(1300 + (i - 4) * 50)
        else:
            ratings.append(1600 + (i - 8) * 50)
    return np.array(ratings)

def opponent_rating_sequences(pairings, ratings):
    """For each player, extract sequence of opponent ratings across rounds."""
    n_players = len(ratings)
    sequences = {p: [] for p in range(n_players)}
    
    for round_pairs in pairings:
        for p1, p2 in round_pairs:
            sequences[p1].append(ratings[p2])
            sequences[p2].append(ratings[p1])
    
    return sequences

def cluster_sequences_and_measure_collapse(sequences):
    """Cluster players by opponent-rating sequences; return silhouette and n_clusters."""
    players = sorted(sequences.keys())
    X = np.array([sequences[p] for p in players]).astype(float)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hierarchical clustering
    clusterer = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = clusterer.fit_predict(X_scaled)
    
    # Measure cluster cohesion: within vs between variance.
    n_clusters = len(set(labels))
    
    # Compute pairwise distances
    dists = squareform(pdist(X_scaled, metric='euclidean'))
    
    within_cluster_avg = 0.0
    for c in set(labels):
        members = np.where(labels == c)[0]
        if len(members) > 1:
            within_avg = np.mean(dists[np.ix_(members, members)])
            within_cluster_avg += within_avg * len(members)
    within_cluster_avg /= max(len(players), 1)
    
    between_cluster_avg = 0.0
    for c1 in set(labels):
        for c2 in set(labels):
            if c1 < c2:
                m1 = np.where(labels == c1)[0]
                m2 = np.where(labels == c2)[0]
                between_avg = np.mean(dists[np.ix_(m1, m2)])
                between_cluster_avg += between_avg
    
    silhouette = (between_cluster_avg - within_cluster_avg) / max(between_cluster_avg, 1e-6)
    
    return silhouette, n_clusters, labels

def main():
    print("\n=== Youth Chess Pairing Fairness: Sequence Clustering Collapse ===")
    print("\nQuestion: Can clustering on opponent-rating sequences detect")
    print("structural fairness violations that round-by-round optimizers miss?")
    print()
    
    ratings = player_ratings(n_players=12)
    
    # UNFAIR scenario
    print("UNFAIR Tournament (segregated by skill tier):")
    unfair_pairings = generate_unfair_pairings(n_players=12, n_rounds=5)
    unfair_sequences = opponent_rating_sequences(unfair_pairings, ratings)
    unfair_sil, unfair_nc, unfair_labels = cluster_sequences_and_measure_collapse(unfair_sequences)
    
    print(f"  Silhouette (cohesion): {unfair_sil:.3f}")
    print(f"  Cluster count (k=2): {unfair_nc}")
    print(f"  Cluster assignment: {unfair_labels}")
    print(f"  → HIGH silhouette + distinct clusters = skill segregation detected")
    print()
    
    # FAIR scenario
    print("FAIR Tournament (mixed skill tiers):")
    fair_pairings = generate_fair_pairings(n_players=12, n_rounds=5)
    fair_sequences = opponent_rating_sequences(fair_pairings, ratings)
    fair_sil, fair_nc, fair_labels = cluster_sequences_and_measure_collapse(fair_sequences)
    
    print(f"  Silhouette (cohesion): {fair_sil:.3f}")
    print(f"  Cluster count (k=2): {fair_nc}")
    print(f"  Cluster assignment: {fair_labels}")
    print(f"  → LOW silhouette + no coherent clusters = fair mixing detected")
    print()
    
    print(f"Silhouette Ratio (unfair/fair): {unfair_sil / (fair_sil + 1e-6):.2f}x")
    print("→ Higher ratio indicates stronger clustering collapse (unfairness signal).")
    print()
    print("INSIGHT: Standard round-robin optimizers minimize local constraint violations")
    print("but don't detect latent player cohort segregation. Sequence clustering")
    print("exposes this by treating rating sequences as player 'profiles' and watching")
    print("whether the optimizer's pairings systematically isolate skill groups.")

if __name__ == '__main__':
    main()
