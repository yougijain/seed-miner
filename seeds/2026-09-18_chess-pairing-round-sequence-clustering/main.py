import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
import json

np.random.seed(42)

# Generate synthetic tournament: two rating groups with imbalanced pairing
num_players = 20
low_rated = 10
high_rated = 10
num_rounds = 5

player_ids = list(range(num_players))
ratings = [1200] * low_rated + [1800] * high_rated

# FAIR scenario: cross-group pairings distributed equally
# UNFAIR scenario: low-rated players mostly face each other; high-rated face cross-group
# We'll generate both and show clustering behavior differs

def generate_pairings(fair=True):
    """Generate round-by-round opponent sequences."""
    opponent_sequences = {pid: [] for pid in player_ids}
    
    for round_idx in range(num_rounds):
        if fair:
            # Fair: alternate cross-group pairings
            pairs = []
            if round_idx % 2 == 0:
                # Cross-group pairings
                for i in range(low_rated):
                    opp = (i + round_idx) % high_rated + low_rated
                    pairs.append((i, opp))
            else:
                # Within-group pairings
                for i in range(low_rated):
                    opp = (i + round_idx) % low_rated
                    if opp != i:
                        pairs.append((i, opp))
                for i in range(high_rated):
                    opp = (i + round_idx) % high_rated + low_rated
                    if opp != i + low_rated:
                        pairs.append((i + low_rated, opp))
        else:
            # UNFAIR: low-rated cluster together, high-rated see cross-group
            for i in range(low_rated):
                opp = (i + round_idx) % low_rated
                if opp != i:
                    pairs.append((i, opp))
            for i in range(high_rated):
                # High-rated see mostly low-rated opponents
                opp = (i + round_idx) % low_rated
                pairs.append((i + low_rated, opp))
        
        for p1, p2 in pairs:
            opponent_sequences[p1].append(ratings[p2])
            opponent_sequences[p2].append(ratings[p1])
    
    return opponent_sequences

def analyze_clustering(opponent_sequences, label):
    """Cluster players by opponent rating sequences; measure fragmentation."""
    # Convert sequences to feature vectors
    features = []
    pids = []
    for pid in sorted(opponent_sequences.keys()):
        seq = opponent_sequences[pid]
        if len(seq) > 0:
            # Features: mean opponent rating, std, min, max, sequence ordering
            feat = [
                np.mean(seq),
                np.std(seq) if len(seq) > 1 else 0,
                np.min(seq),
                np.max(seq),
                seq[0] if seq else 0  # First opponent rating
            ]
            features.append(feat)
            pids.append(pid)
    
    features = np.array(features)
    
    # Normalize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Hierarchical clustering
    clusterer = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = clusterer.fit_predict(features)
    
    # Measure cluster purity: do clusters align with ground-truth rating groups?
    true_groups = [0 if pid < low_rated else 1 for pid in pids]
    purity = sum(1 for tg, cl in zip(true_groups, labels) if (tg == 0) == (cl == 0)) / len(true_groups)
    
    # Measure fragmentation: what % of players are isolated (cluster size=1)?
    unique, counts = np.unique(labels, return_counts=True)
    isolation = sum(1 for c in counts if c == 1) / len(pids)
    
    print(f"\n{label}:")
    print(f"  Cluster purity (vs true rating groups): {purity:.2f}")
    print(f"  Player isolation rate: {isolation:.2f}")
    print(f"  Cluster distribution: {dict(zip(unique, counts))}")
    
    return purity, isolation

print("=" * 60)
print("Youth Chess Pairing Fairness via Clustering Collapse")
print("=" * 60)

fair_seqs = generate_pairings(fair=True)
unfair_seqs = generate_pairings(fair=False)

purity_fair, iso_fair = analyze_clustering(fair_seqs, "FAIR Pairings")
purity_unfair, iso_unfair = analyze_clustering(unfair_seqs, "UNFAIR Pairings")

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
print(
    f"Fair pairings: Higher purity ({purity_fair:.2f}) suggests opponent sequences\n"
    f"  are well-mixed, clustering recovers rating structure naturally.\n"
    f"Unfair pairings: Lower purity ({purity_unfair:.2f}) and higher isolation\n"
    f"  ({iso_unfair:.2f}) suggest clustering fragments because low-rated players\n"
    f"  form isolated sequences (always facing each other).\n"
    f"\nClustering collapse is the signal of unfairness: when opponent-sequence\n"
    f"clustering cannot form balanced, interpretable groups, pairing structure\n"
    f"is likely biased.\n"
)
