import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Generate synthetic tournament data with hidden unfairness
n_players = 24
n_rounds = 5
rating_groups = {
    'elite': (1900, 2100, 6),      # 6 elite players
    'strong': (1600, 1800, 8),     # 8 strong players
    'mid': (1300, 1500, 7),        # 7 mid-level
    'junior': (1000, 1200, 3)      # 3 junior players
}

players = []
ratings = []
group_labels = []

for group_name, (r_min, r_max, count) in rating_groups.items():
    for i in range(count):
        players.append(f"{group_name}_{i}")
        ratings.append(np.random.uniform(r_min, r_max))
        group_labels.append(group_name)

ratings = np.array(ratings)

# Generate pairing sequences: UNFAIR structure
# Elite players face easy sequences (strong → mid → strong)
# Junior players face hard sequences (elite → strong → elite)
opponent_sequences = []

for i, (player, rating) in enumerate(zip(players, ratings)):
    group = group_labels[i]
    opponent_ratings = []
    
    if group == 'elite':
        # Systematically easy pairings
        opponent_ratings = [
            np.random.uniform(1600, 1800),  # Round 1: strong
            np.random.uniform(1300, 1500),  # Round 2: mid
            np.random.uniform(1600, 1800),  # Round 3: strong
            np.random.uniform(1200, 1400),  # Round 4: weak
            np.random.uniform(1500, 1700)   # Round 5: mid-strong
        ]
    elif group == 'junior':
        # Systematically hard pairings
        opponent_ratings = [
            np.random.uniform(1900, 2050),  # Round 1: elite
            np.random.uniform(1700, 1850),  # Round 2: strong
            np.random.uniform(1900, 2050),  # Round 3: elite
            np.random.uniform(1700, 1850),  # Round 4: strong
            np.random.uniform(1850, 2000)   # Round 5: strong-elite
        ]
    else:
        # Balanced (no unfairness)
        opponent_ratings = [
            np.random.uniform(rating - 200, rating + 200)
            for _ in range(n_rounds)
        ]
    
    opponent_sequences.append(opponent_ratings)

opponent_sequences = np.array(opponent_sequences)

# Normalize sequences (zscore across rounds for each player)
seq_normalized = np.array([zscore(seq) for seq in opponent_sequences])

# Cluster on opponent-rating sequences
linkage_matrix = AgglomerativeClustering(
    n_clusters=None, linkage='ward', distance_threshold=0
).fit_predict(seq_normalized)

# Count number of clusters that emerge
from sklearn.cluster import AgglomerativeClustering as AC
clustering = AC(n_clusters=None, linkage='ward', distance_threshold=2.0)
clusters = clustering.fit_predict(seq_normalized)
n_clusters = len(np.unique(clusters))

print(f"Tournament Pairing Fairness Analysis")
print(f"=====================================")
print(f"Players: {n_players} | Rounds: {n_rounds}")
print(f"Opponent-Sequence Clustering: {n_clusters} cohorts detected")
print()

# Analyze cohort composition
cohort_composition = {}
for cluster_id in np.unique(clusters):
    cohort_members = [players[i] for i, c in enumerate(clusters) if c == cluster_id]
    cohort_groups = [group_labels[i] for i, c in enumerate(clusters) if c == cluster_id]
    avg_opp_rating = opponent_sequences[clusters == cluster_id].mean()
    
    cohort_composition[cluster_id] = {
        'members': cohort_members,
        'group_composition': cohort_groups,
        'avg_opponent_rating': avg_opp_rating
    }

print("Cohort Breakdown:")
for cid, info in cohort_composition.items():
    print(f"  Cohort {cid}: {len(info['members'])} players | "
          f"Groups: {set(info['group_composition'])} | "
          f"Avg Opponent Rating: {info['avg_opponent_rating']:.1f}")

print()
print("FAIRNESS INTERPRETATION:")
if n_clusters > 1:
    avg_ratings_by_cohort = sorted(
        [(cid, info['avg_opponent_rating']) for cid, info in cohort_composition.items()],
        key=lambda x: x[1]
    )
    print(f"  ⚠ Clustering fractured into {n_clusters} disconnected cohorts.")
    print(f"  ⚠ Easier cohort avg opponent: {avg_ratings_by_cohort[0][1]:.1f}")
    print(f"  ⚠ Harder cohort avg opponent: {avg_ratings_by_cohort[-1][1]:.1f}")
    print(f"  ⚠ Difference: {avg_ratings_by_cohort[-1][1] - avg_ratings_by_cohort[0][1]:.1f} rating points")
    print(f"  → Tournament structure biases certain skill groups into systematically")
    print(f"    easier or harder pairings despite per-round balance.")
else:
    print(f"  ✓ Single cohort: no clustering collapse. Tournament likely balanced.")
