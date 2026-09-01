import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Generate synthetic disc golf round data:
# - 40 players, 2 9-hole course routings (A, B)
# - Routing A is harder (+3 strokes avg), Routing B is easier
# - Players endogenously choose routing: better players pick harder routing more often
# - Score similarity alone would falsely match weak players (who chose easy routing) with strong players (who chose hard routing)

N_PLAYERS = 40
N_ROUNDS = 5

player_skill = np.random.normal(0, 1, N_PLAYERS)
routing_difficulty = {'A': 3, 'B': 0}  # Routing A is +3 strokes harder

data = []
for player_id in range(N_PLAYERS):
    for round_num in range(N_ROUNDS):
        # Routing choice: better players more likely to pick harder routing
        routing_prob_hard = 1 / (1 + np.exp(-player_skill[player_id]))
        routing = 'A' if np.random.random() < routing_prob_hard else 'B'
        
        # Score: skill + routing difficulty + noise
        score = 72 + player_skill[player_id] + routing_difficulty[routing] + np.random.normal(0, 1)
        data.append({
            'player_id': player_id,
            'round': round_num,
            'routing': routing,
            'skill_true': player_skill[player_id],
            'score': score
        })

df = pd.DataFrame(data)

# NAIVE MATCHING: pair by average score alone
player_avg_score = df.groupby('player_id')['score'].mean()
scaler = StandardScaler()
score_feat = scaler.fit_transform(player_avg_score.values.reshape(-1, 1))

# Compute pairwise distance
dist_naive = squareform(pdist(score_feat, metric='euclidean'))
np.fill_diagonal(dist_naive, np.inf)

# Greedy matching: pair closest players
pairings_naive = []
remaining = set(range(N_PLAYERS))
while len(remaining) >= 2:
    min_dist = np.inf
    best_pair = None
    for i in remaining:
        for j in remaining:
            if i < j and dist_naive[i, j] < min_dist:
                min_dist = dist_naive[i, j]
                best_pair = (i, j)
    if best_pair:
        pairings_naive.append(best_pair)
        remaining.discard(best_pair[0])
        remaining.discard(best_pair[1])

# CONFOUNDED ANALYSIS: naive matching pairs by score, ignoring routing choice
naive_skill_diff = np.mean([abs(player_skill[p1] - player_skill[p2]) for p1, p2 in pairings_naive])
naive_routing_align = np.mean([
    1.0 if (df[df['player_id']==p1]['routing'].mode()[0] == df[df['player_id']==p2]['routing'].mode()[0])
    else 0.0
    for p1, p2 in pairings_naive
])

# TWO-STAGE DECONFOUNDED MATCHING:
# Stage 1: Infer each player's modal routing and 'true' skill (score adjusted for routing)
player_modal_routing = df.groupby('player_id')['routing'].apply(lambda x: x.mode()[0])
player_adjusted_skill = df.groupby('player_id').apply(
    lambda grp: (grp['score'] - grp['routing'].map(routing_difficulty)).mean()
)

# Stage 2: Match on adjusted skill within routing strata
pairings_deconf = []
for routing_choice in ['A', 'B']:
    players_in_stratum = player_modal_routing[player_modal_routing == routing_choice].index.tolist()
    n_strat = len(players_in_stratum)
    if n_strat < 2:
        continue
    
    skills = player_adjusted_skill[players_in_stratum].values.reshape(-1, 1)
    skills_scaled = scaler.fit_transform(skills)
    dist_strat = squareform(pdist(skills_scaled, metric='euclidean'))
    np.fill_diagonal(dist_strat, np.inf)
    
    remaining_strat = set(range(n_strat))
    while len(remaining_strat) >= 2:
        min_d = np.inf
        best_idx = None
        for i in remaining_strat:
            for j in remaining_strat:
                if i < j and dist_strat[i, j] < min_d:
                    min_d = dist_strat[i, j]
                    best_idx = (i, j)
        if best_idx:
            actual_pair = (players_in_stratum[best_idx[0]], players_in_stratum[best_idx[1]])
            pairings_deconf.append(actual_pair)
            remaining_strat.discard(best_idx[0])
            remaining_strat.discard(best_idx[1])

# DECONFOUNDED ANALYSIS
deconf_skill_diff = np.mean([abs(player_skill[p1] - player_skill[p2]) for p1, p2 in pairings_deconf])
deconf_routing_align = np.mean([
    1.0 if (df[df['player_id']==p1]['routing'].mode()[0] == df[df['player_id']==p2]['routing'].mode()[0])
    else 0.0
    for p1, p2 in pairings_deconf
])

print("=== DISC GOLF PLAYER MATCHING UNDER ROUTING ENDOGENEITY ===")
print(f"\nNAIVE MATCHING (by score only):")
print(f"  True skill difference (paired players): {naive_skill_diff:.3f}")
print(f"  Routing alignment (both chose same routing): {naive_routing_align:.2%}")
print(f"  Pairs: {pairings_naive[:3]}...")

print(f"\nDECONFOUNDED MATCHING (stratified by routing, adjusted score):")
print(f"  True skill difference (paired players): {deconf_skill_diff:.3f}")
print(f"  Routing alignment: {deconf_routing_align:.2%}")
print(f"  Pairs: {pairings_deconf[:3]}...")

print(f"\nKEY FINDING:")
print(f"  Naive matching ignores that score is confounded by routing choice.")
print(f"  Deconfounded matching stratifies by routing first, then matches on skill-adjusted score.")
print(f"  Result: deconfounded pairs have {deconf_skill_diff/naive_skill_diff:.1f}x better true skill alignment.")
