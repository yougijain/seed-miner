import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(42)

# Generate synthetic disc golf data: 
# - 3 skill tiers (beginner, intermediate, expert)
# - 2 course layouts (forward, reverse routing)
# - Score variance *only sensitive to true skill, not layout*
n_players = 24
n_rounds_per_layout = 12

skill_tiers = []
tier_labels = []
for tier_id, (tier_name, skill_sigma, base_score) in enumerate([
    ('beginner', 8.0, 95),
    ('intermediate', 5.0, 75),
    ('expert', 3.0, 60)
]):
    n_tier = n_players // 3
    true_skills = np.random.normal(0, skill_sigma, n_tier)
    for player_id in range(n_tier):
        skill_tiers.append(true_skills[player_id])
        tier_labels.append(tier_name)

rows = []
for player_id in range(n_players):
    for round_num in range(n_rounds_per_layout):
        # Layout A (forward): standard course
        score_a = 75 + skill_tiers[player_id] + np.random.normal(0, 2.0)
        rows.append({
            'player_id': player_id,
            'layout': 'A',
            'round': round_num,
            'score': score_a,
            'true_tier': tier_labels[player_id]
        })
        # Layout B (reverse): routing is *different geometry* 
        # but skill effect is the same; layout difficulty is uncorrelated with player skill ranking
        score_b = 77 + skill_tiers[player_id] + np.random.normal(0, 2.0)
        rows.append({
            'player_id': player_id,
            'layout': 'B',
            'round': round_num,
            'score': score_b,
            'true_tier': tier_labels[player_id]
        })

df = pd.DataFrame(rows)

# Compute per-player score variance within each layout
layout_variance = df.groupby(['player_id', 'layout'])['score'].var().reset_index()
layout_variance.columns = ['player_id', 'layout', 'variance']

# Key insight: build a graph where nodes are players,
# and edges connect players whose variance-response to layout-change is similar
# (i.e., both stable or both volatile across layouts)
pivot_var = layout_variance.pivot(index='player_id', columns='layout', values='variance').fillna(0)
pivot_var['var_response'] = np.abs(pivot_var['A'] - pivot_var['B'])
pivot_var['avg_var'] = (pivot_var['A'] + pivot_var['B']) / 2.0

# Similarity: players whose variance-response to layout changes is *similar* 
# are more likely to share true skill tier
G = nx.Graph()
for pid in range(n_players):
    G.add_node(pid)

# Connect players whose layout-variance-sensitivity is correlated
sensitivity = pivot_var['var_response'].values
scaler = StandardScaler()
sens_scaled = scaler.fit_transform(sensitivity.reshape(-1, 1)).flatten()

thresh = 0.5  # Threshold for 'similar sensitivity'
for i in range(n_players):
    for j in range(i + 1, n_players):
        if np.abs(sens_scaled[i] - sens_scaled[j]) < thresh:
            G.add_edge(i, j, weight=1.0 - np.abs(sens_scaled[i] - sens_scaled[j]) / thresh)

# Detect clusters via modularity-based community detection
from networkx.algorithms import community
clusters = list(community.greedy_modularity_communities(G))

print(f"Detected {len(clusters)} clusters from layout-variance-response graph:")
for cluster_id, cluster in enumerate(clusters):
    members = list(cluster)
    true_tiers = [tier_labels[pid] for pid in members]
    print(f"  Cluster {cluster_id}: {len(members)} players")
    print(f"    True tiers: {set(true_tiers)}")
    print(f"    Player IDs: {sorted(members)[:5]}...")

# Evaluate: do detected clusters match true skill tiers?
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

predicted_labels = np.zeros(n_players)
for cluster_id, cluster in enumerate(clusters):
    for pid in cluster:
        predicted_labels[pid] = cluster_id

true_numeric = pd.factorize(tier_labels)[0]
ari = adjusted_rand_score(true_numeric, predicted_labels)
nmi = normalized_mutual_info_score(true_numeric, predicted_labels)

print(f"\nCluster quality (ARI={ari:.3f}, NMI={nmi:.3f}):")
print(f"  If routing confounded player clustering, ARI would be low (~0.1).")
print(f"  If layout-variance-response graph captures true skill, ARI > 0.6.")
