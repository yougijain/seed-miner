import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

np.random.seed(42)

# Synthetic youth chess tournament: 24 players, 5 rounds.
# Each player has an initial ELO rating and performance variance.
# After each round, their rating shifts based on win/loss outcome (endogenous).
# We cluster players by their OPPONENT-STRENGTH sequence (avg rating of opponents faced so far)
# in each round separately. If clustering is stable, same players should stay in same clusters.
# If rating volatility is high, opponent-strength sequences diverge, causing cluster reassignment.

num_players = 24
num_rounds = 5
initial_ratings = np.random.normal(1200, 150, num_players)
initial_ratings = np.clip(initial_ratings, 800, 1800)

player_names = [f"P{i+1}" for i in range(num_players)]
playerdata = {
    "name": player_names,
    "initial_elo": initial_ratings,
    "volatility": np.random.uniform(10, 50, num_players),  # rating swing per round
}
df_players = pd.DataFrame(playerdata)

# Simulate 5 rounds of tournament pairings + outcomes.
rounds_data = []
current_ratings = initial_ratings.copy()

for round_idx in range(num_rounds):
    # Pair players: randomly pair them for this round (simplified Swiss-style).
    shuffled_idx = np.random.permutation(num_players)
    pairings = [
        (shuffled_idx[i], shuffled_idx[i + 1])
        for i in range(0, num_players, 2)
    ]
    
    # Simulate match outcomes: higher rating player more likely to win.
    round_opponents = [[] for _ in range(num_players)]
    for p1_idx, p2_idx in pairings:
        win_prob = 1 / (1 + 10 ** (-(current_ratings[p1_idx] - current_ratings[p2_idx]) / 400))
        p1_wins = np.random.rand() < win_prob
        
        # Update ratings based on outcome (Elo-like swing).
        rating_change_p1 = df_players.loc[p1_idx, "volatility"] * (1 if p1_wins else -1)
        rating_change_p2 = df_players.loc[p2_idx, "volatility"] * (-1 if p1_wins else 1)
        current_ratings[p1_idx] += rating_change_p1
        current_ratings[p2_idx] += rating_change_p2
        
        # Store opponent rating for clustering feature.
        round_opponents[p1_idx].append(current_ratings[p2_idx])
        round_opponents[p2_idx].append(current_ratings[p1_idx])
    
    # Record this round's opponent-strength sequence (avg rating of opponents so far).
    for player_idx in range(num_players):
        avg_opp_strength = (
            np.mean(round_opponents[player_idx])
            if round_opponents[player_idx]
            else current_ratings[player_idx]
        )
        rounds_data.append({
            "round": round_idx + 1,
            "player": player_names[player_idx],
            "player_current_elo": current_ratings[player_idx],
            "avg_opponent_strength": avg_opp_strength,
        })

df_rounds = pd.DataFrame(rounds_data)

# Cluster players by opponent-strength sequence independently per round.
# If clustering is stable, same player should be in same cluster across rounds.
# If rating volatility endogenously shifts opponent sequences, clustering should drift.

cluster_assignments = {}
for round_num in range(1, num_rounds + 1):
    round_df = df_rounds[df_rounds["round"] == round_num].copy()
    X = round_df[["player_current_elo", "avg_opponent_strength"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    cluster_assignments[round_num] = dict(
        zip(round_df["player"].values, clusters)
    )

# Measure clustering stability: for each player, count how many times their cluster assignment changes.
cluster_changes = {}
for player in player_names:
    assignments = [cluster_assignments[r].get(player, -1) for r in range(1, num_rounds + 1)]
    num_changes = sum(1 for i in range(len(assignments) - 1) if assignments[i] != assignments[i + 1])
    cluster_changes[player] = num_changes

# Players with high cluster volatility indicate endogenous feature drift.
high_drift_players = [
    (p, c) for p, c in cluster_changes.items() if c >= 2
]
high_drift_players.sort(key=lambda x: x[1], reverse=True)

print("=== YOUTH CHESS PAIRING CLUSTERING DRIFT ANALYSIS ===")
print(f"\nPlayers with cluster reassignment across rounds (indicating opponent-sequence drift):")
for player, num_changes in high_drift_players[:10]:
    print(f"  {player}: {num_changes} cluster changes")

print(f"\nMedian cluster changes per player: {np.median(list(cluster_changes.values())):.1f}")
print(f"Max cluster changes per player: {max(cluster_changes.values())}")

# Export results.
results = {
    "num_players": num_players,
    "num_rounds": num_rounds,
    "cluster_changes": cluster_changes,
    "high_drift_players": high_drift_players[:10],
    "interpretation": "High cluster drift indicates that player opponent-strength sequences are non-stationary. "
                      "K-means assumptions break when feature vectors shift endogenously mid-tournament due to rating volatility.",
}

with open("cluster_drift_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to cluster_drift_results.json")
