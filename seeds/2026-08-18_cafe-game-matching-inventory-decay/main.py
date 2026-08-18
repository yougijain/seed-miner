import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Generate synthetic cafe checkout log with hidden co-play groups
np.random.seed(42)
N_DAYS = 60
N_GAMES = 12
GAMES = [f'G{i:02d}' for i in range(N_GAMES)]

# Define 3 hidden co-play clusters
CLUSTERS = {
    'cluster_A': ['G00', 'G01', 'G02'],  # Catan/Ticket to Ride/Splendor-like
    'cluster_B': ['G03', 'G04', 'G05'],  # Heavy euros
    'cluster_C': ['G06', 'G07', 'G08'],  # Party games
    'singleton': ['G09', 'G10', 'G11']   # Rarely checked out together
}

checkouts = []
for day in range(N_DAYS):
    # Cluster A: high sync checkout (checked out together ~70% of time)
    if np.random.random() < 0.4:
        for game in CLUSTERS['cluster_A']:
            if np.random.random() < 0.85:
                checkouts.append({'date': day, 'game': game, 'duration_hrs': np.random.randint(2, 6)})
    
    # Cluster B: medium sync
    if np.random.random() < 0.35:
        for game in CLUSTERS['cluster_B']:
            if np.random.random() < 0.70:
                checkouts.append({'date': day, 'game': game, 'duration_hrs': np.random.randint(3, 8)})
    
    # Cluster C: medium sync
    if np.random.random() < 0.45:
        for game in CLUSTERS['cluster_C']:
            if np.random.random() < 0.75:
                checkouts.append({'date': day, 'game': game, 'duration_hrs': np.random.randint(1, 4)})
    
    # Singletons: independent
    for game in CLUSTERS['singleton']:
        if np.random.random() < 0.15:
            checkouts.append({'date': day, 'game': game, 'duration_hrs': np.random.randint(1, 5)})

df = pd.DataFrame(checkouts)
print(f"Generated {len(df)} checkout records across {N_DAYS} days.\n")

# Build absence windows: for each game, find intervals when it's NOT checked out
def build_absence_windows(df, game, window_days=7):
    """Return set of window_days-wide time windows where game is absent."""
    checked_out_days = set(df[df['game'] == game]['date'].values)
    all_days = set(range(N_DAYS))
    absent_days = sorted(all_days - checked_out_days)
    
    # Collapse consecutive absence into windows
    windows = []
    if absent_days:
        win_start = absent_days[0]
        for i, day in enumerate(absent_days[1:], 1):
            if day - absent_days[i-1] > 1:
                windows.append((win_start, absent_days[i-1]))
                win_start = day
        windows.append((win_start, absent_days[-1]))
    return windows

# Compute co-absence overlap matrix
game_list = sorted(df['game'].unique())
co_absence_matrix = pd.DataFrame(0.0, index=game_list, columns=game_list)

for g1 in game_list:
    for g2 in game_list:
        if g1 < g2:
            wins1 = build_absence_windows(df, g1)
            wins2 = build_absence_windows(df, g2)
            
            # Count overlapping absence days
            abs1_days = set()
            for start, end in wins1:
                abs1_days.update(range(start, end + 1))
            
            abs2_days = set()
            for start, end in wins2:
                abs2_days.update(range(start, end + 1))
            
            overlap = len(abs1_days & abs2_days)
            total = len(abs1_days | abs2_days)
            jaccard = overlap / total if total > 0 else 0
            
            co_absence_matrix.loc[g1, g2] = jaccard
            co_absence_matrix.loc[g2, g1] = jaccard

print("Co-absence Jaccard matrix (top-left corner):")
print(co_absence_matrix.iloc[:5, :5])
print()

# Greedy bipartite matching: match games to shelves such that high-affinity pairs share shelves
# Use a simple greedy approach: build groups by matching high co-absence games
matched_groups = []
matched_games = set()

for game in game_list:
    if game in matched_games:
        continue
    
    # Start new shelf group with this game
    shelf = [game]
    matched_games.add(game)
    
    # Greedily add other unmatched games with highest co-absence to this shelf
    # (limit shelf to 3 games for practical cafes)
    while len(shelf) < 3:
        # Find unmatched game with highest avg co-absence to current shelf
        best_game = None
        best_score = -1
        
        for candidate in game_list:
            if candidate not in matched_games:
                avg_affinity = np.mean([co_absence_matrix.loc[candidate, s] for s in shelf])
                if avg_affinity > best_score:
                    best_score = avg_affinity
                    best_game = candidate
        
        if best_game and best_score > 0.3:  # threshold to avoid forced pairings
            shelf.append(best_game)
            matched_games.add(best_game)
        else:
            break
    
    matched_groups.append(shelf)

print(f"Recommended shelf groups (via co-absence bipartite matching):\n")
for i, group in enumerate(matched_groups, 1):
    print(f"  Shelf {i}: {', '.join(group)}")

print("\nGround truth clusters:")
for cname, games in CLUSTERS.items():
    print(f"  {cname}: {', '.join(games)}")

# Evaluate: how many ground-truth cluster members ended up on same shelf?
accuracy = 0
for cluster_games in CLUSTERS.values():
    for shelf in matched_groups:
        overlap = len(set(cluster_games) & set(shelf))
        if overlap >= 2:
            accuracy += overlap

print(f"\nRecovery metric: {accuracy} cluster-member pairs co-shelved (max possible: ~18)")
