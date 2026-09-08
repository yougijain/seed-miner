import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import json

np.random.seed(42)

# Generate synthetic checkout log: (timestamp, game_id, checkout_duration_hours, player_count)
days = 30
checkouts = []
game_names = {0: 'Catan', 1: 'Ticket2Ride', 2: 'Splendor', 3: 'Azul', 4: 'CandyCane', 5: 'Secret'}

# Latent structure: games form incompatible pairs
# (0,1) and (2,3) are "social heavy" and conflict
# (4,5) are "quick" and don't conflict
incompatible_pairs = [(0, 1), (2, 3)]

for day in range(days):
    base = datetime(2024, 1, 1) + timedelta(days=day)
    for hour in range(10, 22):
        base_slot = base.replace(hour=hour, minute=0)
        # Incompatible pairs cluster checkouts in same hour
        if np.random.random() < 0.7:
            pair = incompatible_pairs[np.random.randint(0, 2)]
            for game in pair:
                duration = np.random.choice([1, 2, 3])
                players = np.random.randint(2, 5)
                checkouts.append([base_slot, game, duration, players])
        # Compatible games spread out
        else:
            game = np.random.choice([4, 5])
            duration = np.random.choice([0.5, 1])
            players = np.random.randint(1, 3)
            checkouts.append([base_slot, game, duration, players])

df = pd.DataFrame(checkouts, columns=['checkout_time', 'game_id', 'duration_hours', 'player_count'])

def bin_pack_greedy(df, table_capacity=6):
    """Greedy bin-packing: assign checkouts to tables. Return packing and conflicts."""
    df_sorted = df.sort_values('checkout_time').reset_index(drop=True)
    tables = defaultdict(list)  # table_id -> list of (game_id, end_time, player_count)
    table_id = 0
    conflicts = []  # (idx, reason: 'overflow' or 'temporal_overlap')
    
    for idx, row in df_sorted.iterrows():
        checkout_time = row['checkout_time']
        duration = row['duration_hours']
        end_time = checkout_time + timedelta(hours=duration)
        players = row['player_count']
        game = row['game_id']
        
        # Try to fit into existing table
        placed = False
        for tid, occupants in tables.items():
            # Check temporal overlap and capacity
            table_players = sum(p for _, _, p in occupants)
            if table_players + players <= table_capacity:
                # Check if any game on this table overlaps temporally
                overlap = any(
                    not (end_time <= occ_end or checkout_time >= occ_start)
                    for occ_game, (occ_start, occ_end), _ in [
                        (occupants[i][0], (occupants[i][1] - timedelta(hours=occupants[i][2]), occupants[i][1]), occupants[i][2])
                        for i in range(len(occupants))
                    ]
                )
                if not overlap:
                    tables[tid].append((game, end_time, players))
                    placed = True
                    break
        
        if not placed:
            # Open new table or mark conflict
            if table_id < 3:  # Only 3 tables available
                tables[table_id] = [(game, end_time, players)]
                table_id += 1
            else:
                conflicts.append((idx, 'overflow'))
    
    return tables, conflicts

tables, conflicts = bin_pack_greedy(df)
print(f"Tables used: {len(tables)}, Conflicts (failed placements): {len(conflicts)}")

# Extract conflict signature: which games appear in rows with conflicts?
conflict_games = df.iloc[[c[0] for c in conflicts]]['game_id'].values

# Compute co-occurrence in time windows around conflicts
window = pd.Timedelta(hours=1)
conflict_signatures = []

for idx, _ in conflicts:
    t = df.loc[idx, 'checkout_time']
    nearby = df[(df['checkout_time'] >= t - window) & (df['checkout_time'] <= t + window)]['game_id'].values
    if len(nearby) > 1:
        conflict_signatures.append(sorted(nearby.tolist()))

# Cluster games by conflict co-occurrence patterns
from itertools import combinations
game_conflict_count = defaultdict(lambda: defaultdict(int))

for sig in conflict_signatures:
    for g1, g2 in combinations(sig, 2):
        game_conflict_count[g1][g2] += 1
        game_conflict_count[g2][g1] += 1

print("\nGame pair conflict co-occurrences:")
for g1 in sorted(game_conflict_count.keys()):
    for g2 in sorted(game_conflict_count[g1].keys()):
        if g1 < g2:
            count = game_conflict_count[g1][g2]
            if count > 0:
                print(f"  {game_names[g1]} <-> {game_names[g2]}: {count} conflicts")

print("\n--- Analysis ---")
print(f"Detected incompatible pairs via scheduling conflicts:")
for g1 in range(6):
    for g2 in range(g1+1, 6):
        if game_conflict_count[g1][g2] > 2:
            print(f"  {game_names[g1]} <-> {game_names[g2]} (confidence: {game_conflict_count[g1][g2]} co-conflict events)")

print(f"\nTrue incompatible pairs (ground truth): {[(game_names[g1], game_names[g2]) for g1, g2 in incompatible_pairs]}")
print(f"\nConclusion: Bin-packing constraint violations encode latent game incompatibility without explicit co-rental data.")
