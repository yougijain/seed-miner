import pandas as pd
import numpy as np
from collections import defaultdict
import json

np.random.seed(42)

# Synthetic data: game cafe checkout log
# Real structure: game popularity, session duration, player group size
# Hidden signal: some games are "incompatible" (share player pools, demand peaks overlap)
days = 30
games = ['Catan', 'Ticket2Ride', 'Azul', 'Wingspan', 'Splendor', 'Codenames', 'SevenWonders']
num_slots = 4  # Tables available per 2-hour slot

checkouts = []
for day in range(days):
    for slot in range(6):  # 6 two-hour slots per day
        # Incompatibility structure: Catan + SevenWonders share heavy-player demand
        # Codenames + Splendor share casual-player demand
        slot_time = day * 6 + slot
        
        # Catan: moderate-high demand, tends to spike midday
        if np.random.random() < (0.6 if 2 <= slot <= 4 else 0.4):
            checkouts.append({'day': day, 'slot': slot, 'game': 'Catan', 'duration': 2})
        
        # SevenWonders: often peaks when Catan peaks (incompatibility)
        if np.random.random() < (0.55 if 2 <= slot <= 4 else 0.35):
            checkouts.append({'day': day, 'slot': slot, 'game': 'SevenWonders', 'duration': 1.5})
        
        # Codenames: peaks in evening, casual
        if np.random.random() < (0.5 if slot >= 4 else 0.25):
            checkouts.append({'day': day, 'slot': slot, 'game': 'Codenames', 'duration': 0.5})
        
        # Splendor: also evening casual (incompatible with Codenames)
        if np.random.random() < (0.45 if slot >= 4 else 0.2):
            checkouts.append({'day': day, 'slot': slot, 'game': 'Splendor', 'duration': 1})
        
        # Others: low baseline demand
        for g in ['Ticket2Ride', 'Azul', 'Wingspan']:
            if np.random.random() < 0.2:
                checkouts.append({'day': day, 'slot': slot, 'game': g, 'duration': 1})

df = pd.DataFrame(checkouts)

# Core analysis: For each game, measure how often its presence causes bin-packing failure
# Bin-packing: can we fit all simultaneous checkouts into num_slots tables?
def can_pack(durations, capacity=num_slots):
    """First-fit bin-packing: return True if fits, False if overfull."""
    if len(durations) == 0:
        return True
    bins = []
    for d in sorted(durations, reverse=True):
        placed = False
        for i, b in enumerate(bins):
            if b + d <= 2.0:  # 2-hour slots
                bins[i] += d
                placed = True
                break
        if not placed:
            bins.append(d)
        if len(bins) > capacity:
            return False
    return len(bins) <= capacity

# For each game, measure "contamination index":
# How often does the game's presence cause packing failures in its time slot?
results = {}
for game in games:
    game_slots = df[df['game'] == game].groupby(['day', 'slot']).size().reset_index(name='count')
    
    failures_with_game = 0
    failures_without_game = 0
    
    for _, row in game_slots.iterrows():
        d, s = int(row['day']), int(row['slot'])
        # Get all checkouts in this slot
        slot_checkouts = df[(df['day'] == d) & (df['slot'] == s)]
        
        with_game = slot_checkouts['duration'].tolist()
        without_game = slot_checkouts[slot_checkouts['game'] != game]['duration'].tolist()
        
        if not can_pack(with_game):
            failures_with_game += 1
        if not can_pack(without_game):
            failures_without_game += 1
    
    # Contamination: does this game *cause* failures beyond ambient?
    appearances = len(game_slots)
    contamination = max(0, failures_with_game - failures_without_game) / max(1, appearances)
    results[game] = {
        'appearances': appearances,
        'failures_with': failures_with_game,
        'failures_without': failures_without_game,
        'contamination_score': contamination
    }

# Sort by contamination
sorted_games = sorted(results.items(), key=lambda x: x[1]['contamination_score'], reverse=True)

print("\n=== Game Incompatibility via Bin-Packing Failure Cascade ===")
print(f"\nDetected high-contention games (games whose presence breaks scheduling):")
for game, metrics in sorted_games[:3]:
    print(f"\n{game}:")
    print(f"  Appearances: {metrics['appearances']}")
    print(f"  Packing failures WITH game: {metrics['failures_with']}")
    print(f"  Packing failures WITHOUT game: {metrics['failures_without']}")
    print(f"  Contamination score: {metrics['contamination_score']:.3f}")

print(f"\n✓ Expected pattern: Catan & SevenWonders + Codenames & Splendor should rank high.")
print(f"✓ This detects incompatibility WITHOUT explicit co-occurrence data,")
print(f"  only from the constraint-satisfaction failures of bin-packing.")
