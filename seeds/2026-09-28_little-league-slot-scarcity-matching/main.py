import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

np.random.seed(42)

# Synthetic little league data: 10 teams, 4 time slots across 2 days
NUM_TEAMS = 10
TIME_SLOTS = [
    ("Sat", "09:00"), ("Sat", "11:00"),
    ("Sun", "09:00"), ("Sun", "11:00")
]

# Create schedule: some slots artificially congested (unfair)
schedule = []
team_to_slot = {}

# Slots 0,1 get 4 teams each (scarce); slots 2,3 get 1 team each (oversupply)
slot_caps = [4, 4, 1, 1]  # THIS UNFAIRNESS IS THE SIGNAL TO DETECT
assignments = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3]  # teams 0-3 → slot 0, etc.

for team_id in range(NUM_TEAMS):
    slot_idx = assignments[team_id]
    team_to_slot[team_id] = slot_idx
    schedule.append({"team": f"T{team_id}", "slot_idx": slot_idx, "day": TIME_SLOTS[slot_idx][0], "time": TIME_SLOTS[slot_idx][1]})

schedule_df = pd.DataFrame(schedule)
print("=== OBSERVED SCHEDULE (WITH LATENT UNFAIRNESS) ===")
print(schedule_df.groupby('slot_idx').size())
print(schedule_df)
print()

# REVERSE MATCHING: compute endogenous slot costs based on current occupancy
def compute_slot_desirability(team_to_slot, num_slots):
    """Compute how scarce each slot is (higher = more congested)."""
    slot_loads = defaultdict(int)
    for team, slot in team_to_slot.items():
        slot_loads[slot] += 1
    
    desirability = {}
    for slot in range(num_slots):
        # Slots with MORE teams already assigned are LESS desirable (scarcity cost)
        desirability[slot] = slot_loads.get(slot, 0)
    return desirability

desirability = compute_slot_desirability(team_to_slot, len(TIME_SLOTS))
print("=== ENDOGENOUS SLOT DESIRABILITY (Cost = Load) ===")
for slot_idx, cost in desirability.items():
    print(f"Slot {slot_idx} ({TIME_SLOTS[slot_idx]}): {cost} teams (cost={cost})")
print()

# BIPARTITE MATCHING WITH ENDOGENOUS COST:
# Try to re-assign teams such that slot loads are more uniform.
# The matcher will expose which teams MUST occupy scarce slots (unfairness signal).

def greedy_load_balance_matching(team_to_slot, num_slots, num_games_needed):
    """
    Greedily reassign teams to minimize slot-load variance.
    Returns which teams 'stuck' in overloaded slots (fairness violation signal).
    """
    new_assignment = {}
    slot_loads = defaultdict(int)
    
    # Iterate through teams; try to place each in least-loaded slot
    for team in range(NUM_TEAMS):
        best_slot = min(range(num_slots), key=lambda s: slot_loads[s])
        new_assignment[team] = best_slot
        slot_loads[best_slot] += 1
    
    # Measure how many teams are in overloaded slots vs ideal
    ideal_per_slot = NUM_TEAMS / num_slots
    violations = sum(1 for s, load in slot_loads.items() if load > ideal_per_slot)
    
    return new_assignment, slot_loads, violations

ideal_assignment, ideal_loads, ideal_violations = greedy_load_balance_matching(team_to_slot, len(TIME_SLOTS), 5)
print("=== IDEAL (FAIR) ASSIGNMENT ===")
for slot_idx, load in ideal_loads.items():
    print(f"Slot {slot_idx}: {load} teams")
print(f"Load variance (ideal): {np.var(list(ideal_loads.values())):.2f}")
print()

current_loads = defaultdict(int)
for team, slot in team_to_slot.items():
    current_loads[slot] += 1
print("=== CURRENT (UNFAIR) ASSIGNMENT ===")
for slot_idx in range(len(TIME_SLOTS)):
    print(f"Slot {slot_idx}: {current_loads[slot_idx]} teams")
print(f"Load variance (current): {np.var([current_loads[s] for s in range(len(TIME_SLOTS))]):.2f}")
print()

# DETECT UNFAIRNESS: which teams are "stuck" in overloaded slots?
# These are teams that, if moved, would reduce variance but aren't moved (fairness violation).
stuck_teams = []
for team_id in range(NUM_TEAMS):
    current_slot = team_to_slot[team_id]
    if current_loads[current_slot] > (NUM_TEAMS / len(TIME_SLOTS)):
        stuck_teams.append(team_id)

print(f"=== FAIRNESS DIAGNOSIS ===")
print(f"Teams stuck in overloaded slots (unfairness signal): {stuck_teams}")
print(f"These {len(stuck_teams)} teams face systematically worse availability.")
print()
print(f"Load imbalance (variance): {np.var([current_loads[s] for s in range(len(TIME_SLOTS))]):.2f}")
print(f"Fair load imbalance (variance): {np.var(list(ideal_loads.values())):.2f}")
print(f"Unfairness magnitude: {np.var([current_loads[s] for s in range(len(TIME_SLOTS))]) - np.var(list(ideal_loads.values())):.2f}")