import numpy as np
import pandas as pd
from itertools import combinations

np.random.seed(42)

# Synthetic little league data: 12 teams, 18 game slots
teams = [f"Team_{i}" for i in range(1, 13)]
slots = [(day, time) for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] 
         for time in ['09:00', '18:30']]
slots = slots[:18]  # Trim to 18 slots

# Slot base desirability: weekend morning >> weekday evening
base_pref = {
    ('Sat', '09:00'): 10, ('Sun', '09:00'): 10,
    ('Sat', '18:30'): 8, ('Sun', '18:30'): 8,
    ('Fri', '18:30'): 5, ('Thu', '18:30'): 5,
    ('Wed', '18:30'): 4, ('Tue', '18:30'): 4, ('Mon', '18:30'): 4,
}
# Fill missing slots
for s in slots:
    if s not in base_pref:
        base_pref[s] = np.random.randint(1, 6)

# Simulate team-slot affinity: most teams prefer high-desirability slots
affinity = {}
for team in teams:
    affinity[team] = {}
    for slot in slots:
        # Base preference + noise, but better slots are still better
        affinity[team][slot] = base_pref[slot] + np.random.randn() * 0.5

def compute_slot_costs(matching_state, affinity):
    """
    Endogenous cost: slot cost = -(affinity) + saturation_penalty.
    As more teams get assigned to desirable slots, those slots become less attractive for remaining teams.
    """
    slot_counts = {}
    for team, slot in matching_state:
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
    
    costs = {}
    for team in affinity:
        costs[team] = {}
        for slot in affinity[team]:
            saturation = slot_counts.get(slot, 0) / len(affinity)  # Fraction of teams already using this slot
            # Cost: negative affinity (we want to maximize) + penalty for saturation
            costs[team][slot] = -affinity[team][slot] - 2 * saturation
    return costs

def greedy_matching_with_endogeneity(affinity, max_iterations=100):
    """
    Greedy bipartite matching that iteratively updates slot costs based on current assignments.
    This reveals structural inequity: some teams can never escape low-cost slots because 
    good slots fill up, then subsequent teams face higher cost for those same slots.
    """
    matching = []
    unmatched_teams = set(affinity.keys())
    unmatched_slots = set(affinity[list(affinity.keys())[0]].keys())
    iteration_log = []
    
    for iteration in range(max_iterations):
        if not unmatched_teams or not unmatched_slots:
            break
        
        # Recompute costs given current state
        costs = compute_slot_costs(matching, affinity)
        
        # Greedy: pick (team, slot) pair with minimum cost
        best_cost = float('inf')
        best_pair = None
        for team in unmatched_teams:
            for slot in unmatched_slots:
                if costs[team][slot] < best_cost:
                    best_cost = costs[team][slot]
                    best_pair = (team, slot)
        
        if best_pair is None:
            break
        
        team, slot = best_pair
        matching.append((team, slot))
        unmatched_teams.remove(team)
        unmatched_slots.remove(slot)
        
        # Log: which team got which slot and the cost endogeneity effect
        slot_count_at_match = sum(1 for t, s in matching if s == slot) - 1
        endogenous_penalty = 2 * slot_count_at_match / len(affinity)
        iteration_log.append({
            'iteration': iteration,
            'team': team,
            'slot': slot,
            'base_affinity': affinity[team][slot],
            'endogenous_penalty': endogenous_penalty,
            'final_cost': best_cost
        })
    
    return matching, pd.DataFrame(iteration_log), unmatched_teams

# Run matching
matching, log_df, unmatched = greedy_matching_with_endogeneity(affinity)

print("=== ENDOGENOUS BIPARTITE MATCHING: LITTLE LEAGUE SCHEDULING ===")
print(f"\nMatched pairs: {len(matching)} / {len(teams)} teams")
print(f"Unmatched teams: {unmatched}")

# Identify inequity: teams that got high-penalty slots
penalty_by_team = log_df.groupby('team')['endogenous_penalty'].sum().sort_values(ascending=False)
print(f"\nTeams with highest cumulative endogenous penalties (structural disadvantage):")
print(penalty_by_team.head(5))

# Visualize: which slots became saturated (evidence of slot scarcity creating inequity)
slot_usage = pd.DataFrame(matching, columns=['Team', 'Slot']).groupby('Slot').size().sort_values(ascending=False)
print(f"\nSlot usage (saturation):")
print(slot_usage.head(10))

# Key insight: compare teams matched early (got choice) vs. late (forced into bad slots)
early_matched = set(log_df[log_df['iteration'] < 4]['team'])
late_matched = set(log_df[log_df['iteration'] >= 8]['team'])

if early_matched and late_matched:
    early_affinity_avg = np.mean([log_df[log_df['team'] == t]['base_affinity'].values[0] for t in early_matched if t in log_df['team'].values])
    late_affinity_avg = np.mean([log_df[log_df['team'] == t]['base_affinity'].values[0] for t in late_matched if t in log_df['team'].values])
    print(f"\nEarly-matched teams avg affinity: {early_affinity_avg:.2f}")
    print(f"Late-matched teams avg affinity: {late_affinity_avg:.2f}")
    print(f"Inequity signal: {early_affinity_avg - late_affinity_avg:.2f} (positive = disadvantage for late teams)")
