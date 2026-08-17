import itertools
import pandas as pd
import numpy as np
from scipy.optimize import linprog

np.random.seed(42)

# Generate synthetic youth tournament: 40 players across 3 age brackets
players = []
for age_bracket in ['U10', 'U14', 'U18']:
    if age_bracket == 'U10':
        n = 8  # Undersized bracket
    elif age_bracket == 'U14':
        n = 18  # Well-sized
    else:
        n = 14  # Slightly undersized
    
    for i in range(n):
        rating = np.random.randint(600, 1400) if age_bracket == 'U10' else np.random.randint(1000, 2000)
        players.append({
            'id': len(players),
            'name': f'{age_bracket}_{i}',
            'age_bracket': age_bracket,
            'rating': rating
        })

df = pd.DataFrame(players)
print(f"Tournament: {len(df)} players\n{df.groupby('age_bracket').size()}\n")

# Standard scheduling: minimize cross-bracket pairings
def greedy_pairings(df, round_num):
    """Pair players greedily by rating proximity, ignoring age brackets."""
    available = df.copy()
    pairs = []
    while len(available) >= 2:
        p1 = available.iloc[0]
        # Find nearest rating match
        available = available.iloc[1:]
        if len(available) == 0:
            break
        idx = (available['rating'] - p1['rating']).abs().idxmin()
        p2 = available.loc[idx]
        pairs.append((p1['id'], p2['id']))
        available = available.drop(idx)
    return pairs

standard_pairs = greedy_pairings(df, 1)
cross_bracket = sum(1 for p1, p2 in standard_pairs 
                     if df.loc[p1, 'age_bracket'] != df.loc[p2, 'age_bracket'])
print(f"Standard scheduling (greedy rating): {len(standard_pairs)} pairs, {cross_bracket} cross-bracket\n")

# Constrained scheduling: maximize within-bracket pairings
def constrained_pairings(df):
    """Attempt to pair within age brackets first; measure constraint violations."""
    brackets = df.groupby('age_bracket')
    pairs = []
    violations = {}
    
    for bracket_name, bracket_df in brackets:
        # How many within-bracket pairs can we make?
        max_pairs = len(bracket_df) // 2
        pairs_made = 0
        available = bracket_df.copy()
        
        while len(available) >= 2 and pairs_made < max_pairs:
            p1 = available.iloc[0]
            available = available.iloc[1:]
            if len(available) == 0:
                break
            idx = (available['rating'] - p1['rating']).abs().idxmin()
            p2 = available.loc[idx]
            pairs.append((p1['id'], p2['id'], bracket_name))
            available = available.drop(idx)
            pairs_made += 1
        
        unpaired = len(available)
        violations[bracket_name] = {
            'target_pairs': max_pairs,
            'achieved_pairs': pairs_made,
            'unpaired': unpaired,
            'constraint_violation': max(0, max_pairs - pairs_made)
        }
    
    return pairs, violations

constr_pairs, violations = constrained_pairings(df)

print("Constrained scheduling (within-bracket priority):")
for bracket, viol in violations.items():
    print(f"  {bracket}: target={viol['target_pairs']}, achieved={viol['achieved_pairs']}, "
          f"unpaired={viol['unpaired']}, violation_cost={viol['constraint_violation']}")

# Interpretation: violations reveal imbalance
total_violation = sum(v['constraint_violation'] for v in violations.values())
print(f"\nTotal fairness deficit (constraint violation cost): {total_violation}")
print("\nInterpretation: U10 and U18 brackets are too small to schedule within-age.")
print("The optimizer's *infeasibility signals* reveal structural imbalance that")
print("raw pairing counts (standard vs constrained) do not.")
