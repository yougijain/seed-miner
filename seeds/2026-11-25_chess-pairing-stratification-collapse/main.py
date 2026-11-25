import itertools
import pandas as pd
import numpy as np
from collections import defaultdict

np.random.seed(42)

# Generate synthetic tournament with KNOWN skill clusters
players = []
for cluster_id, (size, base_elo) in enumerate([(8, 1200), (7, 1600), (5, 2000)]):
    for i in range(size):
        elo = base_elo + np.random.normal(0, 50)
        players.append({'id': len(players), 'name': f'P{len(players)}', 'elo': elo, 'cluster': cluster_id})

df_players = pd.DataFrame(players)
n_players = len(df_players)
n_rounds = n_players - 1  # Full round-robin upper bound

print(f"Tournament: {n_players} players in {len(df_players['cluster'].unique())} ELO clusters")
print(df_players.groupby('cluster')['elo'].agg(['count', 'mean', 'std']))
print()

def try_pairing_with_constraint(elo_tolerance):
    """
    Attempt to build a valid round-robin pairing where every player plays
    exactly once per round, and each opponent is within elo_tolerance of ELO rating.
    Returns: (n_rounds_completed, pairing_dict, fragmentation_score)
    """
    pairings = defaultdict(list)  # round -> list of (p1, p2) tuples
    played = defaultdict(set)  # player -> set of opponents already faced
    round_num = 0
    
    while round_num < n_rounds:
        # Try to construct one valid round
        available = set(range(n_players))
        round_pairings = []
        feasible = True
        
        while len(available) >= 2 and feasible:
            # Greedy pairing: take first available, find compatible opponent
            p1 = min(available)  # Deterministic selection
            candidates = [
                p2 for p2 in available 
                if p2 != p1 
                and abs(df_players.loc[p1, 'elo'] - df_players.loc[p2, 'elo']) <= elo_tolerance
                and p2 not in played[p1]
            ]
            
            if not candidates:
                feasible = False
                break
            
            p2 = candidates[0]  # Take first valid candidate
            round_pairings.append((p1, p2))
            played[p1].add(p2)
            played[p2].add(p1)
            available.discard(p1)
            available.discard(p2)
        
        if not feasible or len(round_pairings) == 0:
            break
        
        pairings[round_num] = round_pairings
        round_num += 1
    
    # Compute fragmentation: how many disconnected clusters in the "compatible opponent graph"?
    # Build graph of who CAN play whom under tolerance
    compatible = defaultdict(set)
    for i in range(n_players):
        for j in range(i+1, n_players):
            if abs(df_players.loc[i, 'elo'] - df_players.loc[j, 'elo']) <= elo_tolerance:
                compatible[i].add(j)
                compatible[j].add(i)
    
    # Count connected components via BFS
    visited = set()
    num_components = 0
    for start in range(n_players):
        if start not in visited:
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                queue.extend([n for n in compatible[node] if n not in visited])
            num_components += 1
    
    return round_num, pairings, num_components

print("Pairing Feasibility Under Increasing ELO Tolerances:")
print("-" * 60)
tolerances = [25, 50, 75, 100, 150, 200, 300]
results = []

for tol in tolerances:
    rounds_completed, pairings, fragments = try_pairing_with_constraint(tol)
    pct_coverage = 100 * rounds_completed / n_rounds
    results.append({
        'elo_tolerance': tol,
        'rounds_completed': rounds_completed,
        'max_possible_rounds': n_rounds,
        'coverage_pct': pct_coverage,
        'graph_fragments': fragments
    })
    print(f"Tolerance ±{tol:3d}: {rounds_completed:2d}/{n_rounds} rounds ({pct_coverage:5.1f}%) | {fragments} graph components")

df_results = pd.DataFrame(results)
print()
print("INTERPRETATION:")
print("- Steep drop in coverage = skill clusters too far apart to pair fairly")
print("- Rising graph_fragments = ELO tolerance creates 'incompatible subpopulations'")
print("- Feasibility collapse point reveals hidden tournament structure")
print()

# Show which players end up in largest fragment at tight tolerance
print(f"\nAt tight tolerance (±50 ELO):")
rounds_completed, pairings, fragments = try_pairing_with_constraint(50)
print(f"Only {rounds_completed} rounds feasible. This tight constraint exposes")
print(f"that the {len(df_players)} players don't form a connected 'compatible' graph.")
print(f"Tournament organizers should investigate player skill distribution.")
