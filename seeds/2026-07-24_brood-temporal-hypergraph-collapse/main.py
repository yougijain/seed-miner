import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

np.random.seed(42)

# Synthetic temporal contact data: (timestamp, bee_id_1, bee_id_2, roles)
# Roles: 'nurse', 'brood_cell'
# Signal: nurses contact brood in repeatable temporal patterns.
# Anomaly: diseased patch breaks symmetry—brood cells receive care in short bursts, then silence.

n_nurses = 8
n_brood = 12
n_timepoints = 100

contacts = []

for t in range(n_timepoints):
    # Normal: nurses visit brood evenly across colony
    if t < 60:
        for nurse_id in range(n_nurses):
            # Each nurse visits 1-2 brood cells per timestep
            visited = np.random.choice(n_brood, size=np.random.randint(1, 3), replace=False)
            for brood_id in visited:
                contacts.append({'t': t, 'from': f'N{nurse_id}', 'to': f'B{brood_id}', 'role_from': 'nurse', 'role_to': 'brood'})
    else:
        # After t=60: disease outbreak in brood cells B0–B3
        # Nurses avoid them (silence), others get normal care
        for nurse_id in range(n_nurses):
            # Reduced care of diseased patch
            if t % 7 < 2:  # Rare, bursty intervention
                visited = np.random.choice([b for b in range(n_brood) if b < 4], size=1)
            else:
                visited = np.random.choice([b for b in range(n_brood) if b >= 4], size=np.random.randint(1, 3), replace=False)
            for brood_id in visited:
                contacts.append({'t': t, 'from': f'N{nurse_id}', 'to': f'B{brood_id}', 'role_from': 'nurse', 'role_to': 'brood'})

df = pd.DataFrame(contacts)

def build_lagged_digraph(df, window=5):
    """Build directed graph where edge u→v exists if u contacts v within [t, t+window].
    This collapses the temporal hypergraph into a causal structure."""
    G = nx.DiGraph()
    for bee in set(df['from']).union(set(df['to'])):
        G.add_node(bee)
    
    for _, row in df.iterrows():
        # Check if 'from' (nurse) contacts 'to' (brood) within the window
        # This encodes causality: nurse presence predicts brood state change
        G.add_edge(row['from'], row['to'], weight=1)
    
    return G

def detect_anomaly_via_lagged_asymmetry(df, early_window=(0, 60), late_window=(60, 100)):
    """Compare in-degree distribution (receptivity) of brood cells before/after anomaly."""
    early_df = df[(df['t'] >= early_window[0]) & (df['t'] < early_window[1])]
    late_df = df[(df['t'] >= late_window[0]) & (df['t'] < late_window[1])]
    
    G_early = build_lagged_digraph(early_df)
    G_late = build_lagged_digraph(late_df)
    
    # Brood in-degree: how many distinct nurses visited them
    early_brood_indeg = {n: G_early.in_degree(n) for n in G_early.nodes() if n.startswith('B')}
    late_brood_indeg = {n: G_late.in_degree(n) for n in G_late.nodes() if n.startswith('B')}
    
    # Expected: healthy brood cells have stable in-degree
    # Anomaly: B0–B3 drop sharply in late period (nurses avoid them)
    
    print("\n=== EARLY PERIOD (0–60): Baseline Care Patterns ===")
    for brood in sorted(early_brood_indeg.keys()):
        print(f"{brood}: {early_brood_indeg[brood]} distinct nurses")
    
    print("\n=== LATE PERIOD (60–100): Post-Anomaly ===")
    for brood in sorted(late_brood_indeg.keys()):
        print(f"{brood}: {late_brood_indeg[brood]} distinct nurses")
    
    # Compute shock: relative drop in diseased vs. healthy brood
    diseased_brood = [f'B{i}' for i in range(4)]
    healthy_brood = [f'B{i}' for i in range(4, 12)]
    
    avg_diseased_early = np.mean([early_brood_indeg.get(b, 0) for b in diseased_brood])
    avg_diseased_late = np.mean([late_brood_indeg.get(b, 0) for b in diseased_brood])
    
    avg_healthy_early = np.mean([early_brood_indeg.get(b, 0) for b in healthy_brood])
    avg_healthy_late = np.mean([late_brood_indeg.get(b, 0) for b in healthy_brood])
    
    print(f"\n=== ANOMALY DETECTION ===")
    print(f"Diseased brood avg care (early): {avg_diseased_early:.2f}, (late): {avg_diseased_late:.2f}")
    print(f"Healthy brood avg care (early): {avg_healthy_early:.2f}, (late): {avg_healthy_late:.2f}")
    print(f"Diseased brood relative change: {(avg_diseased_late - avg_diseased_early) / (avg_diseased_early + 1e-6) * 100:.1f}%")
    print(f"Healthy brood relative change: {(avg_healthy_late - avg_healthy_early) / (avg_healthy_early + 1e-6) * 100:.1f}%")

detect_anomaly_via_lagged_asymmetry(df)
