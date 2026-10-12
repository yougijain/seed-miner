import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import json

np.random.seed(42)

# Generate synthetic inspection log: sparse, episodic notes on colonies over 90 days
colonies = [f"C{i}" for i in range(1, 8)]
start_date = datetime(2024, 1, 1)

notes = []
for day in range(90):
    date = start_date + timedelta(days=day)
    # Each day, only 2-3 colonies get inspected (sparse)
    inspected = np.random.choice(colonies, size=np.random.randint(2, 4), replace=False)
    for col in inspected:
        notes.append({
            "date": date,
            "colony": col,
            "day_offset": day,
            "inspected": True
        })

df_notes = pd.DataFrame(notes)

# Build absence matrix: for each pair of colonies, count how many time windows
# they were *both absent* from inspection notes
def build_coabsence_graph(df, window_days=7):
    """Build graph where edge weight = days both colonies were jointly absent."""
    G = nx.Graph()
    for c in colonies:
        G.add_node(c)
    
    # Slide a window across time; for each window, find colonies NOT inspected
    max_day = df["day_offset"].max()
    inspected_by_col_by_day = set(df.groupby("day_offset")["colony"].apply(tuple))
    
    coabsence_count = {(c1, c2): 0 for c1 in colonies for c2 in colonies if c1 < c2}
    
    for start_window in range(0, max_day, window_days):
        end_window = min(start_window + window_days, max_day)
        window_df = df[(df["day_offset"] >= start_window) & (df["day_offset"] < end_window)]
        inspected_in_window = set(window_df["colony"].unique())
        absent_in_window = set(colonies) - inspected_in_window
        
        # For each pair of colonies both absent, increment edge weight
        for c1 in sorted(absent_in_window):
            for c2 in sorted(absent_in_window):
                if c1 < c2:
                    coabsence_count[(c1, c2)] += 1
    
    # Add edges only if coabsence > threshold (signal from noise)
    threshold = 2
    for (c1, c2), count in coabsence_count.items():
        if count > threshold:
            G.add_edge(c1, c2, weight=count)
    
    return G, coabsence_count

G, coabsence = build_coabsence_graph(df_notes, window_days=7)

# Compute centrality: which colonies, if infected, would spread most via contact network?
betweenness = nx.betweenness_centrality(G, weight="weight")
degree = nx.degree_centrality(G)

print("=== Bee Colony Contact Network (inferred from inspection co-absence) ===")
print(f"Nodes: {len(G.nodes())}")
print(f"Edges (co-absence > 2 windows): {len(G.edges())}")
print(f"\nBetweenness Centrality (colonies that bridge clusters):")
for col, bc in sorted(betweenness.items(), key=lambda x: -x[1]):
    print(f"  {col}: {bc:.3f}")
print(f"\nDegree Centrality (most co-absent with others):")
for col, dc in sorted(degree.items(), key=lambda x: -x[1]):
    print(f"  {col}: {dc:.3f}")

print(f"\nEdges (colony pairs with sustained co-absence):")
for c1, c2, w in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"]):
    print(f"  {c1}–{c2}: co-absent {w['weight']} windows")

print(f"\nNetwork density: {nx.density(G):.3f}")
if nx.is_connected(G):
    print("Network is connected.")
else:
    print(f"Network has {nx.number_connected_components(G)} components.")

# Validation: did synthetic data have hidden structure?
print(f"\n=== Interpretation ===")
print(f"Colonies with high betweenness (e.g., {max(betweenness, key=betweenness.get)}) are potential")
print(f"  critical points: if disease enters via them, it spreads to multiple clusters.")
print(f"Edges represent functional coupling inferred purely from WHEN colonies were jointly absent.")
