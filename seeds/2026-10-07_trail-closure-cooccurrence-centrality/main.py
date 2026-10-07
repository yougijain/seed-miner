import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

# Synthetic data: 12 trail segments, some structurally critical (bottlenecks)
# True structure: segments 2,3,5 are bridges; 1,4 depend on 2; 6,7,8 depend on 3; etc.
np.random.seed(42)

# Generate synthetic closure events
# Key insight: when a segment closes, downstream segments often close together (or go unreported)
closure_patterns = [
    [2],           # bridge 2 closes alone
    [2, 1, 4],     # bridge 2 closes → segments 1,4 unreachable
    [3, 6, 7],     # bridge 3 closes → segments 6,7 unreachable
    [5, 8],        # bridge 5 closes → segment 8 unreachable
    [2, 3],        # both bridges close (rare event)
    [1],           # false positive: segment 1 reported closed (noise)
    [6],           # noise closure report
    [2, 1, 4, 9],  # bridge 2 + noise
    [3, 6, 7, 8],  # bridge 3 + unexpected 8
    [5],           # bridge 5 alone
    [2, 3, 1, 4, 6, 7],  # massive cascade
    [4, 1],        # co-closure without bridge (weak signal)
]

# Repeat to simulate 50 closure events over a season
closure_events = []
for _ in range(50):
    pattern = closure_patterns[np.random.randint(0, len(closure_patterns))]
    # Add Poisson-distributed noise: randomly include non-members
    noise_segs = np.random.choice(range(1, 13), size=np.random.poisson(0.5), replace=False)
    event = list(set(pattern + list(noise_segs)))
    closure_events.append(sorted(event))

# Build co-occurrence graph: edge weight = how many times two segments closed together
cooccurrence_graph = nx.Graph()
for seg in range(1, 13):
    cooccurrence_graph.add_node(seg)

for event in closure_events:
    for seg_a, seg_b in combinations(event, 2):
        if cooccurrence_graph.has_edge(seg_a, seg_b):
            cooccurrence_graph[seg_a][seg_b]['weight'] += 1
        else:
            cooccurrence_graph.add_edge(seg_a, seg_b, weight=1)

# Threshold: only keep edges with co-occurrence weight >= 2 (filter noise)
threshold = 2
filtered_cooccurrence = nx.Graph()
for node in cooccurrence_graph.nodes():
    filtered_cooccurrence.add_node(node)
for edge in cooccurrence_graph.edges(data=True):
    if edge[2]['weight'] >= threshold:
        filtered_cooccurrence.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

# Compute centrality on the inferred network
betweenness = nx.betweenness_centrality(filtered_cooccurrence, weight='weight')
closeness = nx.closeness_centrality(filtered_cooccurrence, distance='weight')
degree_centrality = nx.degree_centrality(filtered_cooccurrence)

# Rank segments by centrality
ranked_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

print("=" * 60)
print("TRAIL CLOSURE CO-OCCURRENCE NETWORK ANALYSIS")
print("=" * 60)
print(f"\nInput: {len(closure_events)} closure events (sparse, noisy observations)")
print(f"Inferred network edges after thresholding (weight >= {threshold}): {filtered_cooccurrence.number_of_edges()}")
print(f"Isolated nodes (never co-occurred): {list(nx.isolates(filtered_cooccurrence))}")
print("\nTop 5 segments by BETWEENNESS CENTRALITY (inferred bottlenecks):")
for seg, cent in ranked_betweenness[:5]:
    print(f"  Segment {seg}: {cent:.3f}")

print("\nTop 5 segments by DEGREE CENTRALITY (most co-occurred):")
ranked_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
for seg, cent in ranked_degree[:5]:
    print(f"  Segment {seg}: {cent:.3f} ({int(filtered_cooccurrence.degree(seg))} neighbors)")

print("\nInterpretation:")
print("  Segments 2, 3, 5 should rank highest if the graph reconstruction")
print("  correctly inferred them as bridges. High betweenness = likely bottleneck.")
print("  Isolated segments suggest they never co-occurred with others (true singletons or always closed alone).")
print("\nLimitation:")
print("  Without ground-truth trail topology, we cannot validate whether")
print("  the inferred centrality truly reflects maintenance dependencies.")
print("  Sparse, noisy closure logs may obscure or confound the true structure.")
