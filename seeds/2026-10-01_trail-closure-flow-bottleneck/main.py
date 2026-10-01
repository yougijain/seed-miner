import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations

np.random.seed(42)

# Synthetic trail network: 12 segments, trailhead to summit
# Real data would come from trail GPS traces and closure logs.
G = nx.DiGraph()

# Build a realistic branching trail network
edges = [
    ('trailhead', 'seg_A', {'capacity': 100, 'segment': 'A'}),
    ('trailhead', 'seg_B', {'capacity': 80, 'segment': 'B'}),
    ('seg_A', 'seg_C', {'capacity': 100, 'segment': 'C'}),
    ('seg_A', 'seg_D', {'capacity': 60, 'segment': 'D'}),
    ('seg_B', 'seg_E', {'capacity': 80, 'segment': 'E'}),
    ('seg_B', 'seg_F', {'capacity': 50, 'segment': 'F'}),
    ('seg_C', 'seg_G', {'capacity': 100, 'segment': 'G'}),
    ('seg_D', 'seg_H', {'capacity': 60, 'segment': 'H'}),
    ('seg_E', 'seg_I', {'capacity': 80, 'segment': 'I'}),
    ('seg_F', 'seg_J', {'capacity': 50, 'segment': 'J'}),
    ('seg_G', 'summit', {'capacity': 100, 'segment': 'G_out'}),
    ('seg_H', 'summit', {'capacity': 60, 'segment': 'H_out'}),
    ('seg_I', 'summit', {'capacity': 80, 'segment': 'I_out'}),
    ('seg_J', 'summit', {'capacity': 50, 'segment': 'J_out'}),
]

G.add_edges_from(edges)

# Synthetic closure log: pairs of segments observed closed together
# (Real data: maintenance schedules, weather closures, etc.)
closure_records = [
    (['seg_C', 'seg_G'], 15),  # High co-closure frequency
    (['seg_A', 'seg_C'], 12),
    (['seg_D', 'seg_H'], 10),
    (['seg_B', 'seg_E'], 8),
    (['seg_E', 'seg_I'], 9),
    (['seg_F', 'seg_J'], 7),
    (['seg_C', 'seg_D'], 5),   # Low co-closure (independent)
    (['seg_G', 'seg_I'], 3),
]

# Question: Which segments, when closed, cause the LARGEST *increase* in
# network bottleneck (decrease in max flow)? That segment is structurally critical.

def compute_max_flow_reduction(G, closure_set):
    """Remove closure_set edges, compute flow reduction vs baseline."""
    G_copy = G.copy()
    for seg in closure_set:
        # Remove all edges with this segment label
        edges_to_remove = [(u, v) for u, v, d in G_copy.edges(data=True)
                           if d.get('segment') == seg]
        G_copy.remove_edges_from(edges_to_remove)
    
    try:
        flow_value = nx.maximum_flow_value(G_copy, 'trailhead', 'summit')
    except nx.NetworkXError:
        flow_value = 0  # Network disconnected
    return flow_value

# Baseline max flow (no closures)
baseline_flow = nx.maximum_flow_value(G, 'trailhead', 'summit')
print(f"Baseline max flow (no closures): {baseline_flow}")
print()

# Evaluate impact of each segment closure
print("Single-segment closure impact (flow reduction):")
impacts = []
segments = [d['segment'] for u, v, d in G.edges(data=True)]
for seg in set(segments):
    reduced_flow = compute_max_flow_reduction(G, [seg])
    impact = baseline_flow - reduced_flow
    impacts.append({'segment': seg, 'flow_reduction': impact, 'residual_flow': reduced_flow})
    print(f"  {seg}: -{impact} (residual flow: {reduced_flow})")

print()

# Now: Find pairs of segments that, when *jointly* closed,
# cause LARGER reduction than sum of individual reductions.
# This signals hidden dependency (one segment "backs up" flow that another handles).

print("Synergistic closure pairs (joint impact > sum of individuals):")
synergies = []
for seg1, seg2 in combinations(set(segments), 2):
    impact1 = next((x['flow_reduction'] for x in impacts if x['segment'] == seg1), 0)
    impact2 = next((x['flow_reduction'] for x in impacts if x['segment'] == seg2), 0)
    joint_flow = compute_max_flow_reduction(G, [seg1, seg2])
    joint_impact = baseline_flow - joint_flow
    synergy = joint_impact - (impact1 + impact2)
    
    if synergy > 0.01:  # Threshold to filter noise
        synergies.append({
            'pair': (seg1, seg2),
            'synergy': synergy,
            'joint_impact': joint_impact,
            'sum_individual': impact1 + impact2
        })

synergies.sort(key=lambda x: x['synergy'], reverse=True)
for s in synergies[:5]:
    print(f"  {s['pair']}: synergy={s['synergy']:.1f} "
          f"(joint={s['joint_impact']:.1f}, sum_individual={s['sum_individual']:.1f})")

print()

# Build a "bottleneck graph": edges weighted by flow reduction impact.
# This reveals the hidden dependency structure.
G_bottleneck = nx.Graph()
for seg1, seg2 in combinations(set(segments), 2):
    synergy = next((x['synergy'] for x in synergies if x['pair'] == (seg1, seg2)), 0)
    if synergy > 0:
        G_bottleneck.add_edge(seg1, seg2, weight=synergy)

print(f"Bottleneck dependency graph: {len(G_bottleneck.nodes())} segments, "
      f"{len(G_bottleneck.edges())} synergistic pairs")
if G_bottleneck.edges():
    print(f"  Highest-weight synergy pair: {max(G_bottleneck.edges(data=True), key=lambda x: x[2]['weight'])}")
