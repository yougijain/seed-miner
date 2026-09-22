import pandas as pd
import networkx as nx
from collections import defaultdict
import json

# Synthetic trail closure log: dates when segments were closed
# Key insight: if segment A closes, segments B and C close 2-3 days later
# (e.g., A is a junction; losing A makes B and C inaccessible for maintenance)
closures_data = [
    {'date': '2024-01-01', 'segment': 'A', 'reason': 'rockslide'},
    {'date': '2024-01-01', 'segment': 'B', 'reason': 'access_blocked'},
    {'date': '2024-01-02', 'segment': 'C', 'reason': 'access_blocked'},
    {'date': '2024-01-05', 'segment': 'D', 'reason': 'weather'},
    {'date': '2024-01-06', 'segment': 'E', 'reason': 'access_blocked'},
    {'date': '2024-01-10', 'segment': 'A', 'reason': 'reopened'},
    {'date': '2024-01-11', 'segment': 'B', 'reason': 'reopened'},
    {'date': '2024-01-12', 'segment': 'C', 'reason': 'reopened'},
    {'date': '2024-01-15', 'segment': 'D', 'reason': 'reopened'},
    {'date': '2024-01-16', 'segment': 'E', 'reason': 'reopened'},
    {'date': '2024-02-01', 'segment': 'F', 'reason': 'rockslide'},
    {'date': '2024-02-02', 'segment': 'G', 'reason': 'access_blocked'},
    {'date': '2024-02-03', 'segment': 'H', 'reason': 'access_blocked'},
]

df = pd.DataFrame(closures_data)
df['date'] = pd.to_datetime(df['date'])

def build_cascade_graph(df, lag_days=2):
    """
    Build a directed graph where an edge A->B exists if:
    - A closes (reason != 'reopened')
    - B closes within lag_days of A closing
    - B reopens after A reopens (suggesting B was dependent on A access)
    """
    closures = df[df['reason'] != 'reopened'].copy()
    closures = closures.sort_values('date')
    
    G = nx.DiGraph()
    cascade_edges = []
    
    for seg_a in closures['segment'].unique():
        a_close = closures[closures['segment'] == seg_a]['date'].iloc[0]
        
        # Find segments that close shortly after A
        for seg_b in closures['segment'].unique():
            if seg_a == seg_b:
                continue
            b_close = closures[closures['segment'] == seg_b]['date'].iloc[0]
            days_diff = (b_close - a_close).days
            
            # If B closes 1-lag_days after A, suggest dependency
            if 1 <= days_diff <= lag_days:
                G.add_edge(seg_a, seg_b, lag=days_diff)
                cascade_edges.append((seg_a, seg_b, days_diff))
    
    return G, cascade_edges

def detect_bottlenecks(G):
    """
    A bottleneck segment is one whose removal disconnects the most other segments.
    In cascade terms: removing its closure would prevent cascades to others.
    """
    bottlenecks = {}
    for node in G.nodes():
        # Simulate removing this node: how many nodes become unreachable?
        G_temp = G.copy()
        G_temp.remove_node(node)
        reachable_without = set()
        for start in G_temp.nodes():
            reachable_without.update(nx.descendants(G_temp, start))
        
        # Compare to original
        reachable_with = set()
        for start in G.nodes():
            if start != node:
                reachable_with.update(nx.descendants(G, start))
        
        impact = len(reachable_with) - len(reachable_without)
        if impact > 0:
            bottlenecks[node] = impact
    
    return sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)

# Build graph and analyze
G, edges = build_cascade_graph(df, lag_days=3)
bottlenecks = detect_bottlenecks(G)

print("=== Trail Closure Cascade Graph ===")
print(f"Nodes (segments): {list(G.nodes())}")
print(f"\nEdges (cascade relationships):")
for src, dst, lag in edges:
    print(f"  {src} -> {dst} (lag={lag} days)")

print(f"\nBottleneck segments (key maintenance dependencies):")
for seg, impact in bottlenecks:
    print(f"  {seg}: affects {impact} other segments when closed")

print(f"\nGraph density: {nx.density(G):.3f}")
print(f"Number of connected components: {nx.number_weakly_connected_components(G)}")

if G.number_of_nodes() > 0:
    # Find strongly connected components (mutual dependencies)
    sccs = list(nx.strongly_connected_components(G))
    print(f"\nStrongly connected components (mutual closure dependencies):")
    for scc in sccs:
        if len(scc) > 1:
            print(f"  {scc}")

# Output result
result = {
    'question': 'Which trail segments have topological maintenance dependencies detectable from closure cascades?',
    'bottleneck_segments': [{'segment': seg, 'cascade_impact': impact} for seg, impact in bottlenecks],
    'cascade_edges': edges,
    'graph_metrics': {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'density': float(nx.density(G)),
        'weakly_connected_components': nx.number_weakly_connected_components(G)
    }
}

print(f"\n=== RESULT JSON ===")
print(json.dumps(result, indent=2))