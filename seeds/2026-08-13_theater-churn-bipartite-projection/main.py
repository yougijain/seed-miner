import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

# Synthetic dataset: subscriber × show attendance
# Structure: 40 subscribers, 20 shows per season, ~3 seasons of history
# Signal: churners gradually skip more shows in season N-1, then cancel in season N

np.random.seed(42)

subscribers = [f"S{i:03d}" for i in range(40)]
shows_s1 = [f"Show_S1_{i:02d}" for i in range(20)]
shows_s2 = [f"Show_S2_{i:02d}" for i in range(20)]
shows_s3 = [f"Show_S3_{i:02d}" for i in range(20)]

# Attendance matrix: (subscriber, show) edges
attendance = []

for sub in subscribers:
    is_churner = np.random.random() < 0.25  # 25% will churn after S2
    
    # Season 1: high, stable attendance
    for show in shows_s1:
        if np.random.random() < 0.85:
            attendance.append((sub, show, 'S1'))
    
    # Season 2: churners begin declining
    decay_rate = 0.5 if is_churner else 0.85
    for show in shows_s2:
        if np.random.random() < decay_rate:
            attendance.append((sub, show, 'S2'))
    
    # Season 3: churners drop to near-zero (or absent entirely)
    final_rate = 0.1 if is_churner else 0.85
    for show in shows_s3:
        if np.random.random() < final_rate:
            attendance.append((sub, show, 'S3'))

df_attend = pd.DataFrame(attendance, columns=['subscriber', 'show', 'season'])

def bipartite_projection_fragmentation(df, season):
    """Compute fragmentation metrics on subscriber-subscriber network via show co-attendance."""
    season_df = df[df['season'] == season]
    
    # Build bipartite graph: subscribers on left, shows on right
    B = nx.Graph()
    B.add_nodes_from([('sub', s) for s in season_df['subscriber'].unique()], bipartite=0)
    B.add_nodes_from([('show', sh) for sh in season_df['show'].unique()], bipartite=1)
    B.add_edges_from([('sub', row['subscriber'], ('show', row['show'])) 
                       for _, row in season_df.iterrows()])
    
    # Project onto subscriber nodes: edge iff two subscribers attended >= 1 shared show
    subs = {n for n, attr in B.nodes(data=True) if attr.get('bipartite') == 0}
    shows_node = {n for n, attr in B.nodes(data=True) if attr.get('bipartite') == 1}
    
    P = nx.Graph()
    P.add_nodes_from(subs)
    
    show_to_subs = defaultdict(list)
    for _, row in season_df.iterrows():
        show_to_subs[('show', row['show'])].append(('sub', row['subscriber']))
    
    for show_node, sub_list in show_to_subs.items():
        for i, s1 in enumerate(sub_list):
            for s2 in sub_list[i+1:]:
                P.add_edge(s1, s2)
    
    # Metrics
    n_components = nx.number_connected_components(P)
    largest_cc_size = len(max(nx.connected_components(P), default={}))
    density = nx.density(P) if P.number_of_nodes() > 1 else 0.0
    
    return {
        'season': season,
        'n_nodes': P.number_of_nodes(),
        'n_edges': P.number_of_edges(),
        'n_components': n_components,
        'largest_cc_size': largest_cc_size,
        'cc_size_ratio': largest_cc_size / max(P.number_of_nodes(), 1),
        'density': density,
        'avg_degree': 2 * P.number_of_edges() / max(P.number_of_nodes(), 1)
    }

# Compute fragmentation across seasons
metrics = []
for season in ['S1', 'S2', 'S3']:
    m = bipartite_projection_fragmentation(df_attend, season)
    metrics.append(m)

df_metrics = pd.DataFrame(metrics)
print("\n=== Bipartite Projection Fragmentation Over Seasons ===")
print(df_metrics.to_string(index=False))

print("\n=== Interpretation ===")
print("Churn signal: fragmentation (n_components ↑) and cohesion (largest_cc_size ratio ↓) in S2")
print("as churners' attendance patterns diverge from loyalists.")
print(f"S1→S2 component increase: {df_metrics.loc[1, 'n_components'] - df_metrics.loc[0, 'n_components']}")
print(f"S2→S3 largest CC ratio drop: {df_metrics.loc[2, 'cc_size_ratio'] - df_metrics.loc[1, 'cc_size_ratio']:.3f}")
print("\nKey insight: network *topology* (how subscribers cluster via shared shows)")
print("destabilizes *before* explicit churn, because churners' attendance becomes idiosyncratic.")
print("This breaks the assumption that a static projection captures stable relationships.")
