import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

np.random.seed(42)

# Synthetic thrift store pricing data with endogenous donor-composition shift
# Structure: items transition from donations → category pricing tier → sold
# Donor mix shifts at day 50, triggering hidden inventory rule flip

def generate_thrift_network_data(days=100, shift_day=50):
    items = []
    for day in range(days):
        n_items = np.random.poisson(15)
        # Pre-shift: heavy clothing donations (cheap), light furniture (expensive)
        # Post-shift: heavy furniture donations (cheap), light clothing (expensive)
        is_post_shift = day > shift_day
        
        for _ in range(n_items):
            if not is_post_shift:
                # Pre-shift donor mix: 70% clothing, 30% furniture
                category = np.random.choice(['clothing', 'furniture'], p=[0.7, 0.3])
                price = 3.99 if category == 'clothing' else 19.99
            else:
                # Post-shift donor mix: 30% clothing, 70% furniture
                category = np.random.choice(['clothing', 'furniture'], p=[0.3, 0.7])
                # Store applies *new* rule: reverse pricing after day 50
                price = 8.99 if category == 'clothing' else 7.99
            
            items.append({
                'day': day,
                'category': category,
                'price': price + np.random.normal(0, 0.5)
            })
    
    return pd.DataFrame(items)

df = generate_thrift_network_data()

# Build bipartite graph: days → categories → price_tiers
# Edge weight = number of items flowing through that path
def build_donor_network(df, window_size=7):
    """
    Create bipartite network where:
    - Left: days (sources of donations)
    - Center: categories (inventory buckets)
    - Right: price tiers (output)
    """
    G = nx.DiGraph()
    
    # Discretize prices into tiers
    df['price_tier'] = pd.cut(df['price'], bins=[0, 5, 10, 25], labels=['cheap', 'mid', 'expensive'])
    
    for window_start in range(0, len(df) - window_size, window_size):
        window_df = df.iloc[window_start : window_start + window_size]
        
        # Day → Category edges
        day_node = f"day_{window_start // window_size}"
        G.add_node(day_node, layer='day')
        
        for category in window_df['category'].unique():
            cat_node = f"cat_{category}"
            count = (window_df['category'] == category).sum()
            G.add_node(cat_node, layer='category')
            G.add_edge(day_node, cat_node, weight=count)
        
        # Category → Tier edges
        for category in window_df['category'].unique():
            cat_node = f"cat_{category}"
            cat_data = window_df[window_df['category'] == category]
            
            for tier in cat_data['price_tier'].unique():
                if pd.isna(tier):
                    continue
                tier_node = f"tier_{tier}_{category}"
                count = (cat_data['price_tier'] == tier).sum()
                G.add_node(tier_node, layer='tier')
                G.add_edge(cat_node, tier_node, weight=count)
    
    return G

G = build_donor_network(df)

# Compute betweenness centrality for category nodes (should be bridges)
bet_centrality = nx.betweenness_centrality(G, weight='weight')
cat_nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == 'category']
cat_betweenness = {n: bet_centrality.get(n, 0) for n in cat_nodes}

print("=== Betweenness Centrality (Category Nodes as Bridges) ===")
for node, cent in sorted(cat_betweenness.items(), key=lambda x: x[1], reverse=True):
    print(f"{node}: {cent:.4f}")

# Split by pre/post shift and detect collapse
window_size = 7
pre_shift_windows = 50 // window_size
post_shift_windows = (100 - 50) // window_size

def get_window_betweenness(G, windows_start, windows_end):
    nodes = [n for n, d in G.nodes(data=True) 
             if d.get('layer') == 'category' and f"day_{int(windows_start)}" <= f"day_{int(windows_end)}"]
    return [bet_centrality.get(n, 0) for n in nodes if n in cat_nodes]

pre_bet = []
post_bet = []

for node in cat_nodes:
    window_idx = int(node.split('_')[-1]) if '_' in node else -1
    if window_idx < pre_shift_windows:
        pre_bet.append(bet_centrality.get(node, 0))
    elif window_idx >= pre_shift_windows:
        post_bet.append(bet_centrality.get(node, 0))

print(f"\n=== Betweenness Collapse Signal ===")
print(f"Pre-shift mean betweenness: {np.mean(pre_bet):.4f} (stable donor mix)")
print(f"Post-shift mean betweenness: {np.mean(post_bet):.4f} (shifted donor mix)")
print(f"Collapse magnitude: {abs(np.mean(pre_bet) - np.mean(post_bet)):.4f}")
if abs(np.mean(pre_bet) - np.mean(post_bet)) > 0.01:
    print("✓ SIGNAL DETECTED: Betweenness collapse indicates hidden pricing regime change")
else:
    print("✗ No collapse detected")

print(f"\nNetwork stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")