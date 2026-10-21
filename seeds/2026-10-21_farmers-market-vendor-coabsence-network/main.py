import pandas as pd
import networkx as nx
from collections import defaultdict
import json

# Generate synthetic transaction data: customer visits with vendor purchases
# Structure: each customer visit may span 1+ vendors, revealing substitutability patterns
visits = [
    {"cust_id": 1, "visit_id": "v1", "vendors": ["lettuce_stand", "herb_grower"]},
    {"cust_id": 1, "visit_id": "v2", "vendors": ["lettuce_stand", "tomato_stand"]},
    {"cust_id": 2, "visit_id": "v3", "vendors": ["lettuce_stand", "herb_grower"]},
    {"cust_id": 2, "visit_id": "v4", "vendors": ["tomato_stand", "herb_grower"]},
    {"cust_id": 3, "visit_id": "v5", "vendors": ["lettuce_stand"]},
    {"cust_id": 3, "visit_id": "v6", "vendors": ["tomato_stand"]},
    {"cust_id": 4, "visit_id": "v7", "vendors": ["cheese_stand", "bread_stand"]},
    {"cust_id": 4, "visit_id": "v8", "vendors": ["cheese_stand", "butter_stand"]},
    {"cust_id": 5, "visit_id": "v9", "vendors": ["bread_stand", "butter_stand"]},
    {"cust_id": 5, "visit_id": "v10", "vendors": ["cheese_stand", "bread_stand"]},
    {"cust_id": 6, "visit_id": "v11", "vendors": ["bread_stand"]},
    {"cust_id": 6, "visit_id": "v12", "vendors": ["butter_stand"]},
]

df = pd.DataFrame(visits)

# Build co-visitation graph: edge weight = # of visits where both vendors appear
graph = nx.Graph()
vendor_visits = defaultdict(set)

for _, row in df.iterrows():
    vendors = row["vendors"]
    for v in vendors:
        vendor_visits[v].add(row["visit_id"])
    for i, v1 in enumerate(vendors):
        for v2 in vendors[i+1:]:
            if graph.has_edge(v1, v2):
                graph[v1][v2]["weight"] += 1
            else:
                graph.add_edge(v1, v2, weight=1)

print("=== Co-Visitation Graph ===")
print(f"Vendors (nodes): {list(graph.nodes())}")
print(f"Co-visits (edges):")
for u, v, w in graph.edges(data=True):
    print(f"  {u} <-> {v}: {w['weight']} shared visits")

# Detect substitutability clusters via community detection
# High-weight edges = substitutable vendors
from networkx.algorithms import community
communities = list(community.greedy_modularity_communities(graph))

print("\n=== Vendor Clusters (Substitutability Groups) ===")
for i, comm in enumerate(communities):
    print(f"Cluster {i}: {sorted(comm)}")

# Now simulate vendor absence: remove one vendor and measure network cohesion collapse
print("\n=== Substitutability Robustness (Absence Sensitivity) ===")
for vendor in graph.nodes():
    G_copy = graph.copy()
    G_copy.remove_node(vendor)
    if len(G_copy) > 1:
        remaining_edges = G_copy.number_of_edges()
        orig_edges = graph.number_of_edges()
        cohesion_loss = 1 - (remaining_edges / max(orig_edges, 1))
        print(f"Remove '{vendor}': {cohesion_loss:.2%} network edge loss (high = highly substitutable)")

print("\n=== Interpretation ===")
print("Vendors in same cluster are substitutable (customers visit them together).")
print("High absence-sensitivity vendors are hubs; their loss fragments the network.")
print("Inverse: vendors whose absence causes minimal edge loss are niche/non-substitutable.")
