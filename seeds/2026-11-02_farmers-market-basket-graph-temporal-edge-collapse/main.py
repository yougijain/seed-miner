import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
from itertools import combinations

np.random.seed(42)

# Synthetic data: customer transactions at farmers market
# Structure: each customer visits, buys from 1-4 vendors, recorded with timestamp
def generate_synthetic_basket_data(n_transactions=500, n_vendors=12, n_customers=80):
    """Generate transaction log with hidden vendor complementarity.
    Vendors 0-3: produce (strong co-purchase)
    Vendors 4-7: dairy (strong co-purchase)
    Vendors 8-11: baked goods (weak co-purchase)
    Hidden constraint: if produce vendor A is out of stock (randomly), 
    customers don't buy from produce vendors B,C,D that week.
    """
    records = []
    week_stockouts = defaultdict(set)  # week -> set of vendors out
    
    for txn_id in range(n_transactions):
        customer_id = np.random.randint(0, n_customers)
        timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 60))
        week = timestamp.isocalendar()[1]
        
        # Randomly declare stockouts for produce vendors in some weeks
        if np.random.random() < 0.15:
            week_stockouts[week].add(np.random.randint(0, 4))
        
        # Build basket with vendor complementarity
        basket = set()
        
        # Produce vendors: 0-3. Skip if related vendor is stocked out.
        if np.random.random() < 0.6:
            produce_vendors = [v for v in range(4) if v not in week_stockouts[week]]
            if produce_vendors:
                basket.add(np.random.choice(produce_vendors))
                # Strong co-purchase: if one produce vendor, likely buy from another
                if np.random.random() < 0.7 and len(produce_vendors) > 1:
                    basket.add(np.random.choice([v for v in produce_vendors if v not in basket]))
        
        # Dairy vendors: 4-7. Always available but co-purchase strongly.
        if np.random.random() < 0.5:
            basket.add(np.random.randint(4, 8))
            if np.random.random() < 0.6:
                dairy_choices = [v for v in range(4, 8) if v not in basket]
                if dairy_choices:
                    basket.add(np.random.choice(dairy_choices))
        
        # Baked goods: 8-11. Weak co-purchase.
        if np.random.random() < 0.4:
            basket.add(np.random.randint(8, 12))
            if np.random.random() < 0.3:
                bread_choices = [v for v in range(8, 12) if v not in basket]
                if bread_choices:
                    basket.add(np.random.choice(bread_choices))
        
        for vendor_id in basket:
            records.append({
                'transaction_id': txn_id,
                'customer_id': customer_id,
                'vendor_id': vendor_id,
                'timestamp': timestamp,
                'week': week
            })
    
    return pd.DataFrame(records)

def build_temporal_bipartite_graph(df, week):
    """Build customer-vendor bipartite graph for a week.
    Returns graph and edge set.
    """
    week_data = df[df['week'] == week]
    G = nx.Graph()
    edges = set()
    
    for _, row in week_data.iterrows():
        cust_node = f"C{row['customer_id']}"
        vend_node = f"V{row['vendor_id']}"
        G.add_edge(cust_node, vend_node)
        edges.add((cust_node, vend_node))
    
    return G, edges

def compute_vendor_copurchase_signature(df, week):
    """For each vendor, compute which other vendors it co-appears with in customer baskets."""
    week_data = df[df['week'] == week]
    vendor_baskets = defaultdict(set)
    
    for txn_id in week_data['transaction_id'].unique():
        txn_vendors = set(week_data[week_data['transaction_id'] == txn_id]['vendor_id'].values)
        for v in txn_vendors:
            vendor_baskets[v].update(txn_vendors - {v})
    
    return vendor_baskets

def temporal_edge_jaccard(signatures_w1, signatures_w2):
    """Compute Jaccard similarity of vendor co-appearance signatures week-to-week.
    Low similarity = edge collapse (vendors that co-purchased together stop doing so).
    """
    all_vendors = set(signatures_w1.keys()) | set(signatures_w2.keys())
    jaccards = {}
    
    for v in all_vendors:
        sig1 = signatures_w1.get(v, set())
        sig2 = signatures_w2.get(v, set())
        
        if len(sig1) + len(sig2) == 0:
            jaccards[v] = 1.0
        else:
            intersection = len(sig1 & sig2)
            union = len(sig1 | sig2)
            jaccards[v] = intersection / union if union > 0 else 0.0
    
    return jaccards

def detect_edge_collapse_weeks(df):
    """Find weeks where vendor co-purchase graphs show anomalous edge collapse."""
    weeks = sorted(df['week'].unique())
    collapses = []
    
    for i in range(1, len(weeks)):
        w1, w2 = weeks[i-1], weeks[i]
        sig_w1 = compute_vendor_copurchase_signature(df, w1)
        sig_w2 = compute_vendor_copurchase_signature(df, w2)
        
        jaccards = temporal_edge_jaccard(sig_w1, sig_w2)
        
        # Flag: if produce vendors (0-3) show abnormally low Jaccard, edge collapse detected
        produce_jaccards = [jaccards.get(v, 1.0) for v in range(4)]
        
        if produce_jaccards and np.mean(produce_jaccards) < 0.4:
            collapses.append({
                'week_transition': f"{w1}->{w2}",
                'mean_jaccard': np.mean(produce_jaccards),
                'produce_vendors_affected': [v for v in range(4) if jaccards.get(v, 1.0) < 0.4]
            })
    
    return collapses

if __name__ == '__main__':
    df = generate_synthetic_basket_data()
    
    print("=== Farmers Market Vendor Graph Temporal Edge Collapse ===")
    print(f"Total transactions: {len(df)}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Unique vendors: {df['vendor_id'].nunique()}")
    print(f"Date range: week {df['week'].min()} - {df['week'].max()}\n")
    
    collapses = detect_edge_collapse_weeks(df)
    
    if collapses:
        print("Edge Collapse Detected (vendor co-purchase edges vanish):")
        for collapse in collapses:
            print(f"  {collapse['week_transition']}: Jaccard={collapse['mean_jaccard']:.3f}, "
                  f"produce vendors {collapse['produce_vendors_affected']} lost co-purchase edges")
        print("\n  Interpretation: Hidden supply constraints (vendor stockout) broke produce-vendor complementarity.")
    else:
        print("No anomalous edge collapse detected.")
    
    print("\n=== Bipartite Graph Structure (Week 1) ===")
    G_w1, edges_w1 = build_temporal_bipartite_graph(df, df['week'].min())
    print(f"Nodes: {G_w1.number_of_nodes()}, Edges: {G_w1.number_of_edges()}")
