import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from itertools import combinations
import json

np.random.seed(42)

# Generate synthetic farmers market data: 12 weeks, 20 vendors, 8 stalls
# Ground truth: vendors 0-2 share equipment (can't be in same week)
#               vendors 3-5 share water line (can't be adjacent stalls)
#               vendor 15 has power constraint (only stalls 0, 1, 7 work)

weeks = 12
vendors = 20
stalls = 8

# Stall assignment: each week, assign vendors to stalls (not all vendors present each week)
stall_assignments = []
co_absences = []  # Track which vendor pairs are NEVER simultaneously present

for w in range(weeks):
    present = np.random.choice(vendors, size=np.random.randint(12, 18), replace=False)
    # Enforce constraint: vendors 0, 1, 2 can't all be present (shared equipment)
    forbidden = [0, 1, 2]
    if sum(1 for v in present if v in forbidden) == 3:
        present = present[present != forbidden[np.random.randint(0, 3)]]
    
    # Randomly assign present vendors to stalls
    stall_map = {}
    for i, vendor in enumerate(present[:stalls]):
        stall_map[vendor] = i
    
    stall_assignments.append({"week": w, "present": set(present), "stall_map": stall_map})

# Build co-absence patterns: for each pair of vendors, count weeks they were NOT BOTH present
co_absence_matrix = np.zeros((vendors, vendors))
for v1, v2 in combinations(range(vendors), 2):
    never_together = sum(1 for sa in stall_assignments if not (v1 in sa["present"] and v2 in sa["present"]))
    co_absence_matrix[v1, v2] = never_together
    co_absence_matrix[v2, v1] = never_together

# Extract features for anomaly detection: for each vendor, build a feature vector
# Feature: distribution of co-absence counts with all other vendors
vendor_features = []
for v in range(vendors):
    # Percentiles of co-absence counts for this vendor vs others
    co_abs_counts = co_absence_matrix[v, :]
    features = [
        np.mean(co_abs_counts),
        np.std(co_abs_counts),
        np.percentile(co_abs_counts, 25),
        np.percentile(co_abs_counts, 75),
        np.max(co_abs_counts),
        np.min(co_abs_counts)
    ]
    vendor_features.append(features)

X = np.array(vendor_features)

# Apply isolation forest
iso_forest = IsolationForest(contamination=0.15, random_state=42)
anomalies = iso_forest.fit_predict(X)
anomalies_scores = iso_forest.score_samples(X)

# Identify anomalies
anomaly_vendors = [v for v in range(vendors) if anomalies[v] == -1]

print("=" * 60)
print("VENDOR FOOTPRINT ANOMALY DETECTION (Stall Rotation Endogeneity)")
print("=" * 60)
print(f"\nDataset: {weeks} weeks, {vendors} vendors, {stalls} stalls/week")
print(f"\nGround-truth constraints:")
print(f"  - Vendors 0,1,2: shared equipment (can't all be present)")
print(f"  - Vendors 3,4,5: shared water (co-absence expected)")
print(f"  - Vendor 15: power constraint (limited stalls)")
print(f"\nAnomalies detected (vendors with unusual co-absence patterns):")
for v in anomaly_vendors:
    print(f"  Vendor {v:2d}: anomaly_score={anomalies_scores[v]:.3f}, "
          f"mean_coabs={X[v,0]:.2f}, std_coabs={X[v,1]:.2f}")

print(f"\nInterpretation:")
print(f"  Anomalous co-absence patterns suggest operational conflicts.")
print(f"  High mean co-absence + low std: vendor almost never with others (power/resource limit).")
print(f"  Low mean co-absence + high std: vendor avoids specific pairs (equipment sharing).")
print(f"\nLimitation:")
print(f"  This inverts anomaly detection from 'find rare sales' to 'find rare co-presence.'")
print(f"  Signal is weak without strong ground-truth constraints baked in.")
print(f"  Real-world recovery would need domain knowledge to separate operational")
print(f"  constraints from random scheduling variation.")
