import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import json

np.random.seed(42)

# Synthetic data: 12 colonies, 20 inspection windows
# Each inspection window records which health metrics were *mentioned* in notes
colonies = [f"Col_{i:02d}" for i in range(12)]
metrics = ["pollen_stores", "flight_activity", "brood_pattern", "mite_load", "comb_condition"]
windows = [f"week_{i}" for i in range(20)]

# Generate inspection data: 1 = metric mentioned, 0 = metric NOT mentioned (co-silence signal)
np.random.seed(42)
inspection_matrix = np.random.binomial(1, 0.6, (len(colonies), len(windows) * len(metrics))).reshape(
    len(colonies), len(windows), len(metrics)
)

# Inject foraging-pathway dependency: colonies 0-3 and 6-8 share a forage zone
# They exhibit correlated metric silence in weeks 8-14 (simulating shared resource constraints)
for col in [0, 1, 2, 3]:
    inspection_matrix[col, 8:14, [0, 1]] = 0  # silence on pollen_stores, flight_activity

for col in [6, 7, 8]:
    inspection_matrix[col, 8:14, [0, 1]] = 0

# Compute co-silence: for each pair of colonies, measure how often their inspection windows
# have identical metric-silence patterns (both mention or both omit the same metric in the same window)
co_silence_matrix = np.zeros((len(colonies), len(colonies)))

for i in range(len(colonies)):
    for j in range(i + 1, len(colonies)):
        # Stack windows as feature vectors; each element is (metric_mentioned_by_i, metric_mentioned_by_j)
        silence_agreement = 0
        total_comparisons = 0
        for w in range(len(windows)):
            for m in range(len(metrics)):
                # co-silence: both did NOT mention (both = 0)
                if inspection_matrix[i, w, m] == 0 and inspection_matrix[j, w, m] == 0:
                    silence_agreement += 1
                total_comparisons += 1
        # Normalize: higher value = stronger co-silence pattern
        co_silence_matrix[i, j] = silence_agreement / total_comparisons
        co_silence_matrix[j, i] = co_silence_matrix[i, j]

# Matching cost: convert co-silence agreement to cost (lower cost = higher co-silence)
# We want to match colonies that exhibit strong co-silence (likely share bottleneck)
cost_matrix = 1.0 - co_silence_matrix
np.fill_diagonal(cost_matrix, np.inf)  # No self-matching

# Apply bipartite matching: partition colonies into two groups, match high-co-silence pairs
# Use linear_sum_assignment to find minimum-cost pairing
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Filter to meaningful matches (co-silence > 0.5, i.e., cost < 0.5)
matches = []
for r, c in zip(row_ind, col_ind):
    if r < c and co_silence_matrix[r, c] > 0.5:
        matches.append((colonies[r], colonies[c], co_silence_matrix[r, c]))

matches.sort(key=lambda x: -x[2])

print("=== Colony Foraging Bottleneck Detection via Co-Silence Matching ===")
print(f"\nInspection data: {len(colonies)} colonies × {len(windows)} weeks × {len(metrics)} metrics\n")
print("High co-silence colony pairs (likely share foraging pathway bottleneck):")
for col_a, col_b, score in matches[:8]:
    print(f"  {col_a} ↔ {col_b} : co-silence score = {score:.3f}")

if not matches:
    print("  (No pairs with co-silence > 0.5 detected)")

print(f"\nTotal high-confidence matches: {len(matches)}")

# Validation: check if injected bottleneck colonies cluster
injected_group_a = set(["Col_00", "Col_01", "Col_02", "Col_03"])
injected_group_b = set(["Col_06", "Col_07", "Col_08"])

detected_pairs_a = set()
detected_pairs_b = set()

for col_a, col_b, _ in matches:
    if col_a in injected_group_a and col_b in injected_group_a:
        detected_pairs_a.add((col_a, col_b))
    elif col_a in injected_group_b and col_b in injected_group_b:
        detected_pairs_b.add((col_a, col_b))

print(f"\nValidation:")
print(f"  Detected within-group-A pairs: {len(detected_pairs_a)}")
print(f"  Detected within-group-B pairs: {len(detected_pairs_b)}")
print(f"  Signal recovery: {'✓ DETECTED' if detected_pairs_a or detected_pairs_b else '✗ MISSED'}")
