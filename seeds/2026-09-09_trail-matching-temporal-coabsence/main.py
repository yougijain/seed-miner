import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from datetime import datetime, timedelta
import json

np.random.seed(42)

# Generate synthetic trail closure events (our signal of co-absence)
trails = [f"Trail_{i}" for i in range(10)]
start_date = datetime(2024, 1, 1)
closures = []

# Ground truth: trails cluster into 3 latent difficulty groups
# Easy: 0-2, Medium: 3-5, Hard: 6-9
# Co-manage closures within difficulty; independent closures between groups
easy_trails = set(range(3))
medium_trails = set(range(3, 6))
hard_trails = set(range(6, 10))

for day in range(365):
    date = start_date + timedelta(days=day)
    # Rare weather closure affecting whole difficulty tier
    if np.random.random() < 0.08:
        tier = np.random.choice([easy_trails, medium_trails, hard_trails])
        for t in tier:
            closures.append({"trail_id": t, "date": date.isoformat(), "reason": "weather"})
    # Maintenance cascades within same tier (co-closure signal)
    if np.random.random() < 0.05:
        tier = np.random.choice([easy_trails, medium_trails, hard_trails])
        for t in np.random.choice(list(tier), size=max(1, len(tier)//2), replace=False):
            closures.append({"trail_id": t, "date": date.isoformat(), "reason": "maintenance"})

df = pd.DataFrame(closures)
df["date"] = pd.to_datetime(df["date"])
df = df.groupby("trail_id")["date"].apply(set).reset_index()
df.columns = ["trail_id", "closure_dates"]
print(f"Loaded {len(df)} trails with closure history.")
print(df.head())

# Build co-absence adjacency matrix via temporal overlap
trail_to_idx = {i: i for i in range(len(df))}
co_absence = np.zeros((len(df), len(df)))

for i, row_i in df.iterrows():
    for j, row_j in df.iterrows():
        if i < j:
            overlap = len(row_i["closure_dates"] & row_j["closure_dates"])
            co_absence[i, j] = overlap
            co_absence[j, i] = overlap

# Normalize to [0,1]
co_absence = co_absence / (co_absence.max() + 1e-9)

print(f"\nCo-absence matrix shape: {co_absence.shape}")
print(f"Co-absence range: [{co_absence.min():.3f}, {co_absence.max():.3f}]")

# Bipartite matching: pair trails to maximize co-absence affinity
# Split trails into left (0-4) and right (5-9) partitions
left_trails = list(range(5))
right_trails = list(range(5, 10))

# Bipartite cost matrix: use co-absence as affinity
bipartite_cost = np.zeros((len(left_trails), len(right_trails)))
for i, left_idx in enumerate(left_trails):
    for j, right_idx in enumerate(right_trails):
        bipartite_cost[i, j] = -co_absence[left_idx, right_idx]  # negative for maximization

# Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(bipartite_cost)
matches = [(left_trails[r], right_trails[c], -bipartite_cost[r, c]) for r, c in zip(row_ind, col_ind)]

print(f"\n=== Bipartite Matching Results (co-absence affinity) ===")
for left, right, affinity in sorted(matches, key=lambda x: -x[2]):
    print(f"Trail_{left} <-> Trail_{right}: co-absence affinity = {affinity:.3f}")

# Evaluate: do matched pairs belong to same latent difficulty?
groups = {
    0: "Easy", 1: "Easy", 2: "Easy",
    3: "Medium", 4: "Medium", 5: "Medium",
    6: "Hard", 7: "Hard", 8: "Hard", 9: "Hard"
}

same_tier = 0
for left, right, _ in matches:
    if groups[left] == groups[right]:
        same_tier += 1

accuracy = same_tier / len(matches)
print(f"\n=== Evaluation ===")
print(f"Matches within same difficulty tier: {same_tier}/{len(matches)} = {accuracy:.1%}")
print(f"(Random baseline: ~33% for 3 tiers)")

# Output for inspection
result = {
    "matches": [(int(l), int(r), float(a)) for l, r, a in matches],
    "ground_truth_groups": {k: v for k, v in groups.items()},
    "accuracy": accuracy
}

with open("results.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to results.json")
