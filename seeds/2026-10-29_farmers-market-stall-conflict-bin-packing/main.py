import pandas as pd
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import json

np.random.seed(42)

# Synthetic data: 20 vendors, each with a stall footprint (m²) and hidden affinity groups
vendors = [f"V{i:02d}" for i in range(20)]
footprints = np.random.uniform(8, 25, len(vendors))  # stall size in m²

# Hidden affinity groups: vendors that share equipment/power/water or customer flow
# Group 1: juice/beverage vendors (share water/power)
# Group 2: prepared-food vendors (share heavy power)
# Group 3: produce vendors (share misting system)
affinity_groups = {
    'beverages': [0, 1, 2],
    'prepared_food': [3, 4, 5, 6],
    'produce': [7, 8, 9, 10],
    'independent': list(range(11, 20))
}

def compute_affinity_cost(vendor_pair, affinity_groups):
    """Return conflict cost if two vendors are in the same affinity group (hidden constraint)."""
    for group_vendors in affinity_groups.values():
        if vendor_pair[0] in group_vendors and vendor_pair[1] in group_vendors:
            return 5.0  # Large penalty: vendors in same group conflict when co-scheduled
    return 0.0

def bin_pack_greedy(vendor_subset, footprints, slot_capacity=80):
    """Greedy bin-packing: return (num_bins_used, is_feasible)."""
    indices = sorted(vendor_subset, key=lambda i: footprints[i], reverse=True)
    bins = []
    for idx in indices:
        placed = False
        for b in bins:
            if sum(footprints[j] for j in b) + footprints[idx] <= slot_capacity:
                b.append(idx)
                placed = True
                break
        if not placed:
            bins.append([idx])
    return len(bins), len(bins) <= 5  # 5 available stall slots

def compute_infeasibility_score(vendor_list, footprints, affinity_groups):
    """
    Try to pack all vendors in vendor_list.
    If infeasible (need >5 bins), score = bins_needed - 5.
    If feasible, score = 0 (no signal).
    This reverses the typical bin-packing objective: we *want* to see failures
    because they signal hidden constraints.
    """
    bins_needed, is_feasible = bin_pack_greedy(list(range(len(vendor_list))), footprints)
    if is_feasible:
        return 0.0
    return float(bins_needed - 5)

# Simulate weekly vendor attendance: 50 weeks, each week 15-18 vendors show up
weeks_data = []
for week in range(50):
    n_vendors = np.random.randint(15, 19)
    attending = np.random.choice(len(vendors), n_vendors, replace=False)
    bins_needed, is_feasible = bin_pack_greedy(attending, footprints)
    weeks_data.append({
        'week': week,
        'attending': sorted(attending.tolist()),
        'n_vendors': n_vendors,
        'bins_needed': bins_needed,
        'feasible': is_feasible,
        'infeasibility_score': float(bins_needed - 5) if not is_feasible else 0.0
    })

df = pd.DataFrame(weeks_data)

# Now reverse-engineer: for each vendor pair, count how often they co-attend
# weeks where infeasibility occurred
pair_cooccurrence_in_infeasible_weeks = {}
for pair in combinations(range(len(vendors)), 2):
    v1, v2 = pair
    count = 0
    for _, row in df.iterrows():
        if not row['feasible'] and v1 in row['attending'] and v2 in row['attending']:
            count += 1
    pair_cooccurrence_in_infeasible_weeks[pair] = count

# Rank pairs by co-occurrence in infeasible weeks
top_pairs = sorted(pair_cooccurrence_in_infeasible_weeks.items(),
                   key=lambda x: x[1], reverse=True)[:15]

print("=== Farmers Market Stall Conflict Detection ===")
print(f"Total weeks analyzed: {len(df)}")
print(f"Weeks with infeasibility: {(~df['feasible']).sum()}")
print()
print("Top vendor pairs co-occurring in infeasible weeks (likely hidden dependencies):")
for (v1, v2), count in top_pairs:
    actual_group = None
    for group_name, group_members in affinity_groups.items():
        if v1 in group_members and v2 in group_members:
            actual_group = group_name
            break
    status = "✓ DETECTED" if actual_group else "✗ FALSE POS"
    print(f"  {vendors[v1]} + {vendors[v2]}: {count} co-occurrences {status} (group: {actual_group})")

print()
print("Interpretation: Vendor pairs that repeatedly co-attend during infeasible weeks")
print("likely share hidden operational constraints (power, water, foot-traffic).")
print("Standard bin-packing treats all vendors as independent, but infeasibility")
print("patterns reveal latent affinity dependencies.")

# Export for validation
results = {
    'top_pairs_by_infeasibility_cooccurrence': [
        {'vendor_1': vendors[p[0]], 'vendor_2': vendors[p[1]], 'cooccurrence_count': int(c)}
        for p, c in top_pairs[:10]
    ],
    'total_infeasible_weeks': int((~df['feasible']).sum()),
    'overall_feasibility_rate': float(df['feasible'].mean())
}
print()
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to results.json")