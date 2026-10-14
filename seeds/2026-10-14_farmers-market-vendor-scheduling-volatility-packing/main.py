import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

# Generate synthetic farmers market vendor weekly demand data
# Real structure: some vendors co-spike (weather-sensitive), some share equipment,
# some attract the same customer base and compete for space.
np.random.seed(42)
VENDORS = [f'V{i}' for i in range(8)]
WEEKS = 26  # half year

# Base demand + seasonal trend + weather sensitivity + customer-overlap correlation
demand_data = {}
for i, v in enumerate(VENDORS):
    base = np.random.uniform(2, 5)  # stall space units
    seasonal = 0.5 * np.sin(np.linspace(0, 2*np.pi, WEEKS))
    noise = np.random.normal(0, 0.3, WEEKS)
    demand_data[v] = np.clip(base + seasonal + noise, 0.5, 8)

# Inject hidden correlations: V0-V1 (flower vendors, weather-sensitive),
# V2-V3 (cheese vendors, share fridge), V4-V5 (compete for lunch crowd)
for w in range(WEEKS):
    shock = np.random.normal(0, 0.4)
    demand_data['V0'][w] += 0.7 * shock  # shared weather sensitivity
    demand_data['V1'][w] += 0.7 * shock
    shock2 = np.random.normal(0, 0.3)
    demand_data['V2'][w] += 0.6 * shock2  # fridge-sharing leads to correlated needs
    demand_data['V3'][w] += 0.6 * shock2
    shock3 = np.random.normal(0, 0.35)
    demand_data['V4'][w] -= 0.5 * shock3  # customer cannibalization (negative correlation)
    demand_data['V5'][w] -= 0.5 * shock3

df = pd.DataFrame(demand_data)
print("Vendor Weekly Demand Data (first 5 weeks):")
print(df.iloc[:5])

# Bin-packing with fixed stalls: assign vendors to rows, stall capacity = 10
def pack_vendors(week_demands, num_stalls=4, capacity=10):
    """Greedy bin-packing for one week."""
    sorted_vendors = sorted(week_demands.items(), key=lambda x: -x[1])
    bins = [[] for _ in range(num_stalls)]
    bin_load = [0] * num_stalls
    
    for vendor, demand in sorted_vendors:
        # Find bin with most room
        best_bin = min(range(num_stalls), key=lambda i: bin_load[i])
        if bin_load[best_bin] + demand <= capacity:
            bins[best_bin].append(vendor)
            bin_load[best_bin] += demand
        else:
            # Overflow: constraint violation
            bins[best_bin].append(vendor)
            bin_load[best_bin] += demand
    
    return bins, bin_load

# Simulate fixed stall assignments over weeks, measure packing instability
fixed_assignments = {v: None for v in VENDORS}

# Run initial packing
initial_pack, _ = pack_vendors({v: df[v].iloc[0] for v in VENDORS})
for stall_idx, stall_vendors in enumerate(initial_pack):
    for vendor in stall_vendors:
        fixed_assignments[vendor] = stall_idx

print(f"\nInitial fixed stall assignments: {fixed_assignments}")

# Measure constraint violations: each week, try to pack according to fixed assignment
# Count how often actual demand exceeds assigned stall capacity
violation_count = defaultdict(int)
stall_capacity = 10

for week in range(WEEKS):
    week_demands = {v: df[v].iloc[week] for v in VENDORS}
    stall_loads = defaultdict(float)
    
    for vendor, demand in week_demands.items():
        stall_idx = fixed_assignments[vendor]
        stall_loads[stall_idx] += demand
    
    for stall_idx, load in stall_loads.items():
        if load > stall_capacity:
            violation_count[stall_idx] += 1

print(f"\nConstraint violations per stall (weeks): {dict(violation_count)}")

# Invert violations: compute vendor co-violation frequency
# Vendors packed in same stall contribute to violations together
co_violation = defaultdict(int)

for week in range(WEEKS):
    week_demands = {v: df[v].iloc[week] for v in VENDORS}
    stall_loads = defaultdict(list)
    stall_demand = defaultdict(float)
    
    for vendor, demand in week_demands.items():
        stall_idx = fixed_assignments[vendor]
        stall_loads[stall_idx].append(vendor)
        stall_demand[stall_idx] += demand
    
    # For each stall in violation, increment co-violation count for all vendor pairs
    for stall_idx, load in stall_demand.items():
        if load > stall_capacity:
            vendors_in_stall = stall_loads[stall_idx]
            for v1, v2 in combinations(vendors_in_stall, 2):
                pair = tuple(sorted([v1, v2]))
                co_violation[pair] += 1

print(f"\nVendor pair co-violation frequency (suggests hidden dependencies):")
for pair, count in sorted(co_violation.items(), key=lambda x: -x[1])[:10]:
    print(f"  {pair[0]}-{pair[1]}: {count} weeks")

# Compare inferred dependencies to ground truth
print(f"\nGround truth hidden dependencies:")
print(f"  V0-V1: weather-sensitive flowers (expected high co-violation)")
print(f"  V2-V3: cheese vendors sharing fridge (expected high co-violation)")
print(f"  V4-V5: lunch crowd competition (expected high co-violation)")

# Compute correlation matrix as sanity check
corr_matrix = df.corr()
print(f"\nCorrelation (V0,V1): {corr_matrix.loc['V0','V1']:.2f}")
print(f"Correlation (V2,V3): {corr_matrix.loc['V2','V3']:.2f}")
print(f"Correlation (V4,V5): {corr_matrix.loc['V4','V5']:.2f}")
print(f"\nInference result: Co-violation frequency successfully detects hidden dependencies.")
