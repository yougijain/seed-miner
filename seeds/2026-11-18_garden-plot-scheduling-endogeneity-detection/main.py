import numpy as np
import pandas as pd
from collections import defaultdict
import json

np.random.seed(42)

# Generate synthetic garden plot data with ENDOGENOUS planting shifts.
# Key signal: gardeners who succeed in season N plant earlier in season N+1,
# compressing the planting window and making bin-packing fail.

num_gardeners = 12
num_weeks = 20
num_seasons = 3

data = []

for season in range(num_seasons):
    gardener_yields = {}
    for gid in range(num_gardeners):
        # Base planting week: normally distributed around week 2-4
        base_week = np.random.randint(2, 5)
        
        # ENDOGENOUS SHIFT: gardeners with high prior-season yield shift earlier
        if season > 0 and gid in top_yielders:
            base_week = max(1, base_week - 1)  # shift earlier if successful last season
        
        planting_week = base_week + np.random.randint(-1, 2)
        planting_week = max(1, min(planting_week, num_weeks - 2))
        
        # Yield depends on week (earlier = more time, but frost risk)
        yield_val = 100 + 50 * np.sin((planting_week - 3) / 5) + np.random.randn() * 10
        yield_val = max(20, yield_val)
        
        gardener_yields[gid] = yield_val
        
        for week in range(planting_week, min(planting_week + 3, num_weeks)):
            data.append({
                'season': season,
                'gardener_id': gid,
                'week': week,
                'plot_days': np.random.randint(10, 25),  # occupancy duration
                'yield': yield_val
            })
    
    # Identify top yielders for next season
    top_yielders = sorted(gardener_yields.items(), key=lambda x: x[1], reverse=True)[:4]
    top_yielders = set([g[0] for g in top_yielders])

df = pd.DataFrame(data)

# BIN-PACKING FEASIBILITY TEST:
# Try to fit all gardeners' planting windows into 3 available plot zones
# per week without overlap. Measure how often this fails.

def test_packing_feasibility(season_df, num_plots=3):
    """Greedy bin-packing: assign gardener planting windows to plots.
    Return (success, num_conflicts, avg_plot_utilization)."""
    plots = defaultdict(list)  # plot_id -> [(gid, week_start, week_end)]
    gardeners = season_df.groupby('gardener_id').apply(
        lambda g: (g['week'].min(), g['week'].max())
    ).to_dict()
    
    conflicts = 0
    for gid, (w_start, w_end) in gardeners.items():
        placed = False
        for plot_id in range(num_plots):
            # Check if this gardener's window overlaps with any existing in this plot
            overlap = any(
                not (w_end < other_start or w_start > other_end)
                for _, other_start, other_end in plots[plot_id]
            )
            if not overlap:
                plots[plot_id].append((gid, w_start, w_end))
                placed = True
                break
        if not placed:
            conflicts += 1
    
    utilization = sum(len(v) for v in plots.values()) / (num_plots * len(gardeners))
    return len(conflicts) == 0, conflicts, utilization

# Run feasibility test per season
feasibility_results = []
for season in range(num_seasons):
    season_data = df[df['season'] == season]
    success, conflicts, util = test_packing_feasibility(season_data)
    feasibility_results.append({
        'season': season,
        'packing_success': success,
        'num_conflicts': conflicts,
        'plot_utilization': util
    })
    print(f"Season {season}: feasibility={success}, conflicts={conflicts}, util={util:.2f}")

feas_df = pd.DataFrame(feasibility_results)

# SIGNAL EXTRACTION:
# Measure the *change* in feasibility across seasons.
# Hypothesis: seasons where conflicts spike indicate endogenous timing compression.

conflict_delta = feas_df['num_conflicts'].diff()
print("\n--- Endogeneity Detection ---")
print(f"Conflict delta (season-to-season): {conflict_delta.tolist()}")
print(f"Feasibility trend: {feas_df['packing_success'].tolist()}")

# Identify gardeners with timing clustering (endogeneity signal)
print("\n--- Gardener Planting Timing Distribution ---")
for season in range(num_seasons):
    season_data = df[df['season'] == season]
    timings = season_data.groupby('gardener_id')['week'].min()
    print(f"Season {season}: week_min mean={timings.mean():.1f}, std={timings.std():.1f}")

print("\n--- Interpretation ---")
if conflict_delta[1] > conflict_delta[0]:
    print("Packing conflicts INCREASED from season 1→2.")
    print("Signal: Gardeners compressed their planting window (endogenous adaptation).")
    print("Optimization scheduler detected this as infeasibility, not resource shortage.")
else:
    print("Packing conflicts remained stable.")
    print("Signal: No strong evidence of endogenous timing shifts.")