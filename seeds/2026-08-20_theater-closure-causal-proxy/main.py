import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic: 6 theaters, 12 weeks, staggered 1-week maintenance windows.
# Treatment: one theater closes for 1 week → its shows move to nearby theaters OR are postponed.
# Question: Does sales drop for nearby shows = substitution (causal) or just coincidence?

theaters = ['T_A', 'T_B', 'T_C', 'T_D', 'T_E', 'T_F']
weeks = range(1, 13)
neighbors = {'T_A': ['T_B', 'T_C'], 'T_B': ['T_A', 'T_C'], 'T_C': ['T_A', 'T_B'],
             'T_D': ['T_E', 'T_F'], 'T_E': ['T_D', 'T_F'], 'T_F': ['T_D', 'T_E']}

# Stagger closures: T_A closed week 3, T_D closed week 7
closure_schedule = {'T_A': 3, 'T_D': 7}

data = []
for theater in theaters:
    for week in weeks:
        is_closed = week == closure_schedule.get(theater, None)
        base_sales = 150 + np.random.normal(0, 20)
        
        # Causal spillover: if a neighbor is closed, this theater gains sales from substitution
        neighbor_closed = any(week == closure_schedule.get(n, None) for n in neighbors.get(theater, []))
        spillover_boost = 40 if neighbor_closed else 0
        
        # If this theater is closed, sales = 0; neighbors get spillover
        sales = 0 if is_closed else base_sales + spillover_boost
        sales = max(0, sales + np.random.normal(0, 10))
        
        data.append({
            'theater': theater,
            'week': week,
            'sales': sales,
            'treated': is_closed,
            'neighbor_treated': neighbor_closed and not is_closed,
            'closure_week': closure_schedule.get(theater, None),
        })

df = pd.DataFrame(data)

# --- Naive DiD: Compare treated theaters before/after closure ---
# Problem: We can't observe sales during closure, so we have to exclude those rows.
# This breaks DiD because "after" never happens for the treated unit.

treated_units = [t for t, w in closure_schedule.items()]
df_valid = df[~df['treated']].copy()  # Exclude closed-theater rows

print("=== Naive Approach (breaks because treated unit disappears) ===")
print(f"Treated theaters: {treated_units}")
print(f"Rows after excluding closures: {len(df_valid)} / {len(df)}")
print(f"Cannot compute DiD ATE because treated group is missing post-period.\n")

# --- Causal Proxy Strategy: Use neighbors as proxy for spillover ---
# Q: Does a neighbor's closure cause *neighboring* theaters' sales to rise?
# This is backward causality detection: if neighbor_closed → my sales up,
# then demand didn't disappear; it substituted.

print("=== Causal Proxy: DiD on Neighbors ===")
df_neighbors = df[df['neighbor_treated'] | ~df['neighbor_treated']].copy()
df_neighbors['post'] = df_neighbors['closure_week'].apply(lambda x: 1 if pd.notna(x) else 0)
df_neighbors['treated'] = df_neighbors['neighbor_treated'].astype(int)

# DiD estimator: (E[Y | treated=1, post=1] - E[Y | treated=1, post=0])
#              - (E[Y | treated=0, post=1] - E[Y | treated=0, post=0])
g1p1 = df_neighbors[(df_neighbors['treated'] == 1) & (df_neighbors['post'] == 1)]['sales'].mean()
g1p0 = df_neighbors[(df_neighbors['treated'] == 1) & (df_neighbors['post'] == 0)]['sales'].mean()
g0p1 = df_neighbors[(df_neighbors['treated'] == 0) & (df_neighbors['post'] == 1)]['sales'].mean()
g0p0 = df_neighbors[(df_neighbors['treated'] == 0) & (df_neighbors['post'] == 0)]['sales'].mean()

ate_neighbor = (g1p1 - g1p0) - (g0p1 - g0p0)

print(f"Neighbors of closed theater: post-closure sales change = ${ate_neighbor:.2f}")
print(f"  Treated group: {g1p0:.1f} → {g1p1:.1f} (change: +{g1p1 - g1p0:.1f})")
print(f"  Control group: {g0p0:.1f} → {g0p1:.1f} (change: +{g0p1 - g0p0:.1f})")
print(f"  DiD estimate: {ate_neighbor:.2f}\n")

if ate_neighbor > 15:
    print(f"→ Interpretation: Neighbors' sales rise after a local closure.")
    print(f"  This suggests SUBSTITUTION (demand exists, just moved).")
    print(f"  ⚠ Causal assumption: closure is exogenous (maintenance schedule).\n")
else:
    print(f"→ Interpretation: Weak spillover effect.")
    print(f"  Suggests either demand collapsed or substitution is local.\n")

# --- Sensitivity: Test for spatial confounding ---
print("=== Robustness: Is spillover effect robust to pre-period imbalance? ===")
df_pre = df_neighbors[df_neighbors['post'] == 0]
treated_pre_mean = df_pre[df_pre['treated'] == 1]['sales'].mean()
control_pre_mean = df_pre[df_pre['treated'] == 0]['sales'].mean()
pre_diff = treated_pre_mean - control_pre_mean

print(f"Pre-period: Treated avg = ${treated_pre_mean:.1f}, Control avg = ${control_pre_mean:.1f}")
print(f"Pre-period difference: ${pre_diff:.2f}")
print(f"If |pre-diff| > 20, parallel trends violated → causal estimate biased.\n")

if abs(pre_diff) < 20:
    print(f"✓ Parallel trends plausible. DiD ATE = ${ate_neighbor:.2f} likely causal.")
else:
    print(f"✗ Pre-period imbalance detected. Spillover may confound closure effect.")
