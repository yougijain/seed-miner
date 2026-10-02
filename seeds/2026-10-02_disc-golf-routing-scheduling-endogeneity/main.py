import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json

np.random.seed(42)

# Synthetic data: disc golfers with scores, routing choice (hidden), and true skill.
# The confounder: routing choice affects score variance independently of skill.
n_players = 60
players = []

for i in range(n_players):
    true_skill = np.random.uniform(35, 75)  # handicap-like score range
    routing = np.random.choice(['A', 'B'])  # unobserved routing choice
    
    # Routing affects variance: 'A' adds +8 score penalty (harder), 'B' adds +3
    routing_effect = 8 if routing == 'A' else 3
    observed_score = true_skill + routing_effect + np.random.normal(0, 2)
    
    players.append({
        'id': i,
        'observed_score': observed_score,
        'true_skill': true_skill,
        'routing': routing,
        'routing_effect': routing_effect
    })

df = pd.DataFrame(players)

# Define skill groups (binning by observed score)
def bin_pack_players(df, bin_width=10, max_per_group=4):
    """Greedy bin-packing: group players by score similarity."""
    df_sorted = df.sort_values('observed_score').reset_index(drop=True)
    groups = []
    current_group = []
    current_min_score = None
    
    for idx, row in df_sorted.iterrows():
        if current_min_score is None:
            current_min_score = row['observed_score']
        
        # Try to add to current group
        if (row['observed_score'] - current_min_score <= bin_width and 
            len(current_group) < max_per_group):
            current_group.append(row['id'])
        else:
            # Start new group
            if current_group:
                groups.append(current_group)
            current_group = [row['id']]
            current_min_score = row['observed_score']
    
    if current_group:
        groups.append(current_group)
    
    return groups

groups = bin_pack_players(df, bin_width=12, max_per_group=4)

# Measure constraint violations: for each group, measure routing imbalance
# A perfect packing ignores routing; routing imbalance signals the confounder.
violation_metrics = []
for g_idx, group_ids in enumerate(groups):
    group_data = df[df['id'].isin(group_ids)]
    routing_counts = group_data['routing'].value_counts()
    
    # Violation: routing imbalance (all A or all B)
    routing_imbalance = max(routing_counts.values) / len(group_data)
    
    # Score spread (packing quality)
    score_spread = group_data['observed_score'].max() - group_data['observed_score'].min()
    
    # True skill spread (what we should measure if routing weren't a confounder)
    true_skill_spread = group_data['true_skill'].max() - group_data['true_skill'].min()
    
    violation_metrics.append({
        'group_id': g_idx,
        'size': len(group_data),
        'routing_imbalance': routing_imbalance,
        'score_spread': score_spread,
        'true_skill_spread': true_skill_spread,
        'packing_quality': score_spread,  # how tightly packed by observed score
    })

metrics_df = pd.DataFrame(violation_metrics)

print("=" * 70)
print("DISC GOLF ROUND SCHEDULING: ROUTING ENDOGENEITY DETECTION")
print("=" * 70)
print(f"\nDataset: {n_players} players, {len(groups)} groups")
print(f"Hidden confounder: routing choice (A adds +8 score penalty, B adds +3)")
print(f"\nBin-packing groups (by observed score):")
print(metrics_df.to_string(index=False))

# Diagnosis: correlate routing imbalance with packing quality
routing_quality_corr = metrics_df['routing_imbalance'].corr(metrics_df['packing_quality'])
print(f"\nCorrelation(routing_imbalance, packing_quality): {routing_quality_corr:.3f}")

if abs(routing_quality_corr) > 0.5:
    print("  → DIAGNOSIS: Packing *accepts* routing-imbalanced groups to optimize score spread.")
    print("    This signals that routing is confounding the skill-grouping optimization.")
    print("    True skill spread is LESS correlated with packing:")
else:
    print("  → No strong routing-packing coupling detected.")

true_skill_quality_corr = metrics_df['true_skill_spread'].corr(metrics_df['packing_quality'])
print(f"\nCorrelation(true_skill_spread, packing_quality): {true_skill_quality_corr:.3f}")
print(f"  → Packing optimizes observed score, NOT true skill. Gap = {abs(routing_quality_corr - true_skill_quality_corr):.3f}")

if abs(routing_quality_corr - true_skill_quality_corr) > 0.3:
    print("\n*** CONCLUSION: Bin-packing FAILURE reveals routing endogeneity. ***")
    print("    Groups with high routing imbalance have low true-skill spread.")
    print("    The scheduler's constraint violations ARE the diagnostic signal.")
else:
    print("\n*** No significant endogeneity detected in this synthetic run. ***")

print("\n" + "=" * 70)
