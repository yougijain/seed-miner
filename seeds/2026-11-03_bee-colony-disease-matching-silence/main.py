import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

# Synthetic dataset: 12 colonies, 20 inspection dates, 5 health metrics
np.random.seed(42)
colonies = [f"C{i:02d}" for i in range(1, 13)]
dates = pd.date_range('2024-01-01', periods=20, freq='7D')
metrics = ['pollen_stores', 'brood_pattern', 'flight_activity', 'mite_signs', 'disease_spores']

# Ground truth: 3 hidden disease clusters (C01-C04, C05-C08, C09-C12)
# Colonies in same cluster co-deteriorate and go silent together on disease metrics
data = []
for col in colonies:
    col_idx = int(col[1:]) - 1
    cluster = col_idx // 4  # 0, 1, or 2
    
    for date_idx, date in enumerate(dates):
        row = {'colony': col, 'date': date}
        
        # Metric 1-3: normal, reported every inspection
        row['pollen_stores'] = np.random.choice([None, 'good', 'fair'], p=[0.05, 0.5, 0.45])
        row['brood_pattern'] = np.random.choice([None, 'solid', 'spotty'], p=[0.05, 0.6, 0.35])
        row['flight_activity'] = np.random.choice([None, 'high', 'low'], p=[0.05, 0.55, 0.4])
        
        # Metrics 4-5: disease indicators; colonies in same cluster go silent together
        # Around date 10-15, each cluster's disease metrics become None (silence = disease hiding/collapse)
        silence_start = 10 + cluster * 2
        silence_end = silence_start + 4
        
        if silence_start <= date_idx < silence_end:
            row['mite_signs'] = None
            row['disease_spores'] = None
        else:
            row['mite_signs'] = np.random.choice([None, 'absent', 'present'], p=[0.1, 0.7, 0.2])
            row['disease_spores'] = np.random.choice([None, 'absent', 'present'], p=[0.1, 0.75, 0.15])
        
        data.append(row)

df = pd.DataFrame(data)

# Build silence co-occurrence matrix: for each colony pair, count inspections
# where both jointly went silent (None) on disease metrics
def silence_similarity(c1, c2, df):
    """Count dates where c1 and c2 are both silent on both disease metrics."""
    c1_data = df[df['colony'] == c1]
    c2_data = df[df['colony'] == c2]
    
    merged = c1_data[['date', 'mite_signs', 'disease_spores']].rename(
        columns={'mite_signs': 'm1', 'disease_spores': 'd1'}
    ).merge(
        c2_data[['date', 'mite_signs', 'disease_spores']].rename(
            columns={'mite_signs': 'm2', 'disease_spores': 'd2'}
        ),
        on='date'
    )
    
    # Both colonies silent = both missing disease metrics
    co_silent = ((merged['m1'].isna()) & (merged['m2'].isna()) &
                 (merged['d1'].isna()) & (merged['d2'].isna())).sum()
    return co_silent

# Compute similarity matrix
n_colonies = len(colonies)
similarity = np.zeros((n_colonies, n_colonies))
for i, c1 in enumerate(colonies):
    for j, c2 in enumerate(colonies):
        if i != j:
            similarity[i, j] = silence_similarity(c1, c2, df)
        else:
            similarity[i, j] = 0

# Convert to cost (higher co-silence = lower cost = stronger link)
max_sim = similarity.max() + 1
cost = max_sim - similarity

# Bipartite matching: treat colonies as two partitions (left, right)
# Minimize cost of matching, which prefers colonies with high co-silence
# Partition 1: C01-C06, Partition 2: C07-C12
left_idx = list(range(6))
right_idx = list(range(6, 12))

cost_submatrix = cost[np.ix_(left_idx, right_idx)]
row_ind, col_ind = linear_sum_assignment(cost_submatrix)

matches = [(colonies[left_idx[i]], colonies[right_idx[col_ind[i]]]) 
           for i in range(len(row_ind))]

print("=== Bipartite Matching on Temporal Co-Silence ===")
print(f"Colony pairs matched (strong disease-propagation risk inference):")
for left, right in matches:
    co_silence = silence_similarity(left, right, df)
    print(f"  {left} <-> {right}: co-silent on {co_silence} inspections")

print(f"\nTotal cost of matching: {cost_submatrix[row_ind, col_ind].sum():.1f}")
print(f"(Lower cost = higher co-silence = stronger inferred disease link)")

# Sanity check: ground truth clusters
print(f"\n=== Ground Truth Disease Clusters ===")
for c in range(3):
    cluster_colonies = [colonies[i] for i in range(12) if i // 4 == c]
    print(f"Cluster {c}: {cluster_colonies}")
