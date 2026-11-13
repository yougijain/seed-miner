import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

np.random.seed(42)

# Synthetic data: 12 trail segments, 3 invisible maintenance crews.
# Each crew has a characteristic closure pattern (timing clustering).
N_SEGMENTS = 12
N_CLOSURES_PER_SEGMENT = 20
N_DAYS = 365

# True (hidden) crew assignments: segments 0-3 to crew A, 4-7 to crew B, 8-11 to crew C.
true_crews = [0]*4 + [1]*4 + [2]*4

# Generate closure dates. Within crew: clustered. Between crew: independent.
closures_by_segment = {}
for seg_id in range(N_SEGMENTS):
    crew_id = true_crews[seg_id]
    if crew_id == 0:
        # Crew A works days 50-150
        base_days = np.random.normal(100, 20, N_CLOSURES_PER_SEGMENT)
    elif crew_id == 1:
        # Crew B works days 200-300
        base_days = np.random.normal(250, 20, N_CLOSURES_PER_SEGMENT)
    else:
        # Crew C works days 320-365
        base_days = np.random.normal(340, 20, N_CLOSURES_PER_SEGMENT)
    
    # Add small independent noise to each segment
    closure_days = np.clip(base_days + np.random.normal(0, 5, N_CLOSURES_PER_SEGMENT), 0, N_DAYS-1)
    closures_by_segment[seg_id] = sorted(closure_days.astype(int))

# Compute pairwise closure-timing correlation (backdoor confounder signal).
def closure_timing_overlap(days1, days2, window=30):
    """Fraction of days1 that have a closure within window days of any day in days2."""
    if not days1 or not days2:
        return 0
    overlap = sum(1 for d1 in days1 if any(abs(d1 - d2) < window for d2 in days2))
    return overlap / len(days1)

corr_matrix = np.zeros((N_SEGMENTS, N_SEGMENTS))
for i in range(N_SEGMENTS):
    for j in range(N_SEGMENTS):
        corr_matrix[i, j] = closure_timing_overlap(closures_by_segment[i], closures_by_segment[j])

# Hierarchical clustering on closure-timing correlation (distance = 1 - correlation).
distance_matrix = 1 - corr_matrix
np.fill_diagonal(distance_matrix, 0)

clustering = AgglomerativeClustering(
    n_clusters=3,
    linkage='average',
    metric='precomputed'
)
inferred_crews = clustering.fit_predict(distance_matrix)

# Compare inferred to true crew assignments.
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(true_crews, inferred_crews)
nmi = normalized_mutual_info_score(true_crews, inferred_crews)

results_df = pd.DataFrame({
    'segment': range(N_SEGMENTS),
    'true_crew': true_crews,
    'inferred_crew': inferred_crews,
    'match': [true_crews[i] == inferred_crews[i] for i in range(N_SEGMENTS)]
})

print("=" * 60)
print("CAUSAL INFERENCE ON TRAIL CLOSURE TEMPORAL CLUSTERING")
print("Inferring Hidden Maintenance Crew Structure")
print("=" * 60)
print(f"\nAdjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Info: {nmi:.3f}")
print(f"\nSegment Assignments (True vs Inferred):")
print(results_df)
print(f"\nAccuracy: {results_df['match'].sum() / len(results_df):.2%}")
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("""
This inverts standard causal inference:
- Normal: 'Does treatment (crew assignment) cause outcome (closures)?'
- Inverted: 'Does observed outcome clustering (closure timing) reveal unobserved treatment (crew identity)?'

We used CLOSURE TIMING CO-OCCURRENCE as a noisy signal of a backdoor confounder
(crew identity). Hierarchical clustering on closure-timing correlation recovers
the hidden crew structure WITHOUT observing crew assignments directly.

Limitation: Works only when closures are strictly clustered by crew.
When crews have flexible scheduling or overlapping work zones, the signal weakens.
""")