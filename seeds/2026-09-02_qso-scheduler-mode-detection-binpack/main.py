import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import norm

np.random.seed(42)

# Generate synthetic contest QSO log with hidden mode-switching.
# Operators switch between contest-mode (fast, ~30s) and ragchew (slow, ~5min)
# based on propagation conditions (unobserved).
n_qsos = 300
mode_prob = 0.7  # 70% contest-mode, 30% ragchew
modes = np.random.binomial(1, mode_prob, n_qsos)

# QSO durations (seconds) with mode-dependent distribution
durations = np.where(
    modes == 1,
    np.random.normal(35, 8, n_qsos),  # contest-mode
    np.random.normal(300, 60, n_qsos)  # ragchew-mode
)
durations = np.clip(durations, 10, 600)  # realistic bounds

log = pd.DataFrame({
    'qso_id': range(n_qsos),
    'duration_sec': durations,
    'true_mode': ['contest' if m == 1 else 'ragchew' for m in modes]
})

print("\n=== QSO Log (first 10) ===")
print(log.head(10))
print(f"Duration stats: mean={log['duration_sec'].mean():.1f}s, median={log['duration_sec'].median():.1f}s")
print(f"True mode split: {(modes == 1).sum()} contest, {(modes == 0).sum()} ragchew")

# Bin-packing: fit QSOs into contest time slots (15-minute windows).
# Strategy: first-fit-decreasing. Track which QSOs violate (don't fit cleanly).
slot_capacity_sec = 900  # 15 minutes
log_sorted = log.sort_values('duration_sec', ascending=False).reset_index(drop=True)

slots = []
current_slot = []
current_used = 0
violation_flags = []

for idx, row in log_sorted.iterrows():
    qso_dur = row['duration_sec']
    if current_used + qso_dur <= slot_capacity_sec:
        current_slot.append(idx)
        current_used += qso_dur
        violation_flags.append(0)
    else:
        # Constraint violation: QSO doesn't fit; start new slot.
        if current_slot:
            slots.append(current_slot)
        current_slot = [idx]
        current_used = qso_dur
        violation_flags.append(1)  # Flagged as constraint-violating placement

if current_slot:
    slots.append(current_slot)

log_sorted['constraint_violation'] = violation_flags
log_sorted['original_idx'] = log_sorted.index

print(f"\nBin-packing: {len(slots)} slots needed, {violation_flags.count(1)} constraint violations.")

# Hypothesis: constraint violations cluster around latent mode-switch points.
# Use DBSCAN on (duration, violation_flag) to detect structure.
X = log_sorted[['duration_sec', 'constraint_violation']].values
db = DBSCAN(eps=80, min_samples=5).fit(X)
clusters = db.labels_

log_sorted['detected_cluster'] = clusters

print(f"\nDBSCAN detected {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters (+ noise).")

# Evaluate: do clusters separate true modes?
for cluster_id in sorted(set(clusters)):
    mask = clusters == cluster_id
    subset = log_sorted[mask]
    if cluster_id == -1:
        label = "Noise"
    else:
        label = f"Cluster {cluster_id}"
    true_mode_dist = subset['true_mode'].value_counts()
    mean_dur = subset['duration_sec'].mean()
    viol_rate = subset['constraint_violation'].mean()
    print(f"  {label}: n={mask.sum()}, mean_dur={mean_dur:.1f}s, violation_rate={viol_rate:.2f}, true_modes={dict(true_mode_dist)}")

# Purity: what fraction of largest true_mode in each cluster?
accuracy_parts = []
for cluster_id in set(clusters):
    if cluster_id == -1:
        continue
    mask = clusters == cluster_id
    subset = log_sorted[mask]
    mode_counts = subset['true_mode'].value_counts()
    if len(mode_counts) > 0:
        purity = mode_counts.max() / mode_counts.sum()
        accuracy_parts.append(purity)

if accuracy_parts:
    avg_purity = np.mean(accuracy_parts)
    print(f"\nAverage cluster purity (homogeneity): {avg_purity:.2f}")
    print("(1.0 = perfect mode separation; constraint violations reveal latent structure)")
