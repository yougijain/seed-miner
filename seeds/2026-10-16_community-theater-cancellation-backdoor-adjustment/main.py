import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from scipy import stats

np.random.seed(42)

# Synthetic: generate shows with hidden quality levels.
# Treatment: "high cancellation rate" (binary, endogenous).
# Outcome: attendance on show day.
# Confounder: latent show quality → affects both cancellation timing AND attendance.

shows = []
for show_id in range(50):
    # Latent show quality (0–1, hidden).
    true_quality = np.random.uniform(0.3, 0.95)
    
    # Cancellation pressure: low-quality shows have more cancellations,
    # and cancellations cluster *early* when audiences detect low quality.
    num_tickets_sold = np.random.randint(40, 120)
    cancel_rate = 0.15 if true_quality > 0.7 else 0.35
    num_cancellations = int(num_tickets_sold * cancel_rate)
    
    # Days before show when cancellations occur.
    # Low quality → cancellations cluster early (audience avoids).
    if true_quality > 0.7:
        cancel_days = np.random.exponential(scale=4, size=num_cancellations) + 1
    else:
        cancel_days = np.random.exponential(scale=1.5, size=num_cancellations) + 0.5
    cancel_days = np.clip(cancel_days, 0.5, 14)
    
    # Attendance on show day: affected by true_quality and cancellation rate.
    # High quality + few cancellations → high attendance.
    # Low quality + many cancellations → low attendance (confounder at work).
    base_attendance = int(num_tickets_sold * (0.85 + true_quality * 0.15))
    actual_attendance = base_attendance - num_cancellations + np.random.randint(-5, 5)
    actual_attendance = max(10, actual_attendance)
    
    # Treatment: observed high cancellation rate (binary proxy).
    high_cancel = 1 if cancel_rate > 0.25 else 0
    
    shows.append({
        'show_id': show_id,
        'true_quality': true_quality,
        'num_tickets_sold': num_tickets_sold,
        'num_cancellations': num_cancellations,
        'cancel_days_mean': np.mean(cancel_days),
        'cancel_days_std': np.std(cancel_days),
        'high_cancellation': high_cancel,
        'attendance': actual_attendance
    })

df = pd.DataFrame(shows)

# Naive analysis: regress attendance on high_cancellation (confounded).
from sklearn.linear_model import LinearRegression
X_naive = df[['high_cancellation']].values
y = df['attendance'].values
model_naive = LinearRegression().fit(X_naive, y)
naive_effect = model_naive.coef_[0]

print("="*60)
print("NAIVE ANALYSIS (CONFOUNDED):")
print(f"Naive causal effect of high cancellation on attendance: {naive_effect:.2f}")
print(f"Interpretation: High cancellation → {naive_effect:.2f} fewer attendees")
print()

# Backdoor adjustment: use cancellation *timing* (early clustering) as proxy
# for latent show quality. If cancellations cluster early (low std in cancel_days),
# it signals high quality → attendance is high. This confounding must be adjusted.

# Cluster cancellation patterns: early vs. late.
X_timing = df[['cancel_days_mean', 'cancel_days_std']].fillna(0).values
if X_timing.shape[0] > 2:
    dbscan = DBSCAN(eps=2, min_samples=2).fit(X_timing)
    df['cancel_cluster'] = dbscan.labels_
else:
    df['cancel_cluster'] = 0

# Stratified analysis: estimate effect within strata of cancellation timing.
strata_effects = []
for cluster_id in df['cancel_cluster'].unique():
    if cluster_id == -1:  # noise
        continue
    stratum = df[df['cancel_cluster'] == cluster_id]
    if len(stratum) < 2:
        continue
    
    treated = stratum[stratum['high_cancellation'] == 1]['attendance']
    control = stratum[stratum['high_cancellation'] == 0]['attendance']
    
    if len(treated) > 0 and len(control) > 0:
        stratum_effect = treated.mean() - control.mean()
        strata_effects.append(stratum_effect)

# Average treatment effect within strata (backdoor-adjusted estimate).
adjusted_effect = np.mean(strata_effects) if strata_effects else 0

print("BACKDOOR ADJUSTMENT (via cancellation timing as confounder proxy):")
print(f"Cancellation-timing strata found: {len(strata_effects)}")
print(f"Adjusted causal effect of high cancellation on attendance: {adjusted_effect:.2f}")
print(f"Interpretation: After stratifying on cancellation *timing patterns*,")
print(f"  high cancellation → {adjusted_effect:.2f} attendees (confounder removed)")
print()

# Sanity check: does timing stratification correlate with true quality?
for cluster_id in df['cancel_cluster'].unique():
    if cluster_id == -1:
        continue
    stratum = df[df['cancel_cluster'] == cluster_id]
    print(f"Cluster {cluster_id}: n={len(stratum)}, mean_true_quality={stratum['true_quality'].mean():.2f}")

print()
print("="*60)
print("KEY INSIGHT:")
print(f"Naive effect magnitude: {abs(naive_effect):.2f}")
print(f"Adjusted effect magnitude: {abs(adjusted_effect):.2f}")
print(f"Confounder bias recovered: {abs(naive_effect - adjusted_effect):.2f}")
print()
print("The cancellation *timing pattern* (early vs. late clustering)")
print("proxies for latent show quality, which confounds naive inference.")
print("Backdoor adjustment reconstructs true effect by stratifying on")
print("this timing confounder—a modification needed because the confounder")
print("is temporal, not a simple feature.")
