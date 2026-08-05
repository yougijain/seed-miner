import numpy as np
import pandas as pd
from collections import defaultdict
import json

# Synthetic amateur radio contest + ragchew log
np.random.seed(42)
n_qsos = 120

# Bimodal: contest QSOs ~3-5 min, ragchew ~15-30 min
contest_fraction = 0.75
n_contest = int(n_qsos * contest_fraction)
n_ragchew = n_qsos - n_contest

contest_durations = np.random.uniform(3, 5, n_contest)
ragchew_durations = np.random.uniform(15, 30, n_ragchew)
all_durations = np.concatenate([contest_durations, ragchew_durations])
np.random.shuffle(all_durations)

log = pd.DataFrame({
    'call_sign': [f'W5{i%100:02d}' for i in range(n_qsos)],
    'qso_duration_min': all_durations,
    'frequency_mhz': np.random.choice([7.03, 14.02, 21.07, 28.03], n_qsos),
    'time_utc': pd.date_range('2024-01-01 12:00', periods=n_qsos, freq='2min')
})

log = log.sort_values('time_utc').reset_index(drop=True)

# Baseline: assume 15-minute contest slots; pack greedily into bins
def greedy_bin_pack(durations, bin_capacity=15):
    bins = []
    current_bin = []
    current_load = 0
    
    for dur in durations:
        if current_load + dur <= bin_capacity:
            current_bin.append(dur)
            current_load += dur
        else:
            if current_bin:
                bins.append(current_bin)
            current_bin = [dur]
            current_load = dur
    
    if current_bin:
        bins.append(current_bin)
    
    return bins

log_sorted = log.sort_values('qso_duration_min').reset_index(drop=True)
bins_by_duration = greedy_bin_pack(log_sorted['qso_duration_min'].values)

# Constraint sensitivity: fit a mixture model and compare to bin structure
from sklearn.mixture import GaussianMixture

X = log_sorted['qso_duration_min'].values.reshape(-1, 1)
gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
log_sorted['mode_predicted'] = gmm.predict(X)

# Cross-tab: do bins from packing align with modes from GMM?
log_sorted['bin_id'] = -1
for bin_idx, bin_qsos in enumerate(bins_by_duration):
    mask = log_sorted['qso_duration_min'].isin(bin_qsos)
    log_sorted.loc[mask & (log_sorted['bin_id'] == -1), 'bin_id'] = bin_idx

results = {
    'n_qsos': n_qsos,
    'n_bins_packed': len(bins_by_duration),
    'mean_bin_load': np.mean([sum(b) for b in bins_by_duration]),
    'predicted_mode_counts': log_sorted['mode_predicted'].value_counts().to_dict(),
    'mean_duration_by_predicted_mode': log_sorted.groupby('mode_predicted')['qso_duration_min'].mean().to_dict(),
    'bin_purity_vs_mode': {
        'mean_mode_0_in_bin': float(log_sorted[log_sorted['mode_predicted']==0]['qso_duration_min'].mean()),
        'mean_mode_1_in_bin': float(log_sorted[log_sorted['mode_predicted']==1]['qso_duration_min'].mean()),
    }
}

print(json.dumps(results, indent=2))
log_sorted.to_csv('qso_log_with_modes_and_bins.csv', index=False)
print("\nLog saved to qso_log_with_modes_and_bins.csv")
