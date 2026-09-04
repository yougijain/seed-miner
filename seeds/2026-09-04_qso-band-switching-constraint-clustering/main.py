import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import json

np.random.seed(42)

# Synthetic contest log: operator_id, qso_start_sec, qso_duration_sec, band, signal_quality
# Some operators switch bands frequently (latent switching behavior), others stay put.
qsos = []
op_ids = [f"W5{i:03d}" for i in range(1, 11)]

for op_id in op_ids:
    # Half the operators: band-loyal (stay on one band for whole contest).
    # Half: band-switchers (change bands every 10–15 QSOs).
    is_switcher = (int(op_id[-3:]) % 2 == 0)
    current_band = np.random.choice(["40m", "20m", "15m", "10m"])
    start_sec = 0
    
    n_qsos = np.random.randint(40, 80)
    for i in range(n_qsos):
        if is_switcher and i > 0 and i % np.random.randint(8, 16) == 0:
            current_band = np.random.choice(["40m", "20m", "15m", "10m"])
        duration = np.random.randint(20, 90)
        qsos.append({
            "operator": op_id,
            "start_sec": start_sec,
            "duration_sec": duration,
            "band": current_band,
            "snr": np.random.randint(5, 30)
        })
        start_sec += duration + np.random.randint(2, 8)

df = pd.DataFrame(qsos)

def bin_pack_greedy(qsos_seq, max_bin_duration=600):
    """Pack QSOs into time bins (simulating band-lock intervals).
    Return (n_bins_used, n_constraint_violations).
    Constraint: all QSOs in one bin should be same band (or penalize switches).
    """
    bins = []
    violations = 0
    for qso in qsos_seq:
        placed = False
        for bin_idx, bin_list in enumerate(bins):
            bin_time = sum(q["duration_sec"] for q in bin_list)
            if bin_time + qso["duration_sec"] <= max_bin_duration:
                # Check band constraint
                bin_bands = set(q["band"] for q in bin_list)
                if qso["band"] not in bin_bands and len(bin_bands) > 0:
                    violations += 1  # soft constraint: penalize band change
                bins[bin_idx].append(qso)
                placed = True
                break
        if not placed:
            bins.append([qso])
    return len(bins), violations

# Per-operator packing analysis
results = []
for op_id in op_ids:
    op_log = df[df["operator"] == op_id].sort_values("start_sec").to_dict("records")
    n_bins, violations = bin_pack_greedy(op_log)
    n_qsos = len(op_log)
    violation_rate = violations / n_qsos if n_qsos > 0 else 0
    results.append({
        "operator": op_id,
        "n_qsos": n_qsos,
        "n_bins": n_bins,
        "constraint_violations": violations,
        "violation_rate": violation_rate
    })

res_df = pd.DataFrame(results)

# Cluster operators by violation profile (violation_rate, n_bins / n_qsos)
X = res_df[["violation_rate", "n_bins"]].values
clustering = DBSCAN(eps=0.15, min_samples=2).fit(X)
res_df["cluster"] = clustering.labels_

print("=== Per-Operator Bin-Packing Analysis ===")
print(res_df.to_string())
print()

# Cluster -1 = high violation anomalies (band-switchers), clusters >= 0 = low violation (band-loyal).
anomalies = res_df[res_df["cluster"] == -1]
print(f"\n=== Detected Band-Switchers (Constraint Anomalies) ===")
print(f"Count: {len(anomalies)}")
for _, row in anomalies.iterrows():
    print(f"  {row['operator']}: violation_rate={row['violation_rate']:.2f}, bins={row['n_bins']}, qsos={row['n_qsos']}")

print(f"\n=== Hypothesis Check ===")
for op_id in op_ids:
    is_switcher = (int(op_id[-3:]) % 2 == 0)
    pred = (res_df[res_df["operator"] == op_id]["cluster"].values[0] == -1)
    match = "✓" if is_switcher == pred else "✗"
    print(f"{match} {op_id}: true_switcher={is_switcher}, predicted_anomaly={pred}")
