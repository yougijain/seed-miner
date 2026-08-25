import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from itertools import combinations
import json

np.random.seed(42)

# Synthetic disc golf round data: 18 holes per round, par varies by hole
# par = [3,3,4,3,3,4,3,4,3, 4,3,3,4,3,4,3,3,4] (typical layout)
par_layout = [3,3,4,3,3,4,3,4,3, 4,3,3,4,3,4,3,3,4]

# Generate ~200 normal rounds + inject 5-8 anomalies
rounds = []
labels = []

# Normal rounds: player has consistent skill (~par -2 to par), fatigue increases on back nine
for i in range(190):
    skill_offset = np.random.normal(-1.5, 0.5)  # typically -2 to -1 on each hole
    fatigue_ramp = np.linspace(0, 1.5, 18)  # fatigue accumulates
    hole_scores = []
    for hole_idx, hole_par in enumerate(par_layout):
        noise = np.random.normal(0, 0.8)
        score = hole_par + skill_offset + fatigue_ramp[hole_idx] + noise
        hole_scores.append(max(1, int(np.round(score))))
    rounds.append(hole_scores)
    labels.append(0)  # normal

# Anomalous rounds: mental collapse after hole 10 (sudden +3 per hole) OR early fatigue
for i in range(5):
    skill_offset = np.random.normal(-1.5, 0.5)
    hole_scores = []
    collapse_point = np.random.randint(8, 12)  # mental break happens around hole 9-11
    for hole_idx, hole_par in enumerate(par_layout):
        if hole_idx < collapse_point:
            score = hole_par + skill_offset + np.random.normal(0, 0.8)
        else:
            # after collapse: +3 to +4 per hole
            score = hole_par + skill_offset + 3.5 + np.random.normal(0, 0.8)
        hole_scores.append(max(1, int(np.round(score))))
    rounds.append(hole_scores)
    labels.append(1)  # anomaly

df = pd.DataFrame(rounds, columns=[f'hole_{i+1}' for i in range(18)])
df['label'] = labels

# Feature engineering: compute *transition patterns* instead of raw scores
# For each round, compute (score - par) deltas to normalize across holes
df_normalized = df.copy()
for i, par in enumerate(par_layout):
    col = f'hole_{i+1}'
    df_normalized[col] = df[col] - par  # stroke differential from par

# Build transition-matrix features: co-occurrence of consecutive-hole deltas
# Quantize deltas into buckets: -2 or better, -1, 0, +1, +2+
quantize_delta = lambda x: 0 if x <= -2 else (1 if x == -1 else (2 if x == 0 else (3 if x == 1 else 4)))

transition_counts = []
for _, row in df_normalized.iterrows():
    deltas = [row[f'hole_{i+1}'] for i in range(18)]
    quantized = [quantize_delta(d) for d in deltas]
    # count transitions: (state_t, state_t+1) pairs
    trans_dict = {}
    for state_a, state_b in combinations(range(5), 2):
        trans_dict[(state_a, state_b)] = 0
    for j in range(17):
        pair = tuple(sorted([quantized[j], quantized[j+1]]))
        trans_dict[pair] = trans_dict.get(pair, 0) + 1
    transition_counts.append(list(trans_dict.values()))

trans_df = pd.DataFrame(transition_counts, columns=[f't_{i}' for i in range(len(trans_dict))])

# Normalize and detect anomalies
scaler = StandardScaler()
trans_scaled = scaler.fit_transform(trans_df)

# Isolation Forest on transition matrix features
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies_iso = iso_forest.fit_predict(trans_scaled)

# Also detect on raw score deltas for comparison
raw_deltas = df_normalized[[f'hole_{i+1}' for i in range(18)]].values
raw_scaled = scaler.fit_transform(raw_deltas)
anomalies_raw = iso_forest.fit_predict(raw_scaled)

# Evaluation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

print("\n=== Disc Golf Round Anomaly Detection ===")
print(f"Total rounds: {len(df)} (190 normal, 5 anomalies)\n")

print("Isolation Forest on TRANSITION MATRIX:")
tn_t, fp_t, fn_t, tp_t = confusion_matrix(labels, anomalies_iso == -1).ravel()
print(f"  TP={tp_t}, FP={fp_t}, TN={tn_t}, FN={fn_t}")
if tp_t + fp_t > 0:
    print(f"  Precision: {tp_t / (tp_t + fp_t):.2f}")
if tp_t + fn_t > 0:
    print(f"  Recall: {tp_t / (tp_t + fn_t):.2f}")

print("\nIsolation Forest on RAW SCORE DELTAS (baseline):")
tn_r, fp_r, fn_r, tp_r = confusion_matrix(labels, anomalies_raw == -1).ravel()
print(f"  TP={tp_r}, FP={fp_r}, TN={tn_r}, FN={fn_r}")
if tp_r + fp_r > 0:
    print(f"  Precision: {tp_r / (tp_r + fp_r):.2f}")
if tp_r + fn_r > 0:
    print(f"  Recall: {tp_r / (tp_r + fn_r):.2f}")

print("\nDetailed Results (transition-matrix approach):")
for idx, (anom, true_label) in enumerate(zip(anomalies_iso, labels)):
    if anom == -1 or true_label == 1:
        print(f"  Round {idx}: pred={'ANOMALY' if anom == -1 else 'normal':<7} actual={'ANOMALY' if true_label == 1 else 'normal':<7}")
