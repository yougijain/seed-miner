import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from itertools import combinations
import json

np.random.seed(42)

# Synthetic cafe checkout data: hourly snapshots of game availability + successful table packs
# Structure: for each hour, which games were checked out, and whether the cafe could pack
# remaining available games into open table slots

games = [
    'Catan', 'Ticket_to_Ride', 'Splendor', 'Azul', 'Codenames',
    'Power_Grid', 'Food_Chain_Magnate', 'Agricola', 'Puerto_Rico', 'Heavy_Hitter'
]
n_tables = 6
n_hours = 168  # 1 week

# Generate synthetic checkouts with latent contention structure
data = []
for hour in range(n_hours):
    day = hour // 24
    hour_of_day = hour % 24
    is_peak = 16 <= hour_of_day <= 22  # evening peak
    
    # Base checkout probability increases for popular games during peak
    base_prob = 0.5 if is_peak else 0.25
    
    # HIDDEN CONTENTION: Heavy_Hitter and Power_Grid create bottleneck when both available
    # during peak—they consume long table slots and block other games from packing
    checkouts = {}
    for game in games:
        if game in ['Heavy_Hitter', 'Power_Grid']:
            p = base_prob * 1.4 if is_peak else base_prob * 0.6
        else:
            p = base_prob + (0.1 if is_peak else 0)
        checkouts[game] = 1 if np.random.random() < p else 0
    
    # Compute bin-packing feasibility: can we fit all available games into tables?
    available_games = sum(1 for g in games if checkouts[g] == 0)
    # Simple feasibility: assume each table holds 1 game, need N tables for N available games
    # But two "heavy" games reduce effective capacity
    heavy_penalty = (checkouts['Heavy_Hitter'] + checkouts['Power_Grid']) * 0.5
    effective_tables = n_tables - heavy_penalty
    feasible = 1 if available_games <= effective_tables else 0
    
    row = {'hour': hour, 'day': day, 'is_peak': int(is_peak), 'feasible': feasible}
    row.update({f'checkout_{g}': checkouts[g] for g in games})
    data.append(row)

df = pd.DataFrame(data)

# Feature engineering: for each game, compute its "contention signature"
# = (checkout count during peak) × (packing failure rate when that game is checked out)
features = []
for game in games:
    peak_df = df[df['is_peak'] == 1]
    checkout_rate = peak_df[f'checkout_{game}'].mean()
    failure_when_checked = peak_df[peak_df[f'checkout_{game}'] == 1]['feasible'].mean()
    failure_rate = 1.0 - failure_when_checked
    contention_score = checkout_rate * failure_rate
    features.append(contention_score)

# Anomaly detection: IsoForest on contention scores
# Non-obvious angle: we're not detecting rare checkout patterns;
# we're detecting games whose *presence* correlates with packing failure
X = np.array(features).reshape(-1, 1)
iso = IsolationForest(contamination=0.2, random_state=42)
anomaly_labels = iso.fit_predict(X)
anomaly_scores = iso.score_samples(X)

results = pd.DataFrame({
    'game': games,
    'contention_score': features,
    'anomaly': anomaly_labels,
    'anomaly_score': anomaly_scores
}).sort_values('anomaly_score')

print("=== Board Game Checkout Contention Detection ===")
print("\nGames ranked by contention anomaly (most anomalous = highest contention risk):")
print(results.to_string(index=False))

anomalous_games = results[results['anomaly'] == -1]['game'].tolist()
print(f"\n[ANOMALY] Flagged as contention bottlenecks: {anomalous_games}")
print(f"\nInterpretation: These games, when checked out during peak hours,")
print(f"significantly increase the likelihood that the cafe cannot pack available")
print(f"games into open table slots. They create scheduling bottlenecks.")

# Validation: confirm flagged games correlate with packing failure
for game in anomalous_games:
    peak_checked = df[(df['is_peak'] == 1) & (df[f'checkout_{game}'] == 1)]
    failure_rate = 1.0 - peak_checked['feasible'].mean()
    print(f"  {game}: packing failure rate when checked out = {failure_rate:.2%}")