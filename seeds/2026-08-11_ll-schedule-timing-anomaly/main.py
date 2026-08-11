import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic little-league schedule: 12 teams, 22 games each (round-robin + some)
# Feature: [avg_start_hour, opponent_strength_avg, weekend_game_pct, primetime_slot_pct]
n_teams = 12
n_games_per_team = 22

teams = [f'Team_{i}' for i in range(n_teams)]

# Generate "fair" schedules: most teams get balanced features
schedules = []
for i in range(n_teams):
    avg_hour = np.random.normal(18.5, 0.8)  # games typically 6-7pm
    opp_strength = np.random.normal(0.5, 0.1)  # opponent ELO proxy
    weekend_pct = np.random.normal(0.45, 0.08)
    primetime_pct = np.random.normal(0.40, 0.10)
    schedules.append([avg_hour, opp_strength, weekend_pct, primetime_pct])

# Inject 2 unfair schedules: one gets all early slots + weak opponents, one gets late + strong
schedules[0] = [14.2, 0.35, 0.15, 0.05]  # Team_0: disadvantaged (early, weak, few weekend)
schedules[1] = [20.1, 0.68, 0.72, 0.78]  # Team_1: privileged (late, strong, many weekend)

df = pd.DataFrame(schedules, columns=['avg_start_hour', 'avg_opponent_strength', 'weekend_game_pct', 'primetime_pct'])
df['team'] = teams

# Standardize
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['avg_start_hour', 'avg_opponent_strength', 'weekend_game_pct', 'primetime_pct']])

# Isolation Forest: contamination=0.15 (expect ~2 anomalies in 12 teams)
ifo = IsolationForest(contamination=0.15, random_state=42)
anomalies = ifo.fit_predict(df_scaled)
scores = ifo.score_samples(df_scaled)

df['anomaly'] = anomalies
df['anomaly_score'] = scores
df_sorted = df.sort_values('anomaly_score')

print("\n=== Little-League Schedule Equity: Isolation Forest Results ===")
print(f"\nSchedules flagged as anomalous (inequitable):")
print(df_sorted[df_sorted['anomaly'] == -1][['team', 'avg_start_hour', 'avg_opponent_strength', 'weekend_game_pct', 'primetime_pct', 'anomaly_score']])
print(f"\nTop 5 most anomalous teams (lowest anomaly_score):")
print(df_sorted.head(5)[['team', 'avg_start_hour', 'avg_opponent_strength', 'weekend_game_pct', 'primetime_pct', 'anomaly_score']])
print(f"\nInterpretation:")
print(f"  - Team_0 (early slots, weak opponents, few weekend games) = disadvantaged anomaly")
print(f"  - Team_1 (late slots, strong opponents, many weekend games) = privileged anomaly")
print(f"  - Standard fairness audits count slot types; this method detects *multivariate inequity*.")
