import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import json

np.random.seed(42)

# Generate synthetic little-league schedule data
# Each row = one team's cumulative rest-day pattern over 15 game rounds
def generate_schedule_data(n_teams=12, n_rounds=15):
    """
    Generate rest-day cumulative sequences for teams.
    'Normal' teams have moderate, consistent rest patterns.
    'Anomalies' have either brutal bunching (2+ games in 3 days) or excessive rest (gaps >4 days).
    """
    teams = []
    labels = []
    
    # Normal schedules: rest days gradually increase, smooth curve
    for i in range(n_teams - 2):
        rest_days = np.sort(np.random.choice(range(1, n_rounds), size=n_rounds-1, replace=False))
        rest_sequence = np.cumsum(np.random.randint(1, 4, n_rounds))  # 1-3 days between games
        teams.append(rest_sequence)
        labels.append(0)
    
    # Anomaly 1: brutal bunching (2-3 games crammed into 2-3 days)
    bunched = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 25, 26, 27, 28])  # Big gap then bunched
    teams.append(bunched)
    labels.append(1)
    
    # Anomaly 2: excessive rest (5+ day gaps)
    rested = np.array([2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27])  # Multiple long gaps
    teams.append(rested)
    labels.append(1)
    
    return np.array(teams), np.array(labels)

def compute_fatigue_state_features(rest_sequence, window=3):
    """
    Convert a rest-day sequence into a feature vector representing fatigue state.
    For each game i, compute:
      - days_since_last_game: how long was the previous rest
      - rest_volatility: std dev of rest intervals in last `window` games
      - upcoming_density: average rest days in next `window` games
    """
    diffs = np.diff(rest_sequence)
    features = []
    
    for i in range(len(rest_sequence)):
        if i < window:
            window_diffs = diffs[:i]
        else:
            window_diffs = diffs[i-window:i]
        
        rest_volatility = np.std(window_diffs) if len(window_diffs) > 0 else 0
        
        if i + window < len(diffs):
            upcoming_rest = np.mean(diffs[i:i+window])
        else:
            upcoming_rest = np.mean(diffs[i:]) if i < len(diffs) else 0
        
        features.append([
            diffs[i] if i > 0 else diffs[0],
            rest_volatility,
            upcoming_rest
        ])
    
    return np.array(features)

def detect_anomalies_via_state_clustering(teams, rest_data, true_labels=None):
    """
    Core method: build fatigue state feature matrix, cluster trajectories,
    then flag teams whose state-space path deviates from the dominant cluster.
    """
    all_features = []
    feature_lengths = []
    
    for rest_seq in rest_data:
        feats = compute_fatigue_state_features(rest_seq)
        all_features.append(feats)
        feature_lengths.append(len(feats))
    
    # Flatten and standardize
    flat_features = np.vstack(all_features)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(flat_features)
    
    # Cluster the state trajectories using DBSCAN
    clusterer = DBSCAN(eps=0.8, min_samples=2)
    state_labels = clusterer.fit_predict(normalized)
    
    # Aggregate: label each team by which cluster state it spends most time in
    team_cluster_assignments = []
    idx = 0
    for length in feature_lengths:
        states_for_team = state_labels[idx:idx+length]
        # Count cluster membership; label team by dominant cluster or -1 if too dispersed
        cluster_counts = pd.Series(states_for_team).value_counts()
        if len(cluster_counts) > 0:
            dominant = cluster_counts.idxmax()
            scatter = 1 - (cluster_counts.iloc[0] / len(states_for_team))
        else:
            dominant = -1
            scatter = 1.0
        team_cluster_assignments.append((dominant, scatter))
        idx += length
    
    # Teams that frequently jump between clusters or land in noise cluster (-1) = anomalies
    predictions = np.array([1 if (assign[0] == -1 or assign[1] > 0.4) else 0 
                            for assign in team_cluster_assignments])
    
    if true_labels is not None:
        accuracy = np.mean(predictions == true_labels)
        return predictions, accuracy
    return predictions

# Main execution
teams, true_labels = generate_schedule_data()
rest_data = teams  # Each row is cumulative rest-day sequence

predictions, accuracy = detect_anomalies_via_state_clustering(teams, rest_data, true_labels)

print("=" * 60)
print("Little League Schedule Anomaly Detection via Fatigue State Clustering")
print("=" * 60)
print(f"\nDataset: {len(teams)} teams, 15-round schedule")
print(f"Ground truth anomalies: {np.sum(true_labels)}")
print(f"Predicted anomalies: {np.sum(predictions)}")
print(f"\nDetection accuracy: {accuracy:.2%}")
print(f"\nAnomalous team IDs: {np.where(predictions == 1)[0].tolist()}")
print(f"True anomalous team IDs: {np.where(true_labels == 1)[0].tolist()}")
print("\nKey insight: Anomaly detection on scheduling works by clustering")
print("cumulative rest-state *trajectories*, not point features. Teams whose")
print("fatigue states jump between clusters reveal structural imbalance.")
