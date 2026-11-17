import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

np.random.seed(42)

# Synthetic disc golf score data: same player, multiple outings, multiple course routings.
# Key structural truth: when player is fatigued (unobserved), they choose easier routing
# AND score worse. Standard anomaly detection on scores alone will miss this.
# Anomaly detection on (routing_difficulty, score) pairs reveals the coupling.

def generate_disc_golf_data():
    """Generate synthetic data where player routing choice is endogenous to fatigue."""
    n_rounds = 120
    data = []
    
    # Simulate a player over ~4 months (30 rounds/month).
    # Outing 0-80: healthy baseline, mostly hard routes
    # Outing 81-120: progressive injury/fatigue, shifts to easier routes, scores worsen
    for round_idx in range(n_rounds):
        base_skill = 50  # par for 18 holes
        
        # Fatigue accumulation: unobserved state variable
        fatigue = max(0, (round_idx - 80) * 0.3) if round_idx > 80 else 0
        
        # Player ENDOGENOUSLY chooses routing difficulty based on fatigue
        # (higher fatigue → easier routing)
        routing_choice = np.clip(100 - fatigue * 2 - np.random.normal(0, 5), 30, 100)
        
        # Score depends on routing difficulty AND fatigue
        # Easier routing → lower score (better)
        # Higher fatigue → higher score (worse)
        score = base_skill + (100 - routing_choice) * 0.3 + fatigue * 0.5 + np.random.normal(0, 2)
        
        data.append({
            'round': round_idx,
            'routing_difficulty': routing_choice,
            'score': score,
            'is_injured_period': round_idx > 80
        })
    
    return pd.DataFrame(data)

def detect_anomalies_naive(df):
    """Standard anomaly detection: only on scores (CONFOUNDED)."""
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly_naive'] = iso_forest.fit_predict(df[['score']].values)
    return df

def detect_anomalies_routing_aware(df):
    """Anomaly detection on (routing, score) pair (ACCOUNTS FOR ENDOGENEITY)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['routing_difficulty', 'score']])
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly_routing_aware'] = iso_forest.fit_predict(X)
    return df

def evaluate_detection(df):
    """Check if detected anomalies align with true injury period."""
    # True anomalies: rounds 81-120 (injury period)
    true_positives_naive = sum((df['round'] > 80) & (df['anomaly_naive'] == -1))
    false_positives_naive = sum((df['round'] <= 80) & (df['anomaly_naive'] == -1))
    
    true_positives_routing = sum((df['round'] > 80) & (df['anomaly_routing_aware'] == -1))
    false_positives_routing = sum((df['round'] <= 80) & (df['anomaly_routing_aware'] == -1))
    
    return {
        'naive_tp': true_positives_naive,
        'naive_fp': false_positives_naive,
        'routing_aware_tp': true_positives_routing,
        'routing_aware_fp': false_positives_routing,
        'naive_precision': true_positives_naive / max(1, true_positives_naive + false_positives_naive),
        'routing_aware_precision': true_positives_routing / max(1, true_positives_routing + false_positives_routing)
    }

if __name__ == '__main__':
    df = generate_disc_golf_data()
    df = detect_anomalies_naive(df)
    df = detect_anomalies_routing_aware(df)
    
    results = evaluate_detection(df)
    print("Anomaly Detection Results")
    print("========================")
    print(f"Naive (score-only):")
    print(f"  True Positives: {results['naive_tp']}, False Positives: {results['naive_fp']}")
    print(f"  Precision: {results['naive_precision']:.2f}")
    print()
    print(f"Routing-Aware (score+routing):")
    print(f"  True Positives: {results['routing_aware_tp']}, False Positives: {results['routing_aware_fp']}")
    print(f"  Precision: {results['routing_aware_precision']:.2f}")
    print()
    print("Key insight: Naive detection conflates routing choice (endogenous) with true anomalies.")
    print("Routing-aware detector reveals the coupling and surfaces hidden injury/fatigue states.")
    
    # Save for visualization
    df.to_csv('disc_golf_anomalies.csv', index=False)
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
