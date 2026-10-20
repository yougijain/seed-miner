import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_qso_log():
    """Generate synthetic contest log with hidden mode switches.
    
    Operator alternates between:
    - CONTEST mode: rapid-fire QSOs, short intervals (5-20s), high frequency density
    - RAGCHEW mode: deep conversations, long intervals (2-5 min), lower density
    
    No explicit mode label; only contact timestamps.
    """
    contacts = []
    time = 0  # seconds
    mode = 'CONTEST'
    mode_change_points = [300, 900, 1500, 2100]  # Force mode switches at these times
    
    for i in range(200):
        # Check if we should switch modes
        if time in mode_change_points:
            mode = 'RAGCHEW' if mode == 'CONTEST' else 'CONTEST'
        
        if mode == 'CONTEST':
            interval = np.random.randint(5, 20)
        else:
            interval = np.random.randint(120, 300)
        
        time += interval
        contacts.append({'qso_index': i, 'time_seconds': time, 'mode_true': mode})
    
    return pd.DataFrame(contacts)

def compute_temporal_features(df, window=10):
    """Extract inter-contact intervals and rolling statistics."""
    df['interval'] = df['time_seconds'].diff().fillna(0)
    
    features = []
    for i in range(window, len(df)):
        window_intervals = df.iloc[i-window:i]['interval'].values
        features.append({
            'idx': i,
            'mean_interval': window_intervals.mean(),
            'std_interval': window_intervals.std() + 1e-6,
            'max_interval': window_intervals.max(),
            'min_interval': window_intervals.min(),
            'cv_interval': (window_intervals.std() + 1e-6) / (window_intervals.mean() + 1e-6),
        })
    
    return pd.DataFrame(features)

def segment_and_detect_breaks(features_df, k=3):
    """Cluster temporal features and measure segmentation stability.
    
    Key insight: When operator switches modes, the temporal distribution changes.
    Clustering will fail to produce stable, meaningful segments when stationarity
    assumptions break. We detect this by measuring 'cluster fragmentation' — 
    how much the assigned clusters diverge from a smooth progression.
    """
    X = features_df[['mean_interval', 'std_interval', 'max_interval', 'cv_interval']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Measure segmentation coherence: adjacent points should stay in same cluster
    # when mode is stable; they fragment when mode switches.
    transitions = np.sum(np.diff(labels) != 0)
    inertia = kmeans.inertia_
    silhouette_score = compute_silhouette(X_scaled, labels)
    
    return {
        'labels': labels,
        'cluster_transitions': transitions,
        'inertia': inertia,
        'silhouette': silhouette_score,
        'X_scaled': X_scaled,
        'kmeans': kmeans
    }

def compute_silhouette(X, labels):
    """Simple silhouette score: measure of cluster cohesion."""
    n = len(X)
    silhouette_vals = []
    for i in range(n):
        same_cluster = np.where(labels == labels[i])[0]
        other_clusters = np.where(labels != labels[i])[0]
        
        if len(same_cluster) > 1:
            a_i = np.mean([np.linalg.norm(X[i] - X[j]) for j in same_cluster if j != i])
        else:
            a_i = 0
        
        if len(other_clusters) > 0:
            b_i = np.min([np.mean([np.linalg.norm(X[i] - X[j]) for j in np.where(labels == c)[0]]) 
                         for c in np.unique(labels[other_clusters])])
        else:
            b_i = 0
        
        s_i = (b_i - a_i) / (max(a_i, b_i) + 1e-6)
        silhouette_vals.append(s_i)
    
    return np.mean(silhouette_vals)

def detect_mode_switches(df):
    """Main analysis: detect hidden mode switches via clustering collapse."""
    features = compute_temporal_features(df, window=10)
    result = segment_and_detect_breaks(features, k=3)
    
    # Mode switch hypothesis: high cluster transitions indicate instability
    # caused by unobserved mode changes breaking temporal stationarity.
    
    print("\n=== QSO Mode-Switching Detection via Segmentation Collapse ===")
    print(f"Total contacts: {len(df)}")
    print(f"Temporal window: 10-contact rolling features")
    print(f"\nClustering Results (k=3):")
    print(f"  Cluster transitions (mode-switch indicator): {result['cluster_transitions']}")
    print(f"  Inertia (lower = tighter clusters): {result['inertia']:.2f}")
    print(f"  Silhouette score (coherence, -1 to 1): {result['silhouette']:.3f}")
    print(f"\nInterpretation:")
    print(f"  High transitions (>{len(features)//8}) suggest operator mode-switching")
    print(f"  breaks temporal stationarity, fragmenting clustering.")
    print(f"  Actual transitions observed: {result['cluster_transitions']}")
    print(f"  Threshold for 'likely mode switch': >{len(features)//6} transitions")
    
    if result['cluster_transitions'] > len(features) // 6:
        print(f"\n✓ DIAGNOSIS: Likely hidden mode-switching detected.")
        print(f"  Operator behavior is non-stationary; clustering fails to segment cleanly.")
    else:
        print(f"\n✗ DIAGNOSIS: Stationary operator behavior (no mode switches detected).")
    
    return result, features

if __name__ == '__main__':
    df = generate_qso_log()
    result, features = detect_mode_switches(df)
