import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import json

np.random.seed(42)

def generate_seating_data(n_shows=50, seats_per_row=12, n_rows=8):
    """
    Generate synthetic seating occupancy across multiple shows.
    Healthy shows: random occupancy distribution.
    Failed shows: occupancy clusters (back rows empty, front packed; or side clusters).
    """
    shows = []
    for show_id in range(n_shows):
        occupancy = np.zeros((n_rows, seats_per_row))
        
        # Healthy show: random occupancy ~65-75%
        if show_id < 40:
            fill_rate = np.random.uniform(0.65, 0.75)
            n_occupied = int(fill_rate * seats_per_row * n_rows)
            occupied_seats = np.random.choice(seats_per_row * n_rows, n_occupied, replace=False)
            occupancy[occupied_seats // seats_per_row, occupied_seats % seats_per_row] = 1
            status = "healthy"
        # Failed show: clustered occupancy
        else:
            # Front 3 rows packed, back 5 rows sparse
            occupancy[:3, :] = 1  # Front rows full
            occupancy[3:, :np.random.randint(2, 5)] = 1  # Back rows: only left side
            status = "failed"
        
        shows.append({
            "show_id": show_id,
            "occupancy_flat": occupancy.flatten(),
            "fill_rate": occupancy.sum() / (seats_per_row * n_rows),
            "status": status
        })
    
    return shows

def compute_occupancy_clustering_anomaly(occupancy_pattern, eps=2.5, min_samples=3):
    """
    Compute anomaly score based on spatial clustering of occupied seats.
    Invert detection: high clustering (low entropy of spatial distribution) = anomaly.
    
    Logic: random occupancy has seats spread evenly. Concentrated occupancy
    creates dense seat clusters detected by DBSCAN. Cluster count and density
    become the anomaly signal.
    """
    occupied_indices = np.where(occupancy_pattern == 1)[0]
    
    if len(occupied_indices) < min_samples:
        return np.nan, 0, 0
    
    # Convert flat indices to 2D seat positions
    positions = np.column_stack(np.divmod(occupied_indices, 12))
    
    scaler = StandardScaler()
    positions_scaled = scaler.fit_transform(positions)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions_scaled)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise = list(clustering.labels_).count(-1)
    
    # Anomaly score: high cluster count relative to occupancy = abnormal clustering
    # (should have 1-2 diffuse clusters in healthy case)
    cluster_density = n_clusters / max(len(occupied_indices), 1)
    
    return cluster_density, n_clusters, n_noise

def run_analysis():
    print("Generating synthetic theater seating data...")
    shows = generate_seating_data()
    
    results = []
    for show in shows:
        cluster_density, n_clusters, n_noise = compute_occupancy_clustering_anomaly(
            show["occupancy_flat"]
        )
        results.append({
            "show_id": show["show_id"],
            "fill_rate": show["fill_rate"],
            "cluster_density": cluster_density,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "true_status": show["status"]
        })
    
    df = pd.DataFrame(results).dropna()
    
    # Anomaly threshold: high cluster_density OR high n_clusters
    threshold_density = df["cluster_density"].quantile(0.75)
    threshold_clusters = df["n_clusters"].quantile(0.75)
    
    df["anomaly_flag"] = (
        (df["cluster_density"] > threshold_density) | 
        (df["n_clusters"] > threshold_clusters)
    )
    
    df["predicted_status"] = df["anomaly_flag"].map({True: "failed", False: "healthy"})
    
    print("\n=== Occupancy Clustering Anomaly Detection ===")
    print(df[["show_id", "fill_rate", "cluster_density", "n_clusters", "predicted_status", "true_status"]].to_string())
    
    accuracy = (df["predicted_status"] == df["true_status"]).sum() / len(df)
    print(f"\nDetection Accuracy: {accuracy:.2%}")
    print(f"\nKey Insight: Failed shows exhibit high spatial clustering of occupied seats.")
    print(f"Normal anomaly detection finds rare events. This inverts to find when occupancy")
    print(f"*patterns* (clustering structure) violate random-distribution assumptions.")
    
    return df

if __name__ == "__main__":
    df = run_analysis()
