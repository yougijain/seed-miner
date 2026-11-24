import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# Synthetic dataset: 20 garden plots with stable agronomic features
# and endogenously driven harvest-window overlap that reshapes clustering.
np.random.seed(42)
n_plots = 20

# Base agronomic features (stable):
soil_ph = np.random.uniform(6.0, 7.5, n_plots)
sun_hours = np.random.uniform(4, 10, n_plots)
water_access = np.random.uniform(0, 10, n_plots)  # 0=poor, 10=excellent

# Gardener base crop choice (independent, uniform):
crop_type = np.random.choice([0, 1, 2], n_plots)  # 0=tomato, 1=lettuce, 2=root

# Base harvest window (days from start of season):
base_harvest_day = np.random.uniform(60, 120, n_plots)

# ENDOGENEITY: Shared resources force harvest-window clustering.
# Gardeners with plots near shared compost/water stations are pressured to
# harvest within 7-day windows (logistics, shared equipment availability).
station_proximity = np.random.uniform(0, 1, n_plots)  # 0=far, 1=near shared station
harvest_window_offset = np.where(
    station_proximity > 0.6,
    np.random.choice([-5, 0, 5], sum(station_proximity > 0.6)),
    np.random.uniform(-15, 15, sum(station_proximity <= 0.6))
)
harvest_day = base_harvest_day + harvest_window_offset

# Scenario 1: Cluster on stable agronomic features only.
X_stable = np.column_stack([soil_ph, sun_hours, water_access])
scaler = StandardScaler()
X_stable_scaled = scaler.fit_transform(X_stable)

kmeans_stable = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_stable = kmeans_stable.fit_predict(X_stable_scaled)

# Scenario 2: Cluster on agronomic features + harvest-day proximity.
# This simulates what happens when clustering observes the CONSTRAINED
# feature space (harvest day is now a confounding feature).
X_constrained = np.column_stack([soil_ph, sun_hours, water_access, harvest_day])
X_constrained_scaled = scaler.fit_transform(X_constrained)

kmeans_constrained = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_constrained = kmeans_constrained.fit_predict(X_constrained_scaled)

# Measure cluster stability: how many plots switched clusters?
switches = np.sum(labels_stable != labels_constrained)

# Identify plots near shared station whose labels flipped:
station_near_idx = np.where(station_proximity > 0.6)[0]
switches_in_station = np.sum(labels_stable[station_near_idx] != labels_constrained[station_near_idx])

results = {
    "question": "Do harvest-window constraints (endogenous scheduling) reshape K-means clustering of garden plots?",
    "method": "K-means on stable agronomic features vs. agronomic+harvest-day features",
    "n_plots": n_plots,
    "total_cluster_switches": int(switches),
    "switches_near_station": int(switches_in_station),
    "station_near_total": int(len(station_near_idx)),
    "interpretation": (
        f"Plots near shared stations: {len(station_near_idx)} total, "
        f"{switches_in_station} switched clusters when harvest-day added. "
        f"This suggests harvest-window logistics reshape clustering even though "
        f"soil/sun/water (true agronomic drivers) remain constant."
    ),
    "caveat": "Synthetic data by design exhibits endogeneity; real data would require instrumental variable or causal inference to isolate effect."
}

print(json.dumps(results, indent=2))
