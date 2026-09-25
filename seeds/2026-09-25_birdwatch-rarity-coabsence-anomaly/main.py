import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import json

np.random.seed(42)

# Synthetic competitive birdwatching sighting log.
# Structure: sparse observations; each row is (date, observer_id, species_list).
# Key insight: rare species that share habitat will show *coordinated absences*.

species_pool = ['Robin', 'Sparrow', 'Finch', 'Hawk', 'Warbler', 
                 'Heron', 'Egret', 'Sandpiper', 'Vulture', 'Crane']

# Generate 90-day sighting log: ~20 observations, each observer sees 3-5 species.
observations = []
for day in range(1, 91):
    if np.random.random() < 0.22:  # ~20 observations over 90 days
        observer = f'obs_{np.random.randint(1, 6)}'
        # Normal: each species has 60-80% detection probability per day.
        sightings = [sp for sp in species_pool if np.random.random() < 0.7]
        if sightings:  # only log if non-empty
            observations.append({'day': day, 'observer': observer, 'species': sightings})

# Inject ecological pairing: Heron and Egret always co-occur or co-absent (shared wetland).
for obs in observations:
    if np.random.random() < 0.3:  # 30% of observations
        if 'Heron' in obs['species']:
            if 'Egret' not in obs['species']:
                obs['species'].append('Egret')
        elif 'Egret' in obs['species']:
            if 'Heron' not in obs['species']:
                obs['species'].append('Heron')
        else:
            if np.random.random() < 0.5:
                obs['species'].extend(['Heron', 'Egret'])

print(f"Generated {len(observations)} observations over 90 days.\n")

# Convert to pairwise co-absence feature matrix.
# For each pair of species (s1, s2), compute a feature vector:
# [days_both_absent, days_s1_absent_s2_present, days_s2_absent_s1_present, days_both_present]

species_presence = defaultdict(lambda: [False] * 90)
for obs in observations:
    for sp in obs['species']:
        species_presence[sp][obs['day'] - 1] = True

pairs = []
feature_matrix = []

for i, sp1 in enumerate(species_pool):
    for sp2 in species_pool[i+1:]:
        p1 = np.array(species_presence[sp1])
        p2 = np.array(species_presence[sp2])
        
        both_absent = np.sum((~p1) & (~p2))
        s1_absent_s2_present = np.sum((~p1) & p2)
        s2_absent_s1_present = np.sum(p1 & (~p2))
        both_present = np.sum(p1 & p2)
        
        # Feature for anomaly detection: normalized co-absence strength.
        # High co-absence rate (both absent) relative to discordant absence is anomalous.
        total_absent_days = both_absent + s1_absent_s2_present + s2_absent_s1_present
        if total_absent_days > 0:
            coabsence_rate = both_absent / total_absent_days
        else:
            coabsence_rate = 0
        
        feature_matrix.append([
            both_absent,
            coabsence_rate,
            both_present,
            s1_absent_s2_present + s2_absent_s1_present
        ])
        pairs.append((sp1, sp2))

feature_matrix = np.array(feature_matrix)

# Apply Isolation Forest to detect pairs with anomalously high co-absence structure.
iso_forest = IsolationForest(contamination=0.15, random_state=42)
anomaly_labels = iso_forest.fit_predict(feature_matrix)
anomaly_scores = iso_forest.score_samples(feature_matrix)

print("Species pairs ranked by co-absence anomaly score (most anomalous first):\n")
results = [(pairs[i], anomaly_scores[i], feature_matrix[i]) 
           for i in range(len(pairs))]
results.sort(key=lambda x: x[1])

print(f"{'Species Pair':<30} {'Anomaly Score':<15} {'Co-abs Days':<12} {'Co-abs Rate':<12} {'Co-pres Days':<12}")
print("-" * 85)

for (sp1, sp2), score, feats in results[:12]:
    both_absent, coabs_rate, both_pres, discord = feats
    print(f"{sp1}-{sp2:<27} {score:<15.3f} {int(both_absent):<12} {coabs_rate:<12.2f} {int(both_pres):<12}")

print("\n[Expected anomalies: Heron-Egret should rank high due to ecological pairing.]")
print("\nInterpretation: Pairs with high co-absence rates are candidates for shared rare habitat.")
print("Standard presence-based clustering would miss this; we detect from *silence*.")
