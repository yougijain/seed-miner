import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from itertools import combinations
import json

# Synthetic inspection log data: 20 colonies, 40 inspections each, 6-month window
np.random.seed(42)
colonies = [f"Col_{i:02d}" for i in range(20)]
symptom_vocab = [
    "pollen_stores", "brood_pattern", "flight_activity",
    "varroa_load", "nosema_spores", "wing_deformity",
    "queen_present", "food_stores", "comb_health", "disease_signs"
]

# Generate logs: most colonies report random symptoms.
# Colonies 5-9 exhibit foraging collapse: pollen_stores, brood_pattern, flight_activity
# all stop appearing together around inspection 20-30 (silent co-absence).
logs = []
for col in colonies:
    col_idx = int(col.split("_")[1])
    for insp_idx in range(40):
        if col_idx in range(5, 10) and 20 <= insp_idx <= 30:
            # Foraging collapse window: these three symptoms vanish together
            symptoms = np.random.choice(
                [s for s in symptom_vocab
                 if s not in ["pollen_stores", "brood_pattern", "flight_activity"]],
                size=np.random.randint(1, 4),
                replace=False
            ).tolist()
        else:
            # Normal: random subset of symptoms
            symptoms = np.random.choice(
                symptom_vocab,
                size=np.random.randint(2, 7),
                replace=False
            ).tolist()
        logs.append({
            "colony": col,
            "inspection_id": insp_idx,
            "notes": " ".join(symptoms) if symptoms else "no_issues"
        })

df = pd.DataFrame(logs)

# NLP: vectorize notes into TF-IDF
vectorizer = TfidfVectorizer(analyzer="word", min_df=1, max_features=10)
vec_matrix = vectorizer.fit_transform(df["notes"]).toarray()
feature_names = vectorizer.get_feature_names_out()

# Build co-absence matrix: for each inspection window, track which symptoms are jointly absent
window_size = 5  # 5-inspection sliding window
co_absence_sigs = []

for col in colonies:
    col_mask = df["colony"] == col
    col_insp_ids = df[col_mask]["inspection_id"].values
    col_vecs = vec_matrix[col_mask.values]
    
    for start_idx in range(len(col_insp_ids) - window_size):
        window_vecs = col_vecs[start_idx:start_idx + window_size]
        # Absence signature: which symptoms have *zero* mentions in this window
        absence_sig = (window_vecs.sum(axis=0) == 0).astype(int)
        # Only care if symptoms are actually absent (not just empty log)
        if absence_sig.sum() > 0:
            co_absence_sigs.append({
                "colony": col,
                "window_start": col_insp_ids[start_idx],
                "absence_sig": tuple(absence_sig),
                "symptoms_absent": ", ".join(
                    feature_names[absence_sig.astype(bool)]
                )
            })

co_absence_df = pd.DataFrame(co_absence_sigs)

# Cluster colonies by their temporal absence patterns
# Signature-distance: jaccard on which symptoms are jointly absent
if len(co_absence_df) > 0:
    # Group by absence signature
    sig_groups = co_absence_df.groupby("absence_sig")["colony"].apply(list).to_dict()
    
    # Rank signatures by frequency and symptom-cluster coherence
    sig_stats = []
    for sig, cols in sig_groups.items():
        absent_symptoms = set(feature_names[np.array(sig, dtype=bool)])
        sig_stats.append({
            "signature": sig,
            "num_colonies_affected": len(set(cols)),
            "colonies": ", ".join(sorted(set(cols))),
            "symptoms_jointly_absent": ", ".join(sorted(absent_symptoms)),
            "frequency": len(cols)
        })
    
    sig_stats.sort(key=lambda x: x["num_colonies_affected"], reverse=True)
    
    print("\n=== TEMPORAL CO-ABSENCE CLUSTERING ===")
    print(f"Found {len(sig_stats)} distinct absence signatures.\n")
    for i, stat in enumerate(sig_stats[:5]):
        print(f"Pattern {i+1}:")
        print(f"  Affected colonies: {stat['colonies']}")
        print(f"  Jointly absent: {stat['symptoms_jointly_absent']}")
        print(f"  Frequency: {stat['frequency']} windows\n")
    
    # Hypothesis: colonies 5-9 should cluster together
    foraging_collapse_cols = {"Col_05", "Col_06", "Col_07", "Col_08", "Col_09"}
    print("\n=== FORAGING COLLAPSE HYPOTHESIS ===")
    print(f"Expected collapse colonies: {foraging_collapse_cols}")
    
    for i, stat in enumerate(sig_stats[:3]):
        detected_cols = set(stat["colonies"].split(", "))
        overlap = detected_cols & foraging_collapse_cols
        print(f"Pattern {i+1} overlap with collapse: {overlap} ({len(overlap)}/5)")
else:
    print("No co-absence patterns detected.")
