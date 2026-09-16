import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime, timedelta
import json

# Synthetic trail maintenance logs with temporal closure patterns
logs_text = """
2024-01-05: North Ridge closed for drainage repair. Work on segment A1 and A2. Expected 3 days.
2024-01-08: Drainage work complete on A1, A2. South Loop now closed for root removal, segments B1, B2, B3.
2024-01-12: South Loop (B1, B2, B3) reopened. East Trail (C1) closed for surface grading.
2024-01-15: East Trail C1 surface work ongoing. West Fork (D1, D2) also closed due to erosion remediation.
2024-01-18: C1 grading done. D1 and D2 erosion work continues.
2024-01-22: D1 reopened, D2 still closed. North Ridge A1 degradation noticed, will close next week.
2024-01-28: A1 closed again for additional drainage. D2 erosion complete.
2024-02-01: A1, A2 closure extended. South Loop B1 minor maintenance noted.
2024-02-05: A1, A2 reopened. B1 closure for bridge inspection begins.
2024-02-08: B1 bridge work ongoing. C1 seasonal closure.
2024-02-12: B1 reopened. C1 still seasonal. A1 minor debris clearance.
"""

# Extract segment mentions and closure dates
def extract_segments_and_dates(logs):
    entries = logs.strip().split('\n')
    segment_closures = {}  # segment -> list of (start_date, duration_days)
    
    for entry in entries:
        if not entry.strip():
            continue
        # Parse date
        match = re.search(r'(\d{4}-\d{2}-\d{2})', entry)
        if not match:
            continue
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Extract segments (A1, A2, B1, etc.)
        segments = re.findall(r'([A-D]\d+)', entry)
        
        # Estimate duration from text
        duration_match = re.search(r'(\d+)\s+days?', entry)
        duration = int(duration_match.group(1)) if duration_match else 3
        
        for seg in segments:
            if seg not in segment_closures:
                segment_closures[seg] = []
            segment_closures[seg].append((date_obj, duration))
    
    return segment_closures

# Build co-absence matrix: segments that are closed at overlapping times
def build_coabsence_matrix(segment_closures):
    segments = sorted(segment_closures.keys())
    n = len(segments)
    coabsence = np.zeros((n, n))
    
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if i == j:
                coabsence[i, j] = 0
            else:
                # Count overlapping closure windows
                closures_i = segment_closures[seg_i]
                closures_j = segment_closures[seg_j]
                overlap_count = 0
                for (start_i, dur_i) in closures_i:
                    end_i = start_i + timedelta(days=dur_i)
                    for (start_j, dur_j) in closures_j:
                        end_j = start_j + timedelta(days=dur_j)
                        # Check overlap
                        if start_i < end_j and start_j < end_i:
                            overlap_count += 1
                coabsence[i, j] = overlap_count
    
    return segments, coabsence

# Extract TF-IDF from raw text as baseline
vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
doc_vecs = vectorizer.fit_transform([logs_text]).toarray()[0]

# Main pipeline
segment_closures = extract_segments_and_dates(logs_text)
segments, coabsence_matrix = build_coabsence_matrix(segment_closures)

print("\n=== Trail Segments and Co-Absence Signal ===")
print(f"Segments: {segments}")
print(f"\nCo-absence matrix (overlapping closure windows):")
df_coabsence = pd.DataFrame(coabsence_matrix, index=segments, columns=segments)
print(df_coabsence)

# Cluster segments by co-absence (temporal affinity)
if coabsence_matrix.max() > 0:
    # Convert to distance: fewer overlaps = higher distance
    distance_matrix = 1.0 / (1.0 + coabsence_matrix)
    np.fill_diagonal(distance_matrix, 0)
    
    clustering = AgglomerativeClustering(n_clusters=2, linkage='average')
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    print(f"\n=== Clustering Results (by temporal co-absence) ===")
    for cluster_id in set(cluster_labels):
        cluster_segs = [segments[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        print(f"Cluster {cluster_id}: {cluster_segs}")
    
    # Output
    result = {
        'segments': segments,
        'coabsence_matrix': df_coabsence.to_dict(),
        'clustering': {segments[i]: int(cluster_labels[i]) for i in range(len(segments))}
    }
    with open('trail_clustering_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nResults saved to trail_clustering_result.json")
else:
    print("No co-absence signal detected.")
