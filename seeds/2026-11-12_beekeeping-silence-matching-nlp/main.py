import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import defaultdict

# Generate synthetic inspection logs with realistic silence patterns
np.random.seed(42)
colonies = [f'C{i:02d}' for i in range(1, 16)]
health_metrics = ['brood', 'pollen_stores', 'honey_stores', 'varroa_signs', 'disease_signs', 'flight_activity', 'queen_sighting']
date_range = pd.date_range('2024-01-01', periods=60, freq='W')

# Synthetic data: 3 hidden clusters share silence patterns
logs = []
cluster_map = {**{c: 'A' for c in colonies[:5]}, **{c: 'B' for c in colonies[5:10]}, **{c: 'C' for c in colonies[10:]}}

for colony in colonies:
    cluster = cluster_map[colony]
    for week_idx, date in enumerate(date_range):
        text_parts = []
        for metric in health_metrics:
            # Cluster A: silence on brood+pollen together from week 20
            if cluster == 'A' and week_idx >= 20 and metric in ['brood', 'pollen_stores']:
                continue
            # Cluster B: silence on varroa+disease together from week 25
            if cluster == 'B' and week_idx >= 25 and metric in ['varroa_signs', 'disease_signs']:
                continue
            # Cluster C: normal reporting
            if np.random.rand() > 0.15:
                val = np.random.choice(['strong', 'adequate', 'low', 'present', 'absent'])
                text_parts.append(f"{metric}: {val}")
        
        text = "; ".join(text_parts) if text_parts else "[minimal notes]"
        logs.append({'colony': colony, 'date': date, 'notes': text})

df = pd.DataFrame(logs)

# NLP extraction: detect presence of each metric in each record
def extract_metrics(text):
    found = set()
    for metric in health_metrics:
        if metric.replace('_', ' ') in text.lower() or metric in text.lower():
            found.add(metric)
    return found

df['mentioned_metrics'] = df['notes'].apply(extract_metrics)
df['mentioned_count'] = df['mentioned_metrics'].apply(len)

# Build silence timeline: for each (colony, metric), find first week absent
silence_start = {}
for colony in colonies:
    colony_df = df[df['colony'] == colony].sort_values('date').reset_index(drop=True)
    for metric in health_metrics:
        first_silence = None
        for idx, row in colony_df.iterrows():
            if metric not in row['mentioned_metrics']:
                first_silence = idx
                break
        if first_silence is not None:
            silence_start[(colony, metric)] = first_silence

# Co-silence matching: detect colonies that *jointly* entered silence on same metric/timeframe
silence_pairs = defaultdict(list)
for metric in health_metrics:
    colonies_silent = {c: silence_start.get((c, metric), None) for c in colonies}
    colonies_with_silence = {c: w for c, w in colonies_silent.items() if w is not None}
    if len(colonies_with_silence) > 1:
        silent_list = sorted(colonies_with_silence.items(), key=lambda x: x[1])
        for i in range(len(silent_list)):
            for j in range(i+1, len(silent_list)):
                c1, w1 = silent_list[i]
                c2, w2 = silent_list[j]
                # Co-silence if silence started within ±2 weeks
                if abs(w1 - w2) <= 2:
                    silence_pairs[(c1, c2)].append(metric)

# Output silence graph
print("=== DETECTED CO-SILENCE PATTERNS ===")
print("(colonies that jointly stopped mentioning metrics within 2-week windows)\n")
for (c1, c2), metrics in sorted(silence_pairs.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"{c1} <-> {c2}: joint silence on {', '.join(metrics)}")

# Cluster validation: show that co-silence pairs match hidden clusters
print("\n=== CLUSTER MEMBERSHIP (GROUND TRUTH) ===")
for cluster_label in ['A', 'B', 'C']:
    members = [c for c, cl in cluster_map.items() if cl == cluster_label]
    print(f"Cluster {cluster_label}: {members}")

print("\n=== INFERRED CLUSTERS FROM CO-SILENCE GRAPH ===")
# Simple clustering: colonies with strong co-silence links
from collections import Counter
graph = defaultdict(set)
for (c1, c2), metrics in silence_pairs.items():
    if len(metrics) >= 2:  # strong link: 2+ metrics
        graph[c1].add(c2)
        graph[c2].add(c1)

# BFS clustering
visited = set()
inferred_clusters = []
for start in colonies:
    if start not in visited:
        cluster_nodes = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                cluster_nodes.add(node)
                queue.extend(graph[node] - visited)
        if cluster_nodes:
            inferred_clusters.append(sorted(cluster_nodes))

for i, cluster in enumerate(inferred_clusters):
    print(f"Inferred cluster {i}: {cluster}")

print("\n=== INTERPRETATION ===")
print("Co-silence patterns reveal colonies sharing stress pathways.")
print("NLP extracted *absence* (silence) as a structural signal,")
print("inverting normal entity extraction from 'what was mentioned' to")
print("'which mentions stopped together' — revealing hidden dependencies.")
