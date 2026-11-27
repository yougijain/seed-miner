import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import defaultdict

# Synthetic inspection log dataset with realistic structure
np.random.seed(42)

colonies = [f"C{i:03d}" for i in range(1, 21)]
inspection_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='7D')

# Health metrics commonly mentioned in inspection notes
metrics = ['brood_pattern', 'pollen_stores', 'honey_reserves', 'bee_population', 
           'varroa_mites', 'wing_deformity', 'flight_activity', 'comb_disease']

# Synthetic data: some colonies share hidden transmission pathways
# Pairs (C001, C002), (C005, C006), (C010, C011) are epidemiologically linked
# When one starts showing disease, both stop mentioning certain metrics together

linked_pairs = [('C001', 'C002'), ('C005', 'C006'), ('C010', 'C011')]

logs = []
for colony in colonies:
    for date in inspection_dates:
        # Base: 70% chance of mentioning each metric per inspection
        mentioned_metrics = [m for m in metrics if np.random.random() > 0.3]
        
        # Inject synchronized silence: linked colonies jointly stop mentioning metrics
        for c1, c2 in linked_pairs:
            if colony in [c1, c2]:
                # Around day 180-250, inject a ~40-day disease window where both colonies
                # jointly stop mentioning varroa_mites, wing_deformity, comb_disease
                if 180 <= date.dayofyear <= 250:
                    if np.random.random() > 0.15:  # 85% silence rate
                        for m in ['varroa_mites', 'wing_deformity', 'comb_disease']:
                            if m in mentioned_metrics:
                                mentioned_metrics.remove(m)
        
        note = f"Inspection {date.strftime('%Y-%m-%d')}: "
        note += "; ".join([f"{m}: OK" for m in mentioned_metrics])
        if not mentioned_metrics:
            note += "No abnormalities detected this week."
        
        logs.append({
            'colony_id': colony,
            'date': date,
            'note': note,
            'mentioned_metrics': mentioned_metrics
        })

df = pd.DataFrame(logs)

# NLP Pipeline: Extract mentioned metrics per colony per week
def extract_metrics_from_note(note, metric_list):
    """Extract which metrics are mentioned in inspection note."""
    found = []
    for metric in metric_list:
        if re.search(r'\b' + metric + r'\b', note, re.IGNORECASE):
            found.append(metric)
    return found

df['extracted_metrics'] = df['note'].apply(lambda x: extract_metrics_from_note(x, metrics))

# Key innovation: Build temporal silence profiles
# For each colony and metric, track which weeks it was NOT mentioned
silence_patterns = defaultdict(lambda: defaultdict(list))

for colony in colonies:
    colony_data = df[df['colony_id'] == colony].sort_values('date')
    for metric in metrics:
        for idx, row in colony_data.iterrows():
            if metric not in row['extracted_metrics']:
                silence_patterns[colony][metric].append(row['date'])

# Detect synchronized silence: which colonies jointly stop mentioning metrics in overlapping windows
def find_synchronized_silence(colony1, colony2, metric, silence_patterns, window_days=14):
    """Count overlapping silence windows between two colonies for a given metric."""
    silence1 = silence_patterns[colony1][metric]
    silence2 = silence_patterns[colony2][metric]
    
    if not silence1 or not silence2:
        return 0
    
    overlap = 0
    for d1 in silence1:
        for d2 in silence2:
            if abs((d1 - d2).days) <= window_days:
                overlap += 1
    return overlap

# Compute pairwise synchronized silence scores
pair_scores = []
for i, c1 in enumerate(colonies):
    for c2 in colonies[i+1:]:
        # Average synchronized silence across all metrics
        sync_score = np.mean([
            find_synchronized_silence(c1, c2, m, silence_patterns)
            for m in metrics
        ])
        pair_scores.append({'colony_1': c1, 'colony_2': c2, 'sync_silence_score': sync_score})

pair_df = pd.DataFrame(pair_scores)
pair_df = pair_df.sort_values('sync_silence_score', ascending=False)

print("Top 10 Colony Pairs with Synchronized Silence (potential disease transmission):")
print(pair_df.head(10).to_string(index=False))
print()
print("Known linked pairs (ground truth):")
for c1, c2 in linked_pairs:
    pair = pair_df[((pair_df['colony_1'] == c1) & (pair_df['colony_2'] == c2)) |
                     ((pair_df['colony_1'] == c2) & (pair_df['colony_2'] == c1))]
    if not pair.empty:
        print(f"  {c1} <-> {c2}: sync_silence = {pair['sync_silence_score'].values[0]:.2f}")
