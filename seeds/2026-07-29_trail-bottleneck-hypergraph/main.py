import numpy as np
import pandas as pd
from collections import defaultdict

np.random.seed(42)

# Synthetic trail GPS dataset: 200 hikers over 4 hours
# Trail is linear with one narrow bottleneck at km 6–8 (time t=150–250).
trail_length = 15  # km
hiker_count = 200
time_steps = 400   # minutes

# Generate baseline hiker trajectories (Poisson arrivals, continuous movement)
trail_positions = []
trail_times = []
hiker_ids = []

for hid in range(hiker_count):
    arrival_time = np.random.exponential(scale=1.0) * 100  # minutes
    if arrival_time > time_steps - 30:
        continue
    speed = np.random.normal(0.05, 0.01)  # km/min (bounded positive)
    speed = max(0.02, min(0.08, speed))
    
    # Simulate slowdown in bottleneck zone (km 6–8)
    for t in np.linspace(arrival_time, time_steps, 50):
        pos = (t - arrival_time) * speed
        if pos > trail_length:
            break
        # Bottleneck: reduce speed by 70% in zone [6, 8]
        if 6 <= pos <= 8:
            pos = 6 + (pos - 6) * 0.3 / speed / 0.05  # synthetic slowdown
        if pos <= trail_length:
            trail_positions.append(pos)
            trail_times.append(t)
            hiker_ids.append(hid)

df = pd.DataFrame({
    'hiker_id': hiker_ids,
    'time': trail_times,
    'position': trail_positions
})

df = df.sort_values(['hiker_id', 'time']).reset_index(drop=True)

# Temporal hypergraph collapsing: bin time into windows, treat as digraph
window_size = 20  # minutes
windows = range(0, time_steps, window_size)

def build_temporal_digraph(df_window, window_id):
    """
    Within a time window, treat each hiker position as a node.
    Draw directed edges from hiker[i] at t to hiker[i] at t+dt (self-loop, ignored)
    and from hiker[i] to hiker[j] if they occupy overlapping trail zones.
    Bottleneck = high clustering, low out-degree variance (many hikers forced to same positions).
    """
    if len(df_window) < 2:
        return None, None
    
    edges = defaultdict(int)
    nodes = set()
    
    # Group by position bins (1 km resolution) to detect spatial overlaps.
    pos_bins = (df_window['position'] // 1).astype(int)
    df_window = df_window.copy()
    df_window['pos_bin'] = pos_bins
    
    for pos_bin, group in df_window.groupby('pos_bin'):
        hikers_in_bin = group['hiker_id'].unique()
        nodes.update(hikers_in_bin)
        # If >1 hiker in same position bin, they're in "contact"—edge weight.
        if len(hikers_in_bin) > 1:
            for h in hikers_in_bin:
                edges[('contact', pos_bin)] += 1
    
    if not nodes:
        return None, None
    
    # Compute in-degree skew: bottleneck = high variance (some positions packed, others empty)
    in_deg = defaultdict(int)
    for hid in nodes:
        hiker_pos = df_window[df_window['hiker_id'] == hid]['position'].values
        if len(hiker_pos) > 0:
            in_deg[int(hiker_pos[-1] // 1)] += 1  # count by final position bin
    
    degrees = list(in_deg.values())
    if len(degrees) < 2:
        return None, None
    
    skew = np.std(degrees) / (np.mean(degrees) + 1e-6)  # in-degree Gini-like metric
    return window_id, skew

# Compute temporal graph skew per window
results = []
for w_start in windows:
    w_end = w_start + window_size
    window_data = df[(df['time'] >= w_start) & (df['time'] < w_end)]
    win_id, skew = build_temporal_digraph(window_data, w_start)
    if skew is not None:
        results.append({'window_start': win_id, 'in_degree_skew': skew, 'bottleneck_signal': skew > 0.5})

results_df = pd.DataFrame(results)
print("\n=== Temporal Digraph In-Degree Skew (Bottleneck Detection) ===")
print(results_df[['window_start', 'in_degree_skew', 'bottleneck_signal']].head(20))
print(f"\nBottleneck windows (skew > 0.5): {results_df['bottleneck_signal'].sum()} / {len(results_df)}")
print(f"Expected bottleneck zone: t ~ 150–250 min (trail pos 6–8 km)")
if (results_df[(results_df['window_start'] >= 140) & (results_df['window_start'] <= 260)]['bottleneck_signal']).sum() > 3:
    print("✓ Signal detected in expected zone.")
