import re
import json
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

def tokenize_log(text):
    """Lowercase and split into words; keep minimal structure."""
    return re.findall(r'\b[a-z]+\b', text.lower())

def compute_vocabulary_richness(tokens, window_size=50):
    """Compute type/token ratio (TTR) in rolling windows to track lexical drift."""
    if len(tokens) < window_size:
        return []
    richness = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        unique = len(set(window))
        ttr = unique / len(window)
        richness.append(ttr)
    return richness

def compute_observation_detail(log_entry):
    """Count numeric values and conditional phrases as proxy for inspection depth."""
    num_count = len(re.findall(r'\d+', log_entry))
    cond_phrases = len(re.findall(r'(appear|seem|look|notice|spot|see)', log_entry.lower()))
    return num_count + cond_phrases

def generate_synthetic_logs(num_entries=30, base_vocab=None):
    """Generate synthetic beekeeper logs with coherence decay over time."""
    if base_vocab is None:
        base_vocab = [
            'brood', 'frame', 'queen', 'worker', 'comb', 'honey', 'pollen',
            'flight', 'forager', 'entrance', 'temperature', 'cluster', 'laying',
            'capped', 'uncapped', 'sealed', 'inspected', 'healthy', 'weak', 'strong'
        ]
    
    logs = []
    decay_factor = 0.98  # Coherence decays each entry
    current_richness = 0.7
    
    for day in range(num_entries):
        # Early entries: detailed, repetitive (high coherence)
        if day < 10:
            entry = f"Inspected frames 1-5. Queen laying in pattern. Brood capped sealed. Temperature stable. Pollen stores adequate. Cluster active."
        # Mid entries: slightly less structured
        elif day < 20:
            entry = f"Day {day}: checked brood frames. queen active. honey and pollen good. foragers seen. cluster fine."
        # Late entries: fragmented, less specific (decay in coherence)
        else:
            entry = f"Visit {day}. Inspected. Queen seen. Brood ok. Cluster normal. Entrance activity ok."
        
        # Add noise proportional to day
        if day > 15:
            entry += " checked entrance. activity good." if np.random.rand() > 0.5 else ""
        
        logs.append({"day": day, "entry": entry})
    
    return logs

def analyze_coherence_decay(logs):
    """Measure coherence (vocab richness) over time and detect decay trajectory."""
    all_tokens = []
    window_richness = []
    detail_scores = []
    
    for i, log in enumerate(logs):
        tokens = tokenize_log(log['entry'])
        all_tokens.extend(tokens)
        detail_scores.append(compute_observation_detail(log['entry']))
    
    # Global richness in rolling 5-log windows
    for i in range(len(logs) - 4):
        window_entries = [l['entry'] for l in logs[i:i+5]]
        window_tokens = []
        for e in window_entries:
            window_tokens.extend(tokenize_log(e))
        if window_tokens:
            richness = len(set(window_tokens)) / len(window_tokens)
            window_richness.append(richness)
    
    # Fit linear trend to richness
    if len(window_richness) > 2:
        x = np.arange(len(window_richness))
        coeffs = np.polyfit(x, window_richness, 1)
        decay_slope = coeffs[0]
    else:
        decay_slope = 0
    
    return {
        "window_richness": window_richness,
        "detail_scores": detail_scores,
        "decay_slope": decay_slope,
        "avg_richness": np.mean(window_richness) if window_richness else 0,
        "coherence_signal": "declining" if decay_slope < -0.005 else "stable"
    }

if __name__ == "__main__":
    # Generate synthetic logs
    logs = generate_synthetic_logs(num_entries=30)
    
    # Analyze
    results = analyze_coherence_decay(logs)
    
    print("=== Beekeeper Log Coherence Decay Analysis ===")
    print(f"Decay slope (richness per window): {results['decay_slope']:.6f}")
    print(f"Average vocabulary richness: {results['avg_richness']:.3f}")
    print(f"Coherence signal: {results['coherence_signal']}")
    print(f"\nWindow richness trajectory:")
    for i, r in enumerate(results['window_richness']):
        print(f"  Window {i}: {r:.3f}")
    print(f"\nDetail scores by entry:")
    for i, d in enumerate(results['detail_scores'][:10]):
        print(f"  Entry {i}: {d}")
    print(f"  ... (showing first 10 of {len(results['detail_scores'])})")
    
    print(f"\n=== Interpretation ===")
    print(f"A negative decay slope suggests beekeeper logs become less lexically diverse")
    print(f"and internally inconsistent over the season, potentially signaling fatigue,")
    print(f"reduced colony complexity, or declining inspection thoroughness.")
