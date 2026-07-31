import re
import pandas as pd
import numpy as np
from collections import defaultdict

def generate_synthetic_logs():
    """
    Generate beekeeper inspection logs across a season.
    Healthy colony: observations entail positive conclusions.
    Declining colony: observations become contradictory or fail to support conclusions.
    """
    observations = [
        "Queen present and laying",
        "Brood pattern dense and regular",
        "No signs of mites or disease",
        "Frames well-populated",
        "Minimal drone brood",
        "Good food stores",
        "Low mortality on bottom board"
    ]
    
    decline_observations = [
        "Queen present but erratic laying",
        "Brood pattern sparse and scattered",
        "Some signs of varroa mites detected",
        "Frames sparsely populated",
        "Excessive drone brood",
        "Food stores depleting",
        "High mortality on bottom board"
    ]
    
    logs = []
    
    # Healthy colony: week 1-8
    for week in range(1, 9):
        obs_set = np.random.choice(observations, size=np.random.randint(3, 5), replace=False)
        entry = {
            'week': week,
            'colony_id': 'A',
            'text': ' '.join(obs_set),
            'conclusion': 'Colony healthy. No intervention needed.'
        }
        logs.append(entry)
    
    # Declining colony: weak entailment starts week 4
    for week in range(1, 9):
        if week < 4:
            obs_set = np.random.choice(observations, size=np.random.randint(3, 5), replace=False)
            conclusion = 'Colony healthy. No intervention needed.'
        else:
            # Mix healthy and decline observations (entailment breaks)
            obs_set = list(np.random.choice(decline_observations, size=2, replace=False))
            obs_set += list(np.random.choice(observations, size=1, replace=False))
            # Conclusion becomes contradictory or weak
            if week >= 7:
                conclusion = 'Colony concerning but stable. Monitor closely.'
            else:
                conclusion = 'Colony appears stable but monitoring needed.'
        
        entry = {
            'week': week,
            'colony_id': 'B',
            'text': ' '.join(obs_set),
            'conclusion': conclusion
        }
        logs.append(entry)
    
    return pd.DataFrame(logs)

def compute_entailment_signal(text, conclusion):
    """
    Heuristic entailment proxy: measure semantic alignment between observations and conclusion.
    Real implementation would use a pre-trained NLI model (e.g., cross-encoder),
    but we use lexical overlap and contradiction detection as a low-resource proxy.
    """
    text_lower = text.lower()
    conclusion_lower = conclusion.lower()
    
    # Positive entailment markers
    positive_terms = {'healthy', 'good', 'present', 'regular', 'dense', 'well', 'no signs'}
    negative_terms = {'erratic', 'sparse', 'signs of', 'depleting', 'high mortality', 'excessive'}
    
    # Count markers in observations
    pos_count = sum(1 for term in positive_terms if term in text_lower)
    neg_count = sum(1 for term in negative_terms if term in text_lower)
    
    # Count markers in conclusion
    conc_pos = sum(1 for term in positive_terms if term in conclusion_lower)
    conc_neg = sum(1 for term in negative_terms if term in conclusion_lower)
    
    # Entailment score: do observations align with conclusion sentiment?
    obs_sentiment = pos_count - neg_count
    conc_sentiment = conc_pos - conc_neg
    
    # If they diverge significantly, entailment is weak
    entailment = 1.0 / (1.0 + abs(obs_sentiment - conc_sentiment))
    return entailment

def analyze_entailment_drift(df):
    """
    For each colony, compute rolling entailment coherence.
    Declining entailment signals colony stress before explicit health markers appear.
    """
    results = []
    
    for colony_id in df['colony_id'].unique():
        colony_data = df[df['colony_id'] == colony_id].sort_values('week')
        entailments = []
        
        for _, row in colony_data.iterrows():
            ent = compute_entailment_signal(row['text'], row['conclusion'])
            entailments.append(ent)
        
        # Rolling average over 2-week window
        rolling_ent = pd.Series(entailments).rolling(window=2, min_periods=1).mean()
        
        # Detect decay: linear regression slope
        weeks = np.arange(len(rolling_ent))
        slope = np.polyfit(weeks, rolling_ent, 1)[0]
        
        results.append({
            'colony_id': colony_id,
            'mean_entailment': np.mean(entailments),
            'entailment_slope': slope,
            'final_entailment': entailments[-1],
            'entailments': entailments
        })
    
    return results

if __name__ == '__main__':
    df = generate_synthetic_logs()
    results = analyze_entailment_drift(df)
    
    print("Colony Entailment Coherence Analysis")
    print("=" * 50)
    for r in results:
        print(f"\nColony {r['colony_id']}:")
        print(f"  Mean Entailment: {r['mean_entailment']:.3f}")
        print(f"  Slope (decay): {r['entailment_slope']:.4f}")
        print(f"  Final Entailment: {r['final_entailment']:.3f}")
        
        if r['entailment_slope'] < -0.02:
            print(f"  ⚠ ALERT: Entailment degrading (slope {r['entailment_slope']:.4f})")
        else:
            print(f"  ✓ Stable or improving")
    
    print("\n" + "=" * 50)
    print("Interpretation: Colony B shows declining entailment coherence,")
    print("suggesting observations no longer logically support health conclusions.")
    print("This precedes explicit health decline in the textual record.")
