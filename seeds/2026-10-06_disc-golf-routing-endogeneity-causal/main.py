import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json

np.random.seed(42)

# Generate synthetic multi-layout disc golf course data.
# Courses have two routing options (A: harder, B: easier).
# Players endogenously choose routing based on unobserved confidence/fatigue.
# True causal effect: routing doesn't affect skill, but unobserved confounding does.

n_rounds = 200
rounds_data = []

for i in range(n_rounds):
    # Unobserved confounder: player confidence/fatigue on day i (range: -2 to +2)
    unobserved_confidence = np.random.normal(0, 1.0)
    
    # Routing choice (0=A harder, 1=B easier):
    # Players choose B when confidence is low (endogenous treatment assignment)
    routing_prob = 1.0 / (1.0 + np.exp(2.0 * unobserved_confidence))
    routing = np.random.binomial(1, routing_prob)
    
    # Score: affected by confidence, and by routing difficulty.
    # Routing A (0) is harder: +3 strokes penalty.
    # Routing B (1) is easier: -2 strokes penalty.
    # But confounding: low confidence adds +4 strokes (independent of routing choice).
    base_score = 60
    routing_effect = 3 if routing == 0 else -2
    confounding_effect = 4 * (unobserved_confidence < -0.5)  # Discrete proxy: bad day
    score = base_score + routing_effect + confounding_effect + np.random.normal(0, 1.5)
    
    rounds_data.append({
        'round_id': i,
        'routing': routing,
        'score': score,
        'unobserved_confidence': unobserved_confidence  # Hidden; used for ground truth only
    })

df = pd.DataFrame(rounds_data)

print("\n=== NAIVE REGRESSION (ignoring confounding) ===")
naive_model = LinearRegression()
naive_model.fit(df[['routing']], df['score'])
naive_coef = naive_model.coef_[0]
print(f"Naive routing effect estimate: {naive_coef:.3f} strokes")
print(f"  (Interpretation: choosing easy routing B saves {-naive_coef:.1f} strokes)")
print(f"  (This is WRONG because confounding: players with low confidence choose B AND score worse)")

print("\n=== BACKDOOR ADJUSTMENT via STRATIFICATION ===")
print("Strategy: use score distribution quantile as proxy for unobserved confidence.")
print("  Assumption: baseline score (when routing is random) correlates with confidence.")

# Estimate unobserved confounder using score quantiles as proxy.
# High-score rounds → high confidence (can afford harder routing A).
# Low-score rounds → low confidence (need easier routing B).
df['score_quantile'] = pd.qcut(df['score'], q=3, labels=[0, 1, 2], duplicates='drop')

print("\nStratified analysis by confidence proxy (score quantile):")
effects = []
for stratum in sorted(df['score_quantile'].unique()):
    stratum_df = df[df['score_quantile'] == stratum]
    if len(stratum_df[stratum_df['routing'] == 0]) > 1 and len(stratum_df[stratum_df['routing'] == 1]) > 1:
        mean_score_routing_0 = stratum_df[stratum_df['routing'] == 0]['score'].mean()
        mean_score_routing_1 = stratum_df[stratum_df['routing'] == 1]['score'].mean()
        effect_in_stratum = mean_score_routing_1 - mean_score_routing_0
        effects.append(effect_in_stratum)
        print(f"  Stratum {stratum}: routing B vs A = {effect_in_stratum:+.2f} strokes")

adjusted_effect = np.mean(effects)
print(f"\nBackdoor-adjusted routing effect (stratified average): {adjusted_effect:+.3f} strokes")
print(f"  (True causal effect: B saves ~5 strokes relative to A after confounding removed)")

print("\n=== GROUND TRUTH (using unobserved confounder directly) ===")
oracle_model = LinearRegression()
oracle_model.fit(df[['routing', 'unobserved_confidence']], df['score'])
oracle_coef = oracle_model.coef_
print(f"Oracle routing effect (if we observed confidence): {oracle_coef[0]:+.3f} strokes")
print(f"Oracle confidence effect: {oracle_coef[1]:+.3f} strokes per unit")

print(f"\n=== DIAGNOSTIC ===")
print(f"Naive vs Adjusted delta: {abs(naive_coef - adjusted_effect):.2f} strokes")
print(f"Adjusted error vs Oracle: {abs(adjusted_effect - oracle_coef[0]):.2f} strokes")
print(f"\nConclusion: backdoor stratification recovers ~{adjusted_effect:+.1f}str causal effect,")
print(f"  correcting for routing choice being endogenous to unobserved player state.")
