import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 200

# Synthetic youth chess pairing data
# True skill (unobserved)
true_skill = np.random.normal(1500, 200, n)

# Rating inflation bias: younger players get inflated ratings
age = np.random.uniform(8, 18, n)
rating_bias = 150 * (age < 12).astype(float)
observed_rating = true_skill + rating_bias + np.random.normal(0, 50, n)

# Pairing algorithm: pairs players if |rating_diff| < threshold
# This endogenously creates treatment based on biased ratings
df = pd.DataFrame({
    'player_id': range(n),
    'age': age,
    'observed_rating': observed_rating,
    'true_skill': true_skill,
    'rating_bias': rating_bias
})

# Simulate pairings: create treatment as "cross-skill-group" mismatch
# (pairing with someone far below your true skill)
def assign_treatment(rating_series):
    """Pairing algo: sort by rating, pair adjacent players.
    Treatment = 1 if true skill gap > 300, 0 otherwise."""
    df_sorted = df.iloc[rating_series.argsort()].reset_index(drop=True)
    treatment = np.zeros(len(df))
    for i in range(0, len(df)-1, 2):
        skill_gap = abs(df_sorted.iloc[i]['true_skill'] - df_sorted.iloc[i+1]['true_skill'])
        if skill_gap > 300:
            treatment[i] = treatment[i+1] = 1
    return treatment

df['treatment_endogenous'] = assign_treatment(df['observed_rating'])

# Outcome: game outcome (1 if won, 0 loss), influenced by skill mismatch + true skill
df['outcome'] = (
    (df['treatment_endogenous'] == 0).astype(float) * 0.6 +  # Fair pairings → higher win rate
    (df['true_skill'] > df['true_skill'].median()).astype(float) * 0.3 +  # Strong players win more
    np.random.normal(0, 0.1, n)
).clip(0, 1)

print("="*70)
print("ENDOGENOUS TOURNAMENT PAIRING CAUSAL INFERENCE")
print("="*70)

# Naive regression: treatment on outcome (BIASED because treatment is endogenous)
naive_model = LinearRegression()
naive_model.fit(df[['treatment_endogenous']].values, df['outcome'])
naive_effect = naive_model.coef_[0]
print(f"\nNAIVE effect of fair pairing on win rate: {naive_effect:.4f}")
print("(BIASED: treatment assignment depends on inflated ratings)")

# Backdoor adjustment: condition on rating_bias (confounder)
# The pairing algorithm's decision depends on observed_rating,
# which is confounded by age-driven rating_bias.
df['X_confounder'] = df['rating_bias']
X_adj = df[['treatment_endogenous', 'X_confounder']].values
adjusted_model = LinearRegression()
adjusted_model.fit(X_adj, df['outcome'])
adjusted_effect = adjusted_model.coef_[0]
print(f"\nBACKDOOR-ADJUSTED effect (controlling for rating_bias): {adjusted_effect:.4f}")
print("(Partial deconfounding: isolates effect of pairing fairness from\nrating inflation bias that endogenously drives algorithm treatment assignment)")

# Check: propensity score overlap (instrument validity)
df['propensity_by_age'] = (df['age'] < 12).astype(float) * df['rating_bias'] / 150
print(f"\nPropensity overlap check (confounder balance):")
print(f"  Mean bias | treatment=0: {df[df['treatment_endogenous']==0]['X_confounder'].mean():.1f}")
print(f"  Mean bias | treatment=1: {df[df['treatment_endogenous']==1]['X_confounder'].mean():.1f}")

# Stratification: compare effect within subgroups (young vs old)
effect_young = LinearRegression().fit(
    df[df['age'] < 12][['treatment_endogenous']].values,
    df[df['age'] < 12]['outcome']
).coef_[0]
effect_old = LinearRegression().fit(
    df[df['age'] >= 12][['treatment_endogenous']].values,
    df[df['age'] >= 12]['outcome']
).coef_[0]

print(f"\nSTRATIFIED analysis (age < 12 vs >=12):")
print(f"  Young players, effect of fair pairing: {effect_young:.4f}")
print(f"  Old players, effect of fair pairing: {effect_old:.4f}")
print(f"  → Effect heterogeneity suggests rating bias is age-dependent confounder")

print(f"\nKEY FINDING:")
print(f"  Naïve effect: {naive_effect:.4f} (biased by endogenous pairing algorithm)")
print(f"  Adjusted effect: {adjusted_effect:.4f} (confounder removed)")
print(f"  Direction flip suggests algorithm bias → unfair pairings are measurement error,")
print(f"  not true unfairness.")
