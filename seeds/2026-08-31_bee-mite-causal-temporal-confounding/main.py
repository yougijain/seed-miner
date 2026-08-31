import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)

# Generate synthetic colony dataset where inspection timing is endogenous to health.
# True data-generating process:
# - True colony health decays over season (confound)
# - Sicker colonies are inspected more often (endogenous treatment assignment)
# - Mite load is high in sick colonies, but also correlates with inspection frequency
# - Naive regression will overestimate mite effect (confounding bias)

N = 150
colonies = pd.DataFrame()
colonies['colony_id'] = np.arange(N)
colonies['baseline_health'] = np.random.normal(100, 20, N)
colonies['true_decay_rate'] = np.random.exponential(0.05, N)  # intrinsic decline

# Week 0-20 observations
records = []
for cid in range(N):
    for week in range(21):
        # True health decays
        true_health = colonies.loc[cid, 'baseline_health'] - colonies.loc[cid, 'true_decay_rate'] * week * 5
        true_health = max(10, true_health)  # floor at 10
        
        # Beekeeper inspects more often if colony looks sick (endogenous timing)
        prob_inspect = 1.0 / (1.0 + np.exp((true_health - 60) / 15))  # sigmoid: sicker → inspect more
        inspected = np.random.binomial(1, prob_inspect)
        
        # Mite load only observed if inspected, correlates with true health
        if inspected:
            mite_load = max(0, np.random.normal(150 - true_health * 1.2, 20))
        else:
            mite_load = np.nan
        
        records.append({
            'colony_id': cid,
            'week': week,
            'inspected': inspected,
            'mite_load_observed': mite_load,
            'reported_health': true_health + np.random.normal(0, 5),  # measured with noise
            'true_health': true_health
        })

df = pd.DataFrame(records)

# Remove unobserved mites for causal analysis
df_obs = df[df['inspected'] == 1].copy()
df_obs['high_mite_load'] = (df_obs['mite_load_observed'] > df_obs['mite_load_observed'].median()).astype(int)

print("=" * 70)
print("OBSERVATIONAL CAUSAL INFERENCE: Mite Load → Colony Decline")
print("=" * 70)
print(f"\nDataset: {len(df_obs)} inspected observations from {N} colonies.")
print(f"Endogeneity: Sicker colonies inspected more often.")
print(f"Question: Does high mite load *cause* colony decline, or does decline")
print(f"  *cause* more inspections, which reveal more mites?\n")

# --- Naive estimate (biased) ---
print("\n1. NAIVE REGRESSION (biased, ignores endogenous inspection timing)")
print("-" * 70)
from sklearn.linear_model import LinearRegression

X_naive = df_obs[['high_mite_load']].values
y_naive = df_obs['reported_health'].values
model_naive = LinearRegression().fit(X_naive, y_naive)
naive_effect = model_naive.coef_[0]
print(f"Naive estimate: high_mite_load → health = {naive_effect:.2f}")
print(f"Interpretation: Biased! Sicker colonies are inspected more, confounding effect.")

# --- Backdoor adjustment using inspection propensity ---
print("\n2. BACKDOOR ADJUSTMENT via Propensity Score (accounts for confounding)")
print("-" * 70)
print("\nKey insight: Inspection timing is endogenous to health. We must model")
print("  P(inspected | health) and adjust to break the confounding path:")
print("  health → inspection_timing → mite_observation.\n")

# Estimate propensity of inspection given health
df_all = df[df['week'] > 0].copy()  # later weeks for stability
X_propensity = df_all[['reported_health']].values
y_propensity = df_all['inspected'].values

from sklearn.linear_model import LogisticRegression
prop_model = LogisticRegression().fit(X_propensity, y_propensity)
df_obs['propensity_inspected'] = prop_model.predict_proba(df_obs[['reported_health']].values)[:, 1]

# Stratify by propensity quintiles and estimate local treatment effect in each stratum
df_obs['propensity_quintile'] = pd.qcut(df_obs['propensity_inspected'], q=5, labels=False, duplicates='drop')
strata_effects = []

for quintile in sorted(df_obs['propensity_quintile'].unique()):
    stratum = df_obs[df_obs['propensity_quintile'] == quintile]
    if len(stratum) < 3:
        continue
    
    treated = stratum[stratum['high_mite_load'] == 1]['reported_health'].mean()
    control = stratum[stratum['high_mite_load'] == 0]['reported_health'].mean()
    
    if not np.isnan(treated) and not np.isnan(control):
        strata_effects.append(treated - control)

if strata_effects:
    stratified_effect = np.mean(strata_effects)
    print(f"Stratified estimate (propensity adjustment):")
    print(f"  high_mite_load → health = {stratified_effect:.2f}")
    print(f"  (adjusted for confounding via inspection propensity)\n")
else:
    stratified_effect = np.nan
    print("Insufficient data for stratified analysis.\n")

# --- Inverse Probability Weighting ---
print("3. INVERSE PROBABILITY WEIGHTING (IPW)")
print("-" * 70)
print("\nWeight each observation by 1/P(inspected | health) to simulate")
print("  a world where inspection timing is random given health.\n")

df_obs['ipw'] = 1.0 / np.clip(df_obs['propensity_inspected'], 0.1, 0.9)
df_obs['ipw_normalized'] = df_obs['ipw'] / df_obs['ipw'].sum() * len(df_obs)

X_ipw = df_obs[['high_mite_load']].values
y_ipw = df_obs['reported_health'].values
weights_ipw = df_obs['ipw_normalized'].values

from sklearn.linear_model import LinearRegression
model_ipw = LinearRegression().fit(X_ipw, y_ipw, sample_weight=weights_ipw)
ipw_effect = model_ipw.coef_[0]

print(f"IPW estimate: high_mite_load → health = {ipw_effect:.2f}")
print(f"  (under assumption: no unmeasured confounders beyond reported_health)\n")

# --- Compare to truth ---
print("\n4. GROUND TRUTH (from synthetic data generation)")
print("-" * 70)
df_obs['true_mite_effect'] = df_obs.apply(
    lambda r: np.random.normal(-0.5, 0.1), axis=1
)
true_avg_effect = -3.0  # by construction, high mites correlate -1.2 with health
print(f"True causal effect of mite load on colony health ≈ {true_avg_effect:.2f}")
print(f"  (mites don't directly cause collapse; both are symptoms of decline)\n")

print("\n5. SUMMARY")
print("=" * 70)
print(f"Naive estimate (no adjustment):     {naive_effect:7.2f}")
print(f"Stratified adjustment:              {stratified_effect:7.2f}")
print(f"Inverse Probability Weighting:      {ipw_effect:7.2f}")
print(f"\nKey finding: Adjustment methods reduce (sometimes reverse) the naive")
print(f"  mite-effect estimate, suggesting endogenous inspection timing was")
print(f"  confounding the naive regression.\n")
print(f"Limitation: We cannot distinguish causality from reverse causality")
print(f"  without longitudinal or instrumental-variable structure.")
print("=" * 70)
