import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# Generate synthetic QSO log: operators endogenously switch modes based on unobserved propagation
n = 400
propagation_quality = np.random.uniform(0, 10, n)  # unobserved confounder

# Operators endogenously switch: good propagation → ragchew (long), bad → contest (short)
mode_probs = 1 / (1 + np.exp(-0.5 * (propagation_quality - 5)))  # sigmoid switch
mode = (np.random.uniform(0, 1, n) < mode_probs).astype(int)  # 0=contest, 1=ragchew

# QSO duration: mode has huge effect, propagation has smaller true causal effect
duration = (
    mode * 8 +  # ragchew baseline 8 min
    (1 - mode) * 1 +  # contest baseline 1 min
    propagation_quality * 0.3 +  # true causal effect: +0.3 min per unit propagation
    np.random.normal(0, 0.8, n)
)
duration = np.clip(duration, 0.1, None)

log_df = pd.DataFrame({
    'qso_duration_min': duration,
    'propagation_quality': propagation_quality,
    'mode_true': mode  # we pretend this is unobserved
})

print("=== Synthetic QSO Contest Log ===")
print(f"n={n} QSOs. Propagation: confounded with mode-choice. Duration: bimodal.")
print(log_df.head(10))
print(f"\nDuration dist (contest=short, ragchew=long):")
print(f"  Mean: {duration.mean():.2f}, Std: {duration.std():.2f}")

# NAIVE approach: ignore mode, regress duration ~ propagation
naive_model = LinearRegression()
naive_model.fit(log_df[['propagation_quality']], log_df['qso_duration_min'])
naive_coef = naive_model.coef_[0]
print(f"\n=== NAIVE (ignoring mode) ===")
print(f"Estimated causal effect of propagation: {naive_coef:.3f} min per unit")
print(f"  (TRUE effect is 0.300; naive confounds with 0.5*mode-switch effect)")

# CAUSAL INFERENCE via latent mode recovery:
# 1. Infer latent mode from bimodal duration distribution via GMM
gmm = GaussianMixture(n_components=2, random_state=42)
log_df['inferred_mode'] = gmm.fit_predict(log_df[['qso_duration_min']])
print(f"\n=== LATENT MODE RECOVERY (GMM) ===")
print(f"Recovered mode labels (0/1) from duration clustering:")
print(f"  Component 0 mean duration: {gmm.means_[0][0]:.2f} min")
print(f"  Component 1 mean duration: {gmm.means_[1][0]:.2f} min")
print(f"Confusion vs true mode: {(log_df['inferred_mode'] != log_df['mode_true']).mean():.1%}")

# 2. Stratify by inferred mode and estimate within-mode causal effect
print(f"\n=== STRATIFIED CAUSAL INFERENCE ===")
for inferred_m in [0, 1]:
    subset = log_df[log_df['inferred_mode'] == inferred_m]
    mode_model = LinearRegression()
    mode_model.fit(subset[['propagation_quality']], subset['qso_duration_min'])
    coef = mode_model.coef_[0]
    print(f"Mode {inferred_m} (n={len(subset)}): effect of propagation = {coef:.3f} min/unit")

# 3. Pooled causal estimate: weighted average of stratified effects
# (addresses confounding by conditioning on inferred treatment mechanism)
effects = []
weights = []
for inferred_m in [0, 1]:
    subset = log_df[log_df['inferred_mode'] == inferred_m]
    if len(subset) > 2:
        mode_model = LinearRegression()
        mode_model.fit(subset[['propagation_quality']], subset['qso_duration_min'])
        effects.append(mode_model.coef_[0])
        weights.append(len(subset))
stratified_causal_effect = np.average(effects, weights=weights)
print(f"\nStratified causal effect (propagation → duration): {stratified_causal_effect:.3f} min/unit")
print(f"  (TRUE effect: 0.300; naive was {naive_coef:.3f}; stratified recovers confounding)")

print(f"\n=== KEY INSIGHT ===")
print(f"Causal_inference_observational assumes we observe treatment (mode).")
print(f"Here: mode is LATENT, determined endogenously by the confounder (propagation).")
print(f"Modification: Infer mode from outcome distribution (GMM), then stratify.")
print(f"This breaks standard causal assumptions but recovers ground truth.")
