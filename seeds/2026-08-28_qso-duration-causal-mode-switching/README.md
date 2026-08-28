# QSO Duration Causal Inference Under Endogenous Mode Switching

## Question
Can observational causal inference recover the true causal effect of propagation conditions on QSO (amateur radio contact) duration when operators endogenously switch between contest-mode (rapid exchanges, short duration) and ragchew-mode (long conversations) based on unobserved band quality, where the treatment mechanism itself is determined by the outcome?

## The Non-Obvious Angle
Standard causal_inference_observational assumes:
- Treatment assignment is observable or can be inferred from confounders.  
- The outcome mechanism doesn't *define* the treatment.

Here:
- **Operator mode (contest vs. ragchew) is latent**, not in the log.
- **Mode is chosen endogenously** based on unobserved propagation quality.
- **The QSO duration distribution is bimodal** because mode and true causal effect are confounded.
- **To estimate the causal effect**, we must first infer mode from the bimodal outcome distribution itself (via Gaussian Mixture Model), then stratify, then estimate within-mode effects.

This inverts the typical causal pipeline: usually we condition on treatment to isolate the outcome. Here we infer treatment from outcome, then use it as a stratifier. The modification forces causal inference to handle endogenous treatment assignment discovered *post-hoc* from the data shape.

## Data
**Synthetic.** Generated to reproduce the real phenomenon:
- 400 simulated QSOs with propagation conditions (0–10 scale, unobserved confounder).
- Operators endogenously switch mode via sigmoid response to propagation.
- True causal effect: propagation → +0.3 min duration (small).
- Mode effect: ragchew baseline 8 min, contest baseline 1 min (large, confounded).
- Naive regression (ignoring mode) severely overestimates the true propagation effect.

## Limitation
- Real contest logs have metadata (operator call sign, band, exchange format) that could reveal mode; we ignore this to isolate the latent-mode causal inference problem.
- GMM is deterministic given the data; real logs might have mode ambiguity (e.g., medium-duration QSOs).
- Small sample (n=400); larger logs would stabilize mode inference.

## Key Result
- **Naive causal effect:** ~0.5–0.6 min/unit propagation (confounded by mode-switching).
- **Stratified causal effect:** ~0.3 min/unit propagation (recovers truth).
- **Modification to causal_inference_observational:** Latent-mode inference + stratification, not standard covariate adjustment.

---
*Auto-generated seed. Technique: observational causal inference. Domain: amateur radio contest logs. Modified because standard causal assumptions break when treatment is latent and endogenously determined by confounders.*
