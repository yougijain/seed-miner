# Bee Colony Causal Inference Under Endogenous Inspection Timing

## Question

**Does high mite load causally degrade colony health, or do already-declining colonies trigger more frequent inspections that reveal their (pre-existing) high mite load?**

## The Non-Obvious Angle

Standard observational causal inference assumes treatment assignment is stable and exogenous. Here, the "treatment" (observing high mite load) is endogenous to the outcome (colony health):

- **True causal structure**: Colony weakness → more inspections → we observe mite load
- **Naive regression**: High mites ↔ Low health (confounded)
- **Problem**: Simply adding health as a covariate opens a collider path if we're not careful.

The seam is that **inspection timing itself is a confounder that depends on the outcome**. Standard regression fails. We must model the propensity to inspect *given health status* and adjust via stratification or IPW to break the confounding path.

## Data

**Synthetic.** 

Generated to encode the true data-generating process:
- Colonies have intrinsic decay rates (confound).
- Beekeeper inspects more often if colony looks sick (endogenous).
- Mite load is only observed upon inspection and correlates with colony health.
- Naive regression overestimates the mite → health effect due to confounding.

## Method

1. **Naive regression**: Regress health on mite load (biased).
2. **Stratified adjustment**: Stratify by propensity to inspect, estimate local treatment effects within strata.
3. **Inverse Probability Weighting**: Weight observations by 1/P(inspected | health) to recover the causal effect under the assumption of no unmeasured confounders.

## Limitation

- **No temporal structure** to isolate true vs. reverse causality; requires longitudinal experiment or IV.
- **Synthetic data**: Pattern is illustrative, not empirical.
- **Assumes no unmeasured confounders** (health and inspection propensity capture all confounding).

## Repo Info

Auto-generated seed. Runs in ~5 seconds. One file, ~140 lines, numpy + pandas + sklearn.
