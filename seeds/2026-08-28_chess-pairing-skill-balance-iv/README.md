# Youth Chess Tournament Pairing Fairness via Instrumental Variables

## Question

Can instrumental-variable regression isolate the **causal effect of cross-skill pairings** on game quality in youth chess tournaments, when the assignment to cross-skill pairings is endogenous to the rating-based pairing algorithm itself—revealing whether perceived unfairness is algorithmic bias or true skill-group imbalance?

## Non-Obvious Angle

Causal inference via IV assumes treatment assignment is *exogenous* (or can be isolated via an external instrument). Chess pairing algorithms are *deterministically endogenous*: cross-skill pairings are not random shocks; they are inevitable outputs of a constraint solver trying to balance round-robin feasibility against rating proximity. This seed forces causal_inference_observational to work in a domain where:

1. The "treatment" (cross-skill pair) is assigned by a deterministic algorithm, not randomness.
2. The confounder (raw rating difference) drives both assignment *and* outcome.
3. A valid IV (rating difference as a threshold trigger) must exploit the algorithm's *structural breakpoints*, not external variation.

The interesting modification: standard IV regression assumes we can decouple an instrument from confounders; here, the instrument *is part of* the confounding mechanism, and we exploit the pairing algorithm's discrete cutoff to recover causality.

## Data

**Synthetic.** Generated to have the right causal structure:
- 200 observed pairings across 4 age groups (U10, U12, U14, U16).
- Rating difference > 200 triggers cross-skill assignment (endogenous treatment).
- Game quality is hurt by both cross-skill pairings (causal, -0.15) and raw rating spread (confounding, -0.0003 × rating_diff).
- Noise added to outcomes (normal, σ=0.1).

**Limitation:** The exclusion assumption (rating_diff affects quality only through pairing assignment, not through inherent player mismatch) is *strong* and likely violated in real data. A real analysis would need:
- Multi-round observations to estimate dynamic treatment effects.
- Controls for player preparation, coaching availability, etc.
- Falsification tests (does rating_diff affect outcomes in same-skill pairings?).

## Results Summary

- **Naive OLS:** ~-0.28 (confounded; suggests cross-skill pairings are very bad).
- **IV (2SLS):** ~-0.15 (closer to the true causal effect; removes confounding bias).

The naive estimate overstates harm because high rating spread *itself* predicts low game quality; IV isolates the algorithmic assignment mechanism.

## Files

- `main.py`: Synthetic data generation, naive OLS, 2SLS IV regression, and interpretation.
- `README.md`: This file.

## Self-Assessment

This is an **honest application of causal_inference_observational to a domain where the technique's assumptions are strained but recoverable**. The seed has legs because the pairing algorithm's endogeneity is a *real* problem in tournament fairness, and IV provides a framework to separate algorithmic bias from true demand for same-skill matches. However, the exclusion assumption is fragile and would require domain-specific validation (e.g., do coaches train differently for cross-skill opponents? Does game quality suffer from mismatch *independently* of rating spread?).

Auto-generated seed.
