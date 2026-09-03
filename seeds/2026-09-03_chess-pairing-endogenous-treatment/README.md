# Youth Chess Tournament Pairing Fairness via Endogenous Treatment Assignment

## Question

**Does unfair pairing (cross-skill-group matches) harm player outcomes in youth chess tournaments, or is the appearance of unfairness an artifact of rating-assignment bias that confounds the pairing algorithm itself?**

## Why This Is Non-Obvious

Standard causal inference (e.g., backdoor adjustment) assumes **exogenous** treatment assignment. But tournament pairing algorithms *generate their own treatment labels* based on observed (potentially biased) inputs. In this case:

- Younger players receive **inflated ratings** (true skill ≠ observed rating).
- The pairing algorithm uses **observed ratings** to decide treatment (fair vs. unfair pairing).
- The bias in ratings is a **confounder** that causally influences both the algorithm's decision and the outcome.
- Standard regression comparing pairings is **biased** because it doesn't account for this endogenous feedback loop.

The fix: **backdoor adjustment** must explicitly model the rating-assignment bias as a confounder *that the algorithm depends on*, not just as noise.

## Data

**Synthetic.** Designed to exhibit the structural problem:
- True skill (unobserved): normal(1500, 200)
- Rating bias: +150 for players age < 12 (inflated youth ratings)
- Observed rating: true_skill + bias + noise
- Pairing algorithm: sorts by observed rating, treats |true_skill_gap| > 300 as unfair (treatment=1)
- Outcome (win rate): improves if paired fairly, improves if player is strong

The signal is real: fair pairings *do* improve outcomes, but the naive regression misestimates the effect because it doesn't disentangle fairness from the rating bias that caused the unfair pairing.

## Limitation

This is proof-of-concept on synthetic data designed to expose the endogeneity problem. Real tournament data would require:
- Ground truth skill (e.g., international ratings or long-term historical performance)
- Explicit algorithm documentation
- Longitudinal pairing + outcome records

The technique (backdoor adjustment controlling for confounder that affects treatment assignment) is real; the dataset is constructed to make the causal structure visible.

## Key Output

Runs in ~1 second. Prints:
1. **Naive effect** (biased): treatment effect without confounding adjustment
2. **Backdoor-adjusted effect**: same regression but controlling for rating bias
3. **Stratified analysis**: effect within age subgroups to check confounder balance

Interpretation: If adjusted effect is larger (more negative for unfair pairings), endogenous rating bias was *suppressing* the true harmful effect of unfair play.

---

**Auto-generated seed.** Stdlib + pandas + numpy + scikit-learn. No external data or APIs.
