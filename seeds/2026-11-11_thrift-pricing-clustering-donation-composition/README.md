# Thrift Store Price Clustering Under Donation Composition Shift

## Question

When thrift store donations shift from one source type to another (e.g., estate sales → yard-sale donors), do item-price clusters fragment into spurious segments? Specifically: if we apply K-means clustering trained on one donation regime to data from a *different* regime (without retraining), how much do item-cluster assignments destabilize, and can this instability signal hidden pricing-rule drift?

## The Non-Obvious Angle

Standard clustering diagnostics (silhouette, inertia) measure *quality*, not *regime stability*. This seed inverts the problem: clustering stability itself becomes a diagnostic signal. If the same items (same condition, similar price) cluster differently after a donation-source shift, that's evidence that:

1. An unobserved pricing rule (e.g., donation-day markdowns, seasonal inventory thresholds) has changed, OR
2. The distribution of items arriving has shifted, forcing the price space to reshape.

The technique challenge: standard K-means can't distinguish between genuine pricing-regime change and composition-driven artifact. Clustering *stability* under composition shift is the signal.

## Data

**Synthetic**: Generated inline in `main.py`. Two phases:
- Phase 1 (stable): 200 items from thrift-donor mix (mixed condition, lower prices, tight distribution).
- Phase 2 (shifted): 200 items from estate-sale donors (higher condition, higher prices, wider spread).

Prices are set by unobserved rule: `price = f(condition, category, donation_source)`.

## Method

1. Fit K-means (k=3) on Phase 1 features (condition + price).
2. Apply *same trained model* to Phase 2 (no retraining).
3. Retrain K-means on Phase 2 and compare cluster assignments.
4. Compute instability: fraction of items that switch clusters between "same model" and "retrained" predictions.

High instability → donation-source shift caused regime change, not random noise.

## Limitation

This is a proof-of-concept with synthetic data where the ground truth (donation source) is known and injected. Real thrift stores would require:
- Actual donation-source metadata (estate sales vs. individual donors vs. other chains).
- Pricing rule inference (e.g., via causal inference or regime-detection methods).
- Multiple stores/time periods to validate the signal.

The seed demonstrates the *structural* idea: clustering instability as a diagnostic, but causation (source shift → pricing regime) would need observational causal inference in a real deployment.

## Run

```bash
python main.py
```

Outputs cluster assignments, instability metric, and a 3-panel scatter plot (`clustering_instability.png`).

---

*Auto-generated seed. For exploration only.*
