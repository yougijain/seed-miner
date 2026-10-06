# Disc Golf Skill Regression via Course-Routing Confounding

## Question

When a disc golfer's scores decline over a season, is this **real skill loss** or confounded by their **endogenous choice of easier routing**? Courses with multiple layouts let players choose harder (A) or easier (B) routing; if a player chooses B more often as fatigue sets in, naive comparison will falsely attribute the score decline to routing choice rather than fatigue.

## Technique Angle: Backdoor Adjustment Under Unobserved Confounder

Standard causal inference assumes **treatment is exogenous** (randomly assigned). But routing choice is **endogenous**: players with low confidence/fatigue choose easier routing AND score worse, creating a confounded backdoor path:

```
Unobserved Confidence → Routing Choice → Score
Unobserved Confidence → Score (direct effect)
```

Backdoor adjustment tries to block this path by **stratifying on a proxy for confidence** (e.g., baseline score or recent performance), forcing the technique to recover the true causal effect of routing from observational data where confounding is implicit in behavior.

## Data: Synthetic

Generated inline. Each round has:
- `routing`: 0 (hard A) or 1 (easy B), chosen endogenously based on unobserved confidence.
- `score`: affected by routing difficulty, unobserved confidence, and noise.
- Ground truth: routing B saves ~5 strokes vs A, but naive regression conflates this with confounding.

## Limitation

Backdoor adjustment via stratification is only valid if:
1. The proxy (score quantile) adequately captures the confounder.
2. No unobserved confounders remain.
3. No post-treatment bias (confidence measured after routing choice).

In real data, these are unverifiable; this seed demonstrates the *method*, not proof of causality.

---

*Auto-generated seed. Concept: @llm. Causal inference modification required because typical observational causal methods assume exogenous treatment; disc golf routing is endogenous to player state.*
