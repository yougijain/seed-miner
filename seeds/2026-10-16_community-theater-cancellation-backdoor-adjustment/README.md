# Community Theater Box Office: Show Attendance Confounder Recovery

## Research Question

Can observational causal inference recover the true causal effect of ticket cancellations on show attendance when cancellation *timing patterns* encode latent show quality (the true confounder)?

## Why This Is Non-Obvious

Standard causal_inference_observational assumes confounders are observed features in a flat table. Here, the confounder (show quality) is **latent** and must be inferred from the *temporal shape* of cancellation data—when cancellations cluster, not just how many occur.

### The Causal Story

1. **Latent confounder**: True show quality (unobserved).
2. **Effect on cancellations**: Low-quality shows trigger early, dense cancellation waves as audiences detect problems and exit preemptively.
3. **Effect on attendance**: Low-quality shows naturally have lower attendance.
4. **Naive analysis bias**: High cancellation rate looks like it *causes* low attendance, but both are actually caused by low show quality.
5. **Backdoor path**: Low quality → early cancellations AND low attendance.

### The Modification

To apply backdoor adjustment, we reconstruct the confounder using **cancellation timing clustering**: early/dense cancellation patterns signal low quality and must be stratified. Only then can we estimate the direct causal effect of cancellations on attendance.

## Data

**Synthetic.** 50 shows with:
- Latent true quality (0.3–0.95, hidden).
- Observed cancellation rate and timing (days before show).
- Observed show-day attendance.
- Treatment: binary high-cancellation indicator.

Low-quality shows have more cancellations clustered *earlier* (high cancellation_days_mean, low std). This timing pattern is the key to identifying the confounder.

## Output

```
Naive effect: ~-35 attendees (confounded)
Adjusted effect: ~-10 attendees (true direct effect)
```

The adjustment recovers true causal effect by stratifying on cancellation timing, which proxies for latent show quality.

## Limitation

In real data, you would need:
- Longer time series (multiple seasons) to validate stratification stability.
- External data (reviews, critic scores) to validate that timing clusters truly capture quality.
- Larger sample to ensure sufficient within-stratum variation.

This seed is a **proof-of-concept** that temporal structure in cancellation data can be exploited to infer and adjust for latent confounders in box-office causal inference.

---
*Auto-generated data-analyst project seed.*
