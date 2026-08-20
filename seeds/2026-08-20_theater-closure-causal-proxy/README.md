# Theater Closure Causality: Demand Shock vs. Substitution

## Question

When a community theater temporarily closes for maintenance, do nearby theaters' ticket sales increase because audience *demand shifted* (substitution), or is the sales bump spurious? Can we use staggered natural experiments + difference-in-differences on *neighbors* (proxy estimator) to isolate the causal spillover effect?

## Angle: Why This Breaks Standard Causal Inference

Standard DiD or instrumental-variable designs assume you observe the treated unit both before *and* after treatment. But here:
- **Treated unit vanishes:** When theater A closes, its own sales are undefined (0 isn't sales; it's absence).
- **Treatment is sparse & ragged:** Only 2 of 6 theaters close, on different weeks.
- **Spillover is bidirectional:** Closed theater's demand might leak to neighbors (substitution) *or* might disappear entirely (demand destruction).
- **Causal direction is backward:** We can't observe whether demand collapsed directly; we infer it from neighbors' sales *rising*.

Standard causal inference (DiD, IV, doubly-robust) assume you can measure the outcome for all units. This seed forces the estimator to use *proxy outcomes* (neighbors' sales) to indirectly test causality on the untreated group.

## Data

**Synthetic.** Generated inline with:
- 6 theaters, 12 weeks.
- Staggered 1-week maintenance closures (theater A week 3, theater D week 7).
- **True causal effect:** Closed theater's demand substitutes to neighbors (+$40 per show post-closure).
- **Noise:** Random weekly variation ± $10–20.

## Limitation

1. **Parallel trends:** Assumes treated & control neighbors would move together absent closure. Not verified; assume exogenous scheduling.
2. **Spillover localization:** Neighbors defined by adjacency in synthetic data. Real theater networks might be non-geographic (genre affinity).
3. **Single-dose:** Only 2 closures. Real causal inference would need many staggered events.
4. **No competing outcomes:** Model doesn't distinguish postponed shows (future substitution) from lost shows (demand destruction).

## Running

```bash
python main.py
```

Output: DiD estimate of spillover effect + robustness check (parallel trends test).

---

*This is an auto-generated data-analyst seed. Do not expect production-grade causal assumptions.*
