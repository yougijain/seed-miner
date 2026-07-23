# Farmers Market Vendor Adjacency & Causal Inference

## Question
**What is the causal effect of being a weekend day (vs. weekday) on vendor sales, accounting for spatial confounding (location-based foot traffic and adjacency spillover)?**

## The Non-Obvious Angle
Standard observational causal inference assumes **independent units** (e.g., individuals in a study). Vendor sales in a farmers market violate this: vendors are arranged in a fixed spatial layout, creating **adjacency spillover** (your sales depend partly on your neighbors). 

When you apply backdoor adjustment (the standard technique) to spatially-dependent data, you must:
1. Recognize the spatial confounding graph (location → foot traffic → sales)
2. Stratify by location tier (a crude spatial blocking)
3. Accept that pooling strata may reintroduce bias if location is truly a collider

**The seam**: causal_inference_observational was built for tabular, independent data. Applying it to spatial networks exposes the brittleness of covariate adjustment and why stratification is a partial fix at best.

## Data
**Synthetic, inline-generated.** 
- 8 vendor stalls in a 2×4 grid.
- 14 observed market days (2 weeks of Sat/Sun).
- Sales depend on: day-of-week (treatment), fixed location, adjacency spillover, random noise.
- Ground truth: weekend drives ~30% higher foot traffic, which (by design) increases sales uniformly—but location confounds the measurement.

## Limitation
With observational data alone, we **cannot** cleanly separate the causal effect of "day" from location effects when location is spatially embedded. A true causal estimate would require:
- **Natural experiment** (e.g., weather variation affecting traffic differently by location)
- **RCT** (random vendor rotation, infeasible in practice)
- **Instrumental variable** (no plausible IV in this domain)

This project demonstrates why.

## Running
```bash
python main.py
```

Output shows naive, adjusted, and stratified causal estimates, illustrating how spatial confounding resists standard adjustment.

---
*Auto-generated seed. This is early-stage exploratory code.*
