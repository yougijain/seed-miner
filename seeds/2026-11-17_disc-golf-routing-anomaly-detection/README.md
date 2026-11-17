# Disc Golf Routing Endogeneity & Anomaly Detection

## Question

Can we detect hidden player fatigue/injury by watching where anomaly detection breaks down when players endogenously choose easier course routings in response to unobserved physical decline?

## The Problem

Standard anomaly detection (e.g., isolation forest) on disc golf scores alone will miss deterioration patterns because:

- A fatigued/injured player **strategically switches to easier course routings**
- Easier routings → lower scores (better) **independent of skill decline**
- Raw score anomalies conflate "worse performance" with "different routing choice"
- The detection target (fatigue) **endogenously determines** the routing choice, invalidating the i.i.d assumption

## The Fix

Anomalies in **(routing_difficulty, score) pairs** expose the coupling:

- Healthy period: hard routing + good scores (normal cluster)
- Injury period: easy routing + mediocre scores (novel joint pattern)
- Routing-aware detector flags the **bivariate** anomaly that score-only detection misses

## Data

**Synthetic**, generated inline in `main.py`. Structure:
- 120 simulated rounds over ~4 months
- Rounds 0–80: healthy, mostly hard routings, strong scores
- Rounds 81–120: progressive fatigue, endogenous shift to easy routings, score degradation
- True signal: routing choice **causally driven by latent fatigue state**

## Limitation

This proof-of-concept assumes:
- Routing difficulty is a known, measurable feature (e.g., PDGA course rating or hole-by-hole distance)
- Fatigue is the only unobserved confounder (real life is messier)
- Linear relationship between routing and fatigue-induced score change

Real disc golf data would need explicit routing labels or derived difficulty scores (e.g., from course designer notes or historical player performance by layout).

## Expected Result

Routing-aware anomaly detection should achieve **higher precision** (fewer false positives) in surfacing the injury period compared to score-only detection, because it accounts for endogenous routing selection.

---

**Auto-generated seed.** This project is exploratory scratch work for a data-analyst seed farm.
