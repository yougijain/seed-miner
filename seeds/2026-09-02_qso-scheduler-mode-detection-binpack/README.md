# QSO Scheduler: Detecting Mode-Switching via Bin-Packing Constraint Collapse

## Question
Can bin-packing optimization for staggered amateur radio contest time slots detect latent operator mode-switching (rapid-fire "contest-mode" vs. long "ragchew" conversations) when the optimization algorithm's constraint violations act as a proxy signal for hidden mixture structure in QSO duration distributions?

## The Non-Obvious Angle
Traditional bin-packing assumes item sizes follow a single (or known mixed) distribution. In amateur radio contests, QSO durations are *bimodal* and *endogenous*:
- Operators switch modes based on propagation conditions (unobserved).
- Contest-mode QSOs ≈30s (rapid exchanges).
- Ragchew-mode QSOs ≈5 min (long conversations).

The technique adaptation: standard bin-packing treats constraint violations (items that don't fit cleanly) as *failures*. Here, we treat them as a *clustering signal*—QSOs that force packing violations cluster around latent mode boundaries. By analyzing which durations cause slot overflow, we infer the hidden bimodal structure without explicit mode labels.

## Data
**Synthetic.** Generated inline in `main.py`:
- 300 QSOs with true hidden mode labels (70% contest, 30% ragchew).
- Durations drawn from mode-conditional normals:
  - Contest: N(35, 8) seconds.
  - Ragchew: N(300, 60) seconds.
- Bin-packing into 15-minute (900s) contest time slots.

## Limitation
This is a **proof-of-concept synthetic study**. Real contest logs would require:
- Actual propagation metadata (to validate that mode-switches correlate with band conditions).
- Multi-band or seasonal data (to confirm bimodality isn't an artifact).
- Domain expert labeling of true modes (to validate detection accuracy).

The constraint-violation clustering approach is *suggestive*, not diagnostic. It shows that packing pressure reveals structure; real application requires ground truth.

## How to Run
```bash
python main.py
```

Output:
- QSO log summary (duration statistics, true mode distribution).
- Bin-packing efficiency (slots needed, constraint violations).
- DBSCAN clustering on (duration, violation_flag).
- Per-cluster mode composition and purity (homogeneity of detected clusters).

## Key Result
If the detected clusters separate true contest/ragchew modes with >0.75 purity, the hypothesis holds: bin-packing constraint violations correlate with latent mode boundaries.

---
*This is an auto-generated data-analyst project seed. It is exploratory scratch work.*
