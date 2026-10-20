# QSO Mode-Switching Detection via Temporal Segmentation Collapse

## Question
Can we detect hidden operator mode-switching in amateur radio contest logs (from contest-rapid to ragchew-long-form) by clustering contact intervals and watching where segmentation breaks down—without explicit mode labels?

## Core Idea
Operators in contests shift between two unobserved behavioral modes:
- **CONTEST mode**: rapid-fire QSOs, short inter-contact intervals (seconds), high frequency density
- **RAGCHEW mode**: deep conversations, long intervals (minutes), lower density

Standard clustering assumes *stationarity*—that the data-generating process stays constant. When an operator switches modes, that assumption breaks: the temporal interval distribution suddenly changes. This causes the clustering segmentation to *fail*: the algorithm produces unstable, highly fragmented clusters instead of coherent regions.

**The technique is reversed**: instead of using clusters to understand the data, we use cluster *collapse* as a diagnostic signal for unobserved state changes.

## Method
1. Compute inter-contact intervals for each QSO
2. Extract rolling-window temporal features (mean, std, max, min, coefficient of variation of intervals)
3. Perform k-means clustering on these features
4. Count cluster transitions: how many times does a consecutive QSO pair belong to different clusters?
5. **Hypothesis**: High transitions → non-stationary operator behavior → hidden mode switches

## Data
**Synthetic.** Generated inline with:
- 200 simulated QSOs
- 4 explicit mode switches injected at t=300, 900, 1500, 2100 seconds
- Realistic interval distributions (5-20s for contest, 120-300s for ragchew)
- No mode labels provided to the algorithm

## Limitation
This is a proof-of-concept on synthetic data with *known* mode switches. Real contest logs would need:
- Actual timestamp data (we have synthetic)
- Validation that high clustering fragmentation correlates with human-annotated mode switches
- Tuning of the transition threshold and feature window size per operator and propagation conditions

The seam is real: standard clustering is not designed to *diagnose* process non-stationarity; we're repurposing its failure mode as a signal detector.

## How to Run
```bash
python main.py
```

Expected output: detection of 4 mode switches inferred from clustering collapse, without using ground-truth mode labels.

---
*Auto-generated seed. Question: can clustering breakage reverse-engineer unobserved categorical state?*
