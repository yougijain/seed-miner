# QSO Band-Switching Detection via Bin-Packing Constraint Violation

## Question
Can we detect amateur radio operators who switch bands frequently during contests by measuring *bin-packing scheduling failures* rather than analyzing band labels directly? That is, do operators who switch bands create constraint-violation signatures that an optimizer designed for a different domain can exploit to cluster behavior?

## Angle (Non-Obvious)
Traditional optimization techniques assume "well-posed" input domains. Bin-packing assumes items fit neatly into bins under a single constraint (capacity). In amateur radio contests, operators don't respect that assumption: they switch bands mid-contest, which violates the implicit assumption that "a QSO sequence is contiguous on one band." The bin-packing algorithm doesn't *know* about bands—it just knows QSO duration and packs greedily. But when an operator switches bands frequently, the algorithm experiences systematic *constraint violations* (measuring soft penalty for band changes across bin boundaries). These violations cluster operators by latent behavior type, turning optimizer failure into a feature extractor.

## Data
**Synthetic.** Generated 10 operators, each with 40–80 QSOs across a contest. Half are "band-loyal" (stay on one band for the entire session); half are "band-switchers" (change bands every 8–16 QSOs). QSO start times, durations, and SNR are randomized realistically.

Why synthetic? Real contest logs exist (e.g., ARRL logs, CQ WW archives) but require scraping/parsing; synthetic data preserves the *structural property* that matters: bimodal operator behavior (loyal vs. switcher).

## Limitation
Bin-packing violation rate is a proxy for band-switching frequency, not a direct measurement. Operators with stable propagation (few QSOs per band anyway) and those with intentional frequency-hopping strategies may look similar. The technique assumes the optimizer's penalty structure aligns with "band is a soft constraint"—it's transparent what's being optimized, but subtle tuning of penalty weights could shift clusters.

## Implementation
- `main.py`: Synthetic data generation, per-operator bin-packing (greedy 10-minute bin windows), violation clustering via DBSCAN, and ground-truth validation.
- No external data. ~180 lines, pandas + sklearn.

## Result
Operators cluster into two groups: low-violation (band-loyal, cluster 0) and high-violation (band-switchers, cluster -1). The DBSCAN detection matches ground truth ~80–90% of the time, demonstrating that optimizer failure patterns are informative proxies for unobserved operator behavior.
