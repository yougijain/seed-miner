# Disc Golf Course Layout Recovery from Player Clustering Collapse

## Question

Can we infer the hidden dependency structure of a disc golf course (which holes create bottlenecks that condition downstream scores) by observing how player score-sequence clustering **degrades** as the distance threshold is tightened?

## The Non-Obvious Angle

Standard graph clustering assumes static, explicit edges. Here, **edges are implicit in score sequences**, and they're **conditioned on player skill and prior hole outcomes**. When we vary the clustering threshold (epsilon), clusters don't just shrink uniformly—they fragment in a way that reveals which holes behave as network pivots:

- **Low epsilon (tight clustering):** All rounds cluster by overall skill.
- **Intermediate epsilon:** Rounds separate by bottleneck outcomes (hole 3, 6), because these holes' variance propagates downstream.
- **High epsilon (loose clustering):** Rounds cluster by skill again (noisy holes matter less).

The **fragmentation pattern**—where clusters suddenly split—encodes the course's causal structure. Bottleneck holes create conditional dependencies that make score sequences more or less similar depending on what happened upstream.

## Data

**Synthetic.** The course has 9 holes; holes 3 and 6 are designated bottlenecks:
- Bottleneck holes: high variance, score depends on prior hole outcome.
- Regular holes: low variance, independent of neighbors.
- Player skill modulates noise magnitude across all holes.

This structure is **intentional** so we can verify the hypothesis: clustering collapse should reveal holes 3 and 6 as structural pivots.

## Limitation

The synthetic data is hand-crafted to exhibit the signal. Real disc golf data would need:
- Hundreds of rounds per course.
- Accurate scoring and player metadata.
- Confirmation that bottlenecks are causal (not confounded by hole difficulty or player routing).

The core insight—that **clustering degeneracy reveals network structure**—is real and transferable, but applying it to real data would require validation against course designer intent or expert opinion on hole dependencies.

## Findings

Run `python main.py`. Output includes:
- Cluster count at each epsilon threshold.
- Fragmentation points (where the number of clusters suddenly changes).
- Hole-by-hole score variance.
- Hypothesis confirmation: bottleneck holes have higher variance and drive clustering collapse.

## Auto-Generated Seed

This project is an auto-generated data-analyst seed from a domain × technique matrix. It prioritizes finding a non-obvious application of graph-network analysis to disc golf, not a polished analysis.
