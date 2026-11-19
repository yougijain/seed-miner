# Birdwatcher Skill Clustering via Rarity-Discovery Curve Shape

## Question

Can K-means clustering segment competitive birdwatchers into skill tiers by clustering on the *curvature and saturation properties* of their cumulative species discovery lists, rather than raw species counts—and does this reveal that experienced watchers follow power-law saturation curves while novices accumulate linearly?

## The Non-Obvious Angle

Standard clustering treats all features as comparable distributions in feature space. But birdwatcher sighting trajectories have fundamentally different *statistical shapes*:

- **Novices**: Linear accumulation (nearly constant weekly discovery rate)
- **Experts**: Power-law saturation (fast early, then plateau as remaining species become rare)

Raw species counts alone are insufficient—you must extract **curve-shape features** (log-domain slope, saturation timing, curvature) before K-means can partition them. This forces the technique to invert from "cluster on raw magnitudes" to "cluster on the *structure* of the discovery process."

## Data

**Synthetic.** 20 birdwatchers over 365 days:
- 10 novices: linear trajectories with slope 0.08–0.12 species/day
- 10 experts: power-law curves with exponent 0.55–0.70

Synthetic because real competitive birdwatching sighting records are fragmented across private apps and local networks. The structure (linear vs. power-law) is realistic: novices learn common species at steady rates; experts must work harder for rare, geographically-specific taxa.

## Results

The script:
1. Extracts four shape-based features from each watcher's cumulative curve
2. Normalizes and clusters with K-means (k=2)
3. Prints cluster profiles and visualizes sighting curves + feature space

**Expected outcome**: Clean separation of novices (high slope, early saturation) from experts (low slope, late saturation).

## Limitation

This proof-of-concept assumes two distinct skill tiers. Real competitive birdwatching likely has more granular gradations. Also, the synthetic data generates *perfect* separation—real data would have overlapping curves and require tuning of feature weights.

---

*Auto-generated data-analyst project seed. See `main.py` for full pipeline.*
