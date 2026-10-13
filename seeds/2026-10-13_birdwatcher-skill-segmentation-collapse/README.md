# Birdwatcher Skill Inference via Sighting-List Segmentation Collapse

## Research Question

Can we detect when clustering-based skill tier segmentation *fails* among competitive birdwatchers, and what does that failure reveal about the geometry of skill-driven discovery patterns?

## The Non-Obvious Angle

Standard clustering works on **static feature vectors** (final species count, diversity, etc.). But competitive birdwatching skill manifests as **shape** in the cumulative discovery curve over time:

- **Novices**: Find common species first; discovery rate plateaus early (linear curve)
- **Intermediate**: Moderate acceleration on rare species; curve flattens mid-timeline
- **Experts**: Aggressively target rare species; discover rarest birds even late in outings (power-law curve with steep exponent)

This means traditional clustering on aggregate features can *collapse* — unable to distinguish intermediate from expert watchers — because the distinguishing information is **temporal trajectory geometry**, not static counts.

The seed forces clustering to confront this by deliberately extracting derived features (curvature, rarity bias) from cumulative discovery curves, then diagnosing when segmentation fails.

## Data: Real vs. Synthetic

**Synthetic** (generated inline in `main.py`):
- 50 birdwatchers assigned latent skill levels (0.3=novice, 0.6=intermediate, 0.9=expert)
- Each watcher has 12 outings; species discovered per outing follows a skill-weighted Poisson process
- Species rarity follows power-law distribution; skill level determines exponent (affects discovery curve shape)
- This structure ensures **real power-law geometry** so the segmentation signal is not noise

**Why synthetic?** Competitive birdwatching logs are not publicly aggregated with skill labels. Synthetic data lets us control ground truth and verify the technique detects skill-driven curve geometry.

## Limitation

1. **Feature engineering**: Curvature and rarity bias are hand-engineered. A more principled approach would fit power-law exponents directly to cumulative curves (e.g., nonlinear regression), then cluster on exponent estimates.
2. **Small scale**: 50 watchers is toy-sized. Real tournaments have hundreds; scaling would require robust exponent fitting and outlier rejection.
3. **Single country/region assumption**: Real birdwatching has strong regional species availability variation; skill inference would need geographic normalization.

## What You'll See

Run `python main.py`:

1. **Silhouette scores for k ∈ [2,7]**: Shows whether the data naturally partitions into skill tiers
2. **Cluster composition by true skill**: Reveals if clustering recovered ground-truth skill levels
3. **Segmentation collapse diagnosis**: Explains whether standard feature clustering succeeds or fails to distinguish expertise
4. **Visualization**: Scatter plots of (curvature, rarity_bias) colored by cluster vs. true skill

### Expected Output

If the power-law geometry is weak enough, clustering will collapse to k=2 (novice vs. rest), failing to separate intermediate from expert. If geometry is strong, k=3 will win. This itself is the finding: **can trajectory shape alone recover skill tiers without domain expertise labels?**

## Dependencies

- `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- No API keys, no external data, runs standalone

## Auto-Generated Seed Metadata

This is an automatically generated data-analyst project seed. It may be discarded or expanded into a full analysis.
