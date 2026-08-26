# Vendor Temporal Clustering: Detecting Market-Day Synchronized Behavior

## Question

Can clustering on vendor hourly revenue *timing patterns* detect latent vendor segments when traditional category-based clustering fails, because the market-day domain encodes behavior in *when* sales concentrate, not *what* sells?

## Core Insight (Non-Obvious Angle)

Most vendor segmentation clusters by **item category** (produce, prepared foods, flowers). This seed instead clusters by **temporal features extracted from hourly revenue**: peak hour, morning/lunch/evening revenue proportion, revenue concentration (Gini coefficient).

The seam is: farmers markets have strong **crowd-flow dynamics** (morning rush, lunch slump, after-work return). Vendors exploit these differently:
- **Morning peakers** (coffee, donuts) concentrate sales 9–11am
- **Lunch peakers** (prepared foods) concentrate sales 12–1pm  
- **Evening peakers** (flowers, take-home produce) concentrate sales 4–5pm

These segments are **invisible to category-based clustering** but emerge naturally when the feature space is temporal patterns. The technique forces the clustering algorithm to work on a domain dimension (time-of-day synchrony) that doesn't exist in typical vendor datasets.

## Data

**Synthetic.** 30 vendors, 8-hour market day (9am–4pm), three ground-truth temporal segments (morning/lunch/evening peak).
- Why synthetic: real farmers market hourly data is not publicly available at this granularity.
- Why the structure is valid: farmers markets do exhibit strong diurnal crowd flows; vendor revenue timing is plausibly correlated with product type (though the causality varies).

## Limitation

This seed is **proof-of-concept only**. Real validation would require:
1. Actual hourly point-of-sale data from a farmers market (not available in this repo).
2. Confirmation that temporal clusters predict behavior change (e.g., vendor churn, inventory patterns, booth relocation decisions) better than categorical clustering.
3. Exploration of whether temporal segments interact with weather, holidays, or market-day scheduling in ways category clustering misses.

Without real data, we cannot distinguish whether temporal clustering is *genuinely useful* or merely a different way to express the same category structure.

## Files

- `main.py`: Generates synthetic vendor hourly revenue, extracts temporal features, applies k-means, visualizes PCA projection and average patterns by cluster.
- `temporal_clustering.png`: Output visualization.

## Running

```bash
python main.py
```

## Self-Assessment

This is a real seam in the clustering-segmentation domain. Standard vendor clustering doesn't use temporal features because vendor data is usually aggregated (total revenue per market day, category mix). Forcing the technique to use hourly patterns *changes the feature space fundamentally*. The modification is small (extract temporal stats from hourly revenue) but the conceptual shift is genuine: clustering market-day behavior instead of product type. Whether it's *useful* depends on real data, which is the honest limitation.

---

*Auto-generated seed for farmers_market_vendor_sales + clustering_segmentation.*
