# Temporal Clustering Collapse in Community Theater Demand

## Question

Can clustering on *inter-purchase arrival times* (minutes between successive ticket purchases) recover latent audience-cohort segments in community theater box-office data when traditional categorical clustering (seat type, price tier) fails to expose behavioral structure?

## The Non-Obvious Angle

Community theater audiences have hidden temporal signatures:
- **Planners** buy in tight bursts 5–14 days before a show (families coordinating)
- **Walk-ups** arrive sporadically in the last 24 hours (individuals, no coordination)
- **Subscribers** renew in synchronized pulses at fixed days (7 and 14 days prior)

These are *not* captured by ticket category, price, or seat location. They are encoded in **when purchases cluster in time**. This forces the technique to work on a feature space (inter-arrival times) that is unusual for clustering and requires careful handling: the "features" are not attributes but *arrival microstructure*.

## Data

**Synthetic.** Generated to have the correct ground truth: three cohorts with distinct temporal signatures. Real community theater box-office data is rarely public and typically sparse for small venues. The synthetic data is structured to mirror realistic demand: bulk family purchases, last-minute walk-ups, and subscription renewal surges.

## Method

DBSCAN clustering on features:
- `hours_before_show`: absolute timing relative to show date
- `inter_arrival_min`: seconds between consecutive purchases
- `inter_arrival_lag_mean` / `inter_arrival_lag_std`: rolling window statistics to capture burst clustering

DBSCAN is chosen because temporal bursts create dense local regions (core points close in time), and sparse walk-ups are isolated—exactly the structure DBSCAN exploits.

## Limitation

This is a **proof-of-concept on synthetic data**. Real theater data would need:
1. Timestamped transaction logs (many venues only store daily summaries)
2. Enough transactions per show to recover patterns (small venues may have <50 purchases per show)
3. Validation against qualitative knowledge of audience composition

The clustering *will* recover the synthetic cohorts perfectly (by construction), but on real data, interpretation would require domain expertise to link clusters back to audience intent.

## How to Run

```bash
python main.py
```

Expected output: Adjusted Rand Index ~0.7+, cluster recovery of planners, walk-ups, and subscribers.

---

*Auto-generated seed. This repo is exploratory.*
