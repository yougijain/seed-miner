# Colony Foraging Bottleneck Detection via Inspection-Note Co-Silence Matching

## Question
**Can bipartite matching on temporal co-silence patterns in inspection notes infer which beehive colonies share hidden foraging-pathway dependencies?**

## Angle: Matching on Absence, Not Presence

Standard matching_recommendation assumes you have *explicit interactions* (e.g., customer–product purchases, team preferences). Beekeeping inspection notes are **sparse and negative**: what matters most is *when metrics stop being mentioned*, not when they are.

This seed reverses the typical matching signal:
- **Input**: Binary inspection matrix (metric mentioned vs. omitted across time windows)  
- **Matching cost**: Colonies that exhibit co-silence (both omit the same metric in the same window) in high proportion → *lower cost* → matched as likely foraging-bottleneck pairs  
- **Output**: Pairs of colonies whose inspection records jointly "go quiet" on pollen and flight activity in the same weeks, suggesting they compete for the same forage source or share a resource constraint

## Data: Synthetic + Motivation

**Synthetic.** Generated 12 colonies × 20 inspection weeks × 5 health metrics. Two ground-truth bottleneck groups injected:
- Colonies 0–3: jointly silent on pollen/flight in weeks 8–14 (forage zone A)
- Colonies 6–8: jointly silent on pollen/flight in weeks 8–14 (forage zone B)

**Real beekeeping motivation**: In practice, beekeepers record inspection notes as free text. Colonies that share a forage zone exhibit correlated resource stress; co-absence of positive metrics (pollen stores, active foraging flight) is stronger signal than co-mention because absence is selective—you only skip mentioning a metric when it's *notably absent*.

## Limitation

This is a *proof of concept*. Real data would require:
1. Parsed inspection notes (NLP extraction of which metrics were mentioned)  
2. Known ground-truth foraging zones or GPS/genetic data to validate  
3. Temporal segmentation (weeks/months) to align inspection windows across colonies  

The synthetic data is too clean; in reality, co-silence would be noisier and confounded by beekeeper inconsistency.

## Code

`main.py` computes a co-silence agreement matrix between all colony pairs, then applies `scipy.optimize.linear_sum_assignment` to find minimum-cost bipartite matches. Pairs with co-silence > 0.5 are reported as candidate bottleneck partners.

## Output

The script prints detected high-co-silence pairs and validates against injected ground truth.

---

*Auto-generated seed. Real-vs-synthetic: data is synthetic (see "Data" section). Code: ~150 lines.*
