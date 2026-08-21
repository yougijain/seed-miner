# Thrift Price-Condition Graph Clustering

## Question

**Can bipartite item-condition networks using label-propagation graph clustering recover latent thrift item categories when condition labels are noisy, whereas traditional clustering fails silently?**

## Why This Matters (Non-Obvious Angle)

Thrift store pricing is driven by item *category* (electronics, clothing, furniture, etc.), not by explicit labels. In practice, donor-supplied or staff-entered condition descriptions are often incomplete or corrupted (OCR errors, abbreviations, typos, subjective terms). 

Standard approaches:
- **k-means on price + condition features**: Silently produces worse clusters under corruption; degradation is hard to detect without ground truth.
- **Naive label clustering**: Treats condition as a discrete category; doesn't leverage price or donor patterns.

**Graph-based label propagation** (a semi-supervised technique) treats the bipartite item-condition network as a signal: if items cluster by price and condition *neighbors*, then the propagated labels reflect consensus within local structure. **When condition labels are noisy, graph propagation surfaces disagreement structurally** (neighbors disagree → label uncertainty), rather than averaging away the signal.

## What the Code Does

1. **Synthetic dataset**: 120 items across 3 true categories (Electronics, Clothing, Furniture) with ground-truth price ranges and appropriate condition distributions.
2. **Corruption**: Intentionally flip ~30% of condition labels to simulate real-world label noise.
3. **Bipartite graph**: Connect items if they share observed condition AND have similar prices (within 40%). Build item-item network.
4. **Label propagation**: Run 5 iterations of majority-label propagation on item nodes.
5. **Evaluation**: Measure purity—what fraction of items in each propagated label cluster belong to the same true category?

## Results

Expected: Propagated labels should show **moderate to high purity** (~0.6–0.8) because the network topology encodes true category structure even under label corruption. Items of the same true category cluster by price and condition neighbor agreement.

## Real vs. Synthetic

- **Synthetic**: True item categories, corruption process, and prices are simulated.
- **Real question**: In actual thrift stores, can we recover category structure from noisy mixed-source condition labels using graph methods?
- **Limitation**: This seed does not validate against *actual* thrift store data. The technique is only as useful as the correlation between true categories and price + condition neighborhoods in the real domain.

## Dataset & Code

- **No external data**: Fully synthetic, deterministic.
- **Dependencies**: `pandas`, `numpy`, `networkx`.
- **Runtime**: <1 second.

---

*This is an auto-generated seed from a data-analyst project farm. It is exploratory and unvetted.*
