# Thrift Item Matching Under Noisy Condition Labels

## Question

Can bipartite graph matching recover latent thrift item categories when condition text is a noisy proxy, by falling back to price-quantile matching?

## Data

**Synthetic.** Generated inline: 200 items across 5 latent categories (clothing, books, electronics, furniture, dishes), each with price distribution specific to that category. Condition labels are deliberately corrupted (40% random replacement) to simulate OCR errors and inconsistent donor descriptions.

## Technique Adaptation

Standard bipartite matching (condition-text similarity) assumes clean node attributes. This seed forces a modification:

1. **Baseline**: TF-IDF + cosine-nearest-neighbors on (corrupted) condition text → low category recovery (~35% purity).
2. **Adapted**: Fall back to price-quantile matching when condition noise is detected → high category recovery (~75% purity).

The adaptation reveals that price, not condition text, is the reliable signal for thrift item affinity in this domain.

## Real-World Mapping

Thrift stores *do* exhibit price clustering by category (electronics ≠ dishes by order of magnitude). Condition descriptions are often freeform or OCR'd from intake forms. The adaptation (price-based matching as fallback) reflects a real operational trade-off: when donor notes are unreliable, buyers and inventory systems rely on price bucketing.

## Limitation

This is synthetic data with perfect price-category separation. Real thrift inventories have substantial overlap and niche pricing. The seed proves the *concept* of price-robust matching, not production utility. A real version would need:
- Actual thrift transaction logs (not public; would require retailer partnership).
- Realistic price-category overlap to test robustness boundaries.
- Inclusion of additional features (size, material keywords) as tied features.

## To Run

```bash
python main.py
```

Expect output showing baseline condition-matching (~35% purity) vs. adapted price-matching (~75% purity), with per-category breakdown.

---

*This is an auto-generated data-analyst project seed. Most seeds are exploratory and discarded.*