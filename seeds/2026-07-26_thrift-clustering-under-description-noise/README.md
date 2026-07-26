# Thrift Clustering Under Description Noise

## Question

**Can k-means clustering recover meaningful price tiers from thrift store items when the condition metadata is corrupted?**

Specifically: given price + a noisy proxy for condition (simulated OCR corruption of grade labels), does clustering produce clusters whose composition aligns with the true underlying condition grades—or does the noise defeat structure recovery?

## Why This Angle Is Non-Obvious

Standard clustering tutorials assume clean features. Thrift pricing is driven by **item condition**, but condition data is often sparse, subjective, or poorly digitized (handwritten tags, OCR errors, inconsistent grading across staff). This seed asks: **what does clustering do when the feature encoding that *should* drive the segmentation is deliberately corrupted?**

The non-obvious bit: clustering *doesn't require* that input features be clean in any absolute sense—only that they be *consistent* internally. So a corrupted text→signal conversion might still yield partial recoverable structure if the corruption is systematic. The project tests this edge case.

## Data

**Synthetic, generated inline.** 200 thrift items, true condition grades 1–5 (40 items each), with prices proportional to condition plus noise. Condition labels are then character-corrupted (simulating ~40% OCR/data-entry error rate) before being converted to a numeric signal via a heuristic (text length + vowel count).

### Why Synthetic?
Real thrift store data with ground-truth condition grades and OCR-corrupted descriptions is not readily public. The synthetic setup is designed to *match the structure* of the problem: noisy metadata, heterogeneous quality, a clusterable signal partially obscured by corruption.

## Limitation

**Honesty:** This is small-scale toy clustering. The real payoff would come from applying the same noise-stress-test to real transaction data (Goodwill API dumps, eBay thrift category sales, Salvation Army pricing histories—if available). The current version confirms the *principle* that partial structure survives noise but doesn't address:
- How much noise is recoverable in production data?
- How to *automatically* tune corruption tolerance?
- Whether other features (category, item type, material) add redundancy that helps clustering survive noise.

## Run

```bash
python main.py
```

Outputs purity score and a JSON file with results.

---

*This is an auto-generated seed from a data-analyst project farm. Clustering under feature noise is real, but this particular seed is exploratory and not yet production-grade.*