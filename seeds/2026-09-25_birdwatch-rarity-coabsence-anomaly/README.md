# Birdwatching Rarity Detection via Temporal Co-Absence Anomaly Detection

## Research Question
Can anomaly detection applied to *temporal co-absence patterns* in sparse sighting logs infer which species share rare habitats, when co-occurrence data is too sparse to cluster on presence alone?

## The Non-Obvious Angle
Standard anomaly detection works on observed features (presence, counts, attributes). Here, the signal is in **silence**: species that share a rare habitat are jointly absent from most observations. We convert temporal co-absence into a feature (co-absence rate during days when at least one species is missing) and treat *high*, consistent co-absence* as anomalous—inverting the normal presence→covariance logic.

## Data: Real vs. Synthetic
**Synthetic.** The dataset is generated inline with:
- 90-day observation window
- ~20 sighting events (sparse, as in real competitive birdwatching logs)
- 10 species with baseline 60–80% detection per day
- Injected ecological pairing: Heron and Egret show 30% coordinated co-absence/co-presence (wetland habitat)

This structure ensures the anomaly signal actually exists in the data.

## Method
1. Convert sighting log to species-presence binary matrix (90 days × 10 species).
2. For each species pair, compute features: days both absent, co-absence rate (during any-absence days), days both present, days of discord.
3. Apply Isolation Forest to detect pairs with anomalously high co-absence structure.
4. Rank pairs by anomaly score; top pairs are candidates for shared rare habitat.

## Key Limitation
**Causality ambiguity:** High co-absence could mean shared rare habitat (ecological signal) *or* just low-detection species that are individually rare. To disambiguate, you'd need independent habitat/detection-probability data. This seed detects the pattern; attributing it requires domain knowledge.

## Expected Result
Heron-Egret should rank high in anomaly scores because they are engineered to co-absent/co-present. Other pairs show normal independent absence patterns.

## Running
```bash
python main.py
```

Output: ranked species pairs with anomaly scores and co-absence statistics.

---
*Auto-generated project seed. Not a polished analysis. Meant to explore whether anomaly detection can be repurposed for temporal co-silence inference.*
