# Disc Golf Round Anomalies via Temporal Clustering Collapse

## Question
Can anomaly detection recover anomalous disc golf rounds when the ground truth is *sequence structure* (learned transitions between consecutive holes) rather than marginal score outliers?

## Why Non-Obvious
Standard anomaly detectors (Isolation Forest, LOF, etc.) treat each round as a point in feature space (e.g., 18 hole scores → 18D vector) and flag outliers by statistical deviation. **This misses the core structure of golf**: each hole's difficulty and player fatigue condition the next hole—rounds are *paths through state-space*, not points. A round can have normal marginal scores but anomalous *transitions*.

This seed forces anomaly detection to work on derived features that capture temporal structure (transition-matrix co-occurrence patterns) rather than raw observations. The technique doesn't naturally fit the domain, and that mismatch is the point.

## The Modification
- **Standard anomaly detection**: flag rounds where individual hole scores deviate from population mean.
- **This approach**: quantize score deltas from par into buckets, compute (state_t, state_t+1) transition pairs for each round, treat transitions as a 10D feature vector, run Isolation Forest on the *transition distribution* rather than the raw scores.

## Data
- **Synthetic**: 190 normal rounds (consistent skill ~par -1.5, fatigue ramp on back nine) + 5 injected anomalies (sudden +3-4 per hole after hole 9-11, simulating mental collapse).
- **Why synthetic**: real disc golf score data is sparse, and we need control over anomaly mechanism to validate that the transition-matrix approach actually recovers path-dependent anomalies.
- **Limitation**: proves the concept on one anomaly type (collapse after a hole). Real anomalies might include early fatigue, inconsistent form, or equipment issues that show different signatures.

## Files
- `main.py`: generates synthetic rounds, computes transition-matrix features, runs Isolation Forest on transitions vs. raw scores, reports confusion matrix and precision/recall.

## Expected Outcome
The transition-matrix approach should outperform raw-score anomaly detection on detecting mental-collapse anomalies because transitions capture the *timing and sequence* of degradation. Raw-score detection will likely flag individual bad holes scattered throughout normal rounds instead of catching the coordinated collapse.

---
*This is an auto-generated seed from a domain×technique matrix.*
