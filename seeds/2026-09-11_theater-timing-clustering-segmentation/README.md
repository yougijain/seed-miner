# Community Theater Audience Segmentation via Temporal Matching

## Question
Can temporal bipartite matching (matching purchases by inter-purchase interval similarity) recover latent audience segments when categorical features (show type, price, seat section) fail to align with true behavioral groups?

## The Non-Obvious Angle
Traditional recommendation/matching systems pair items by *what* they are (metadata, category, features). But in community theater box-office data, audience *behavior* encodes preference in *when* people buy: season subscribers have regular 14-day intervals, weekend-only patrons cluster on 7-day boundaries, walk-ins appear random. **Applying bipartite matching to temporal intervals instead of categorical features forces the algorithm to work on a feature space the technique wasn't designed for**—but that feature space is where the domain's real signal lives.

## Method
1. Generate synthetic ticket purchase data with three latent audience segments (walk-ins, season subscribers, weekend-only), each with distinct inter-purchase interval distributions.
2. Intentionally add categorical noise (random show types and prices that don't align with segments).
3. Build two similarity matrices:
   - **Categorical**: matching purchases by show type + seat section overlap (traditional approach).
   - **Temporal**: matching purchases by inter-purchase interval features (arrival-time affinity).
4. Apply DBSCAN clustering on temporal features and greedy bipartite matching on both matrices.
5. Compare: do temporal matches recover the ground-truth segments better than categorical matches?

## Real vs. Synthetic
**Entirely synthetic.** True community theater transaction data is proprietary and proprietary. The synthetic data is *structurally* realistic: real theaters do see subscriber patterns (regular intervals), walk-in noise (random), and event-driven surges (weekends). The clustering signal is real in the data—you can run the script and observe that temporal intervals reveal segments categorical features cannot.

## Limitation
The synthetic data is small (200 purchases) and the segments are exaggerated for clarity. Real theater data would be messier (holidays, weather, competing events, marketing campaigns all shift intervals). The proof-of-concept here is **conceptual**: that reformulating matching_recommendation to operate on temporal intervals is meaningful and recovers structure. A production system would need real transaction logs, domain validation, and more sophisticated temporal alignment (e.g., aligning to fiscal/seasonal calendar rather than raw days).

## How to Run
```bash
python main.py
```

Expect output showing that temporal clustering detects ~3 clusters with distinct interval patterns, while categorical matching produces generic show-type pairings.

---
*This is an auto-generated seed. Not production code.*
