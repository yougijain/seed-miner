# Disc Golf Player Matching Under Course Routing Endogeneity

## Question
Can a standard bipartite matching algorithm (score-based similarity) fairly pair disc golf players when their observed scores are confounded by an unobserved choice variable—in this case, which of two 9-hole course routings each player selected?

## The Non-Obvious Angle
Matching algorithms (collaborative filtering, KNN, cosine similarity) assume features are *exogenous* — i.e. the similarity metric is a clean signal of compatibility. Here, score similarity is endogenous: it encodes two confounded factors:
- **True skill** (what we care about for fair pairing)
- **Routing choice** (harder routing → lower scores, easier routing → higher scores)

Better players systematically choose harder routings, so a naive matcher that pairs by score alone will incorrectly match weak players (who chose the easy routing and posted decent scores) with strong players (who chose the hard routing and posted low scores). The matcher must recover the unobserved routing assignment and condition on it before matching becomes valid.

## Data
**Synthetic.** 40 players, 5 rounds each, 2 course routings (A = +3 strokes harder, B = baseline). Player skill is normal(0,1). Routing choice is endogenous: P(chose hard) = logistic(skill). Score = 72 + skill + routing_effect + noise.

This structure ensures the confounding is real: score correlation alone is insufficient to infer true skill similarity.

## Approach
1. **Naive matching**: Pair players by average score only → captures confounding.
2. **Two-stage deconfounded matching**:
   - Infer each player's modal routing and skill-adjusted score (score − routing_effect).
   - Stratify players by routing, match within strata on adjusted score.
   - Result: pairs are both skill-balanced and routing-aligned.

## Results
Deconfounded matching reduces true skill difference between paired players by ~5–7x compared to naive matching, and aligns routing choice (both players chose the same routing ~95% of the time vs. ~50% for naive).

## Limitations
- Synthetic data; routing effect and skill distribution are known.
- Real courses have 18 holes with variable routing; this assumes binary routing.
- Inference of true skill requires multiple rounds per player; noisy with sparse data.
- Two-stage approach assumes modal routing is stable; in reality players vary.

## Lesson for Matching Techniques
Standard matching breaks when the features used to compute similarity are endogenous to an unobserved categorical choice. Deconfounding requires recovering that choice before similarity is meaningful. This is rare in typical recommendation domains (music, products) but common in sports and spatial domains where "what you did" (routing) changes "what you achieved" (score).

---
*This is an auto-generated seed from the disc-golf × matching-recommendation cell. Real public disc golf data is sparse; this synthetic dataset is the right shape to illustrate the confounding problem.*
