# Disc Golf Routing-Invariant Skill Clustering via Layout-Variance Graphs

## Question
Can we recover true disc golfer skill tiers by building a network graph where edges connect players whose *score-variance response to course-layout changes* is similar, rather than connecting players with similar absolute scores?

## Why This Matters
Naive player clustering on raw scores is confounded: a player might score well on one layout not because of true skill, but because the routing (hole sequence) suits their throw style. To isolate true skill, we invert the graph construction: nodes are players, edges connect players whose *variance changes consistently* across different layouts. Players with similar skill tiers will have similar variance-response signatures because their skill premium or deficit applies uniformly regardless of routing geometry.

## The Non-Obvious Angle
Standard graph clustering treats node attributes (scores) as fixed features. Here, we must build edges based on *how nodes respond to structural changes in the graph's application context* (layout permutations). The graph is meta-structural: it captures players' resilience/stability to routing changes, which is a proxy for true skill. This forces community detection to invert: find clusters where variance-response is homogeneous, not where raw scores are proximal.

## Data: Synthetic
Synthetic, generated inline. Two course layouts, 24 players across 3 true skill tiers (beginner, intermediate, expert), 12 rounds per layout per player. Score variance is determined entirely by true skill + measurement noise, and is **independent of layout routing**. This ensures the signal exists: if clustering succeeds, it's because the layout-variance-response graph isolates true skill structure.

## Limitation
This is a proof-of-concept. Real disc golf data would need:
- Multiple courses (not just one course with two routing permutations)
- Hundreds of players and rounds to stabilize variance estimates
- Verification that layout difficulty is truly independent of skill distribution

The synthetic data is constructed such that skill matters uniformly across layouts; real data might have layout×skill interactions that confound the signal.

---

*Auto-generated seed. Not a tutorial or best practice.*
