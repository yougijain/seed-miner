# Little League Schedule Fairness via Endogenous Slot-Scarcity Bipartite Matching

## Question
**Can bipartite matching detect whether a little league schedule is unfair by treating "how many other teams already occupy a time slot" as endogenous cost, forcing the matcher to work backward from slot congestion into fairness diagnosis?**

## Why This Angle?

Standard bipartite matching assumes fixed, exogenous costs (e.g., "team A prefers Saturday"). This seed flips that: the **cost** of assigning a team to a slot emerges from the *current schedule itself* — slots with many teams already assigned become less desirable (scarce, crowded).

This forces matching to **reverse-engineer fairness violations**: by trying to load-balance teams across slots, the matcher exposes which teams are "stuck" in overloaded slots. This revealed constraint violation is the unfairness signal.

## Data

**Synthetic.** 10 teams, 4 game slots (2 days × 2 times). The "unfair" schedule loads 4 teams into slots 0 and 1, but only 1 team into slots 2 and 3. 

Run `main.py` to see:
- Current schedule's load variance
- Ideal (fair) schedule's load variance  
- Which teams are "stuck" in congested slots (fairness violation diagnosis)

## Limitation

This is a **toy proof-of-concept**. Real little league scheduling must handle:
- Team-pair conflicts (can't play each other twice in a row)
- Field capacities
- Coach availability
- Travel time
- Age/skill divisions

The seed only demonstrates the *matching diagnostic* — detecting unfairness through constraint-solver behavior. A production system would layer constraints on top.

## Real vs Synthetic

100% synthetic. No real little league data used. The signal (load imbalance → unfairness) is baked into the synthetic structure by design to prove the technique works.

---

*Auto-generated data-analyst project seed.*