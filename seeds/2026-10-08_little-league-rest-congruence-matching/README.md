# Little League Schedule Fairness via Reverse Bipartite Matching on Rest-Day Congruence

## Question
Can we detect whether a little league schedule is *structurally unfair* by treating the emergent clustering of teams' rest-day patterns as a signal that the schedule was constructed to artificially align certain teams' availability, rather than matching preferences or team requests?

## The Non-Obvious Angle
Traditional bipartite matching (e.g., for scheduling) works **forward**: given team preferences or constraints, compute a fair assignment. 

This project **reverses** it: given a *realized* schedule (games already assigned), we ask whether the emergent temporal structure (which teams share identical rest patterns) indicates the schedule was constructed unfairly.

**Key inversion:** The "matching recommendation" technique here must work *backward* from a realized bipartition (teams split into clusters of high rest-day congruence) into a fairness diagnosis, rather than forward from preferences into assignments.

## Data
**Synthetic.** We generate:
- 10 teams
- 8 weeks of games
- ~14 game slots per week at ~40% occupancy
- A schedule where games are randomly assigned

**Real-world proxy:** This structure mimics actual little league schedules (e.g., Cal Ripken, Babe Ruth leagues), where each team plays ~14 games across ~8 weeks, with multiple slots available per week.

## Method
1. Extract each team's rest-day binary vector (0 = plays in week, 1 = rests).
2. Compute pairwise rest-day congruence (normalized Hamming similarity).
3. Apply **Spectral Clustering** on the congruence matrix to detect hidden team groupings.
4. Compute intra- vs inter-cluster congruence.
5. **Diagnosis:** If intra-cluster congruence >> inter-cluster congruence, the schedule may have been constructed to deliberately align rest patterns—a fairness red flag.

## Limitation
This synthetic schedule is *random*, so no real anomaly emerges. A fair test would require:
- Real historical schedules from leagues known to be fair, and leagues with documented unfairness.
- Comparison against null-model schedules (random assignment under real constraints).
- Validation that rest-day clustering correlates with other fairness metrics (e.g., travel distance, opponent strength variance).

## Status
Auto-generated seed for exploratory work. Not a production fairness auditor.
