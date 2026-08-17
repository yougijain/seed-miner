# Youth Chess Pairings: Fairness via Constraint Violation Signals

## Research Question

Can a tournament scheduling optimizer, when forced to respect age-bracket homogeneity constraints, expose *latent imbalance* in cohort sizes through its constraint violation cost—revealing which youth brackets are structurally too small to schedule fairly?

## The Non-Obvious Angle

Standard tournament pairing optimization (Swiss system, round-robin) minimizes travel distance and board color balance. It does not optimize for *fairness across demographic cohorts*. When you layer in a fairness constraint—"maximize pairings within age brackets"—the optimizer's inability to satisfy that constraint becomes diagnostic. The *cost of infeasibility* (how many bracket-homogeneous pairs could not be formed) is a direct measure of cohort imbalance.

This flips optimization scheduling on its head: instead of "solve the scheduling problem," we use "why can't the scheduling problem be solved?" as a data analysis signal.

## Dataset

**Synthetic.** Generated inline in `main.py`:
- 40 youth chess players across three age brackets: U10 (8), U14 (18), U18 (14).
- Each player has a rating (skill level).
- U10 and U18 are undersized relative to round-robin pairing ideals.

**Why synthetic:** Real youth tournament data requires consent and anonymization. The synthetic dataset is *generated with the right structure*: two brackets that are deliberately too small to pair entirely within-age, creating the constraint violation signal.

## What the Code Does

1. **Standard Scheduling (Greedy):** Pairs players by rating proximity, ignoring age brackets. Produces valid pairings but may violate fairness.
2. **Constrained Scheduling:** Attempts to pair players within their age bracket. For brackets that are odd-sized or too small, this fails—players must be paired across brackets.
3. **Violation Reporting:** For each bracket, reports how many within-age pairs could not be formed. This is the fairness deficit.

## Limitation & Honesty

This is a *proof-of-concept*, not a production fair-scheduling system. Real constraints include:
- No two players can meet twice (round history).
- Color balance (white/black alternation).
- Bye (unpaired) player minimization.
- Actual rating-based pairing strength (Elo compatibility).

The code does not handle these; it demonstrates the *principle* that constraint violation cost can be a diagnostic.

## Output

The script prints:
- Player counts by age bracket.
- Standard (greedy) pairings and cross-bracket rate.
- Constrained pairings and fairness violations per bracket.
- Total fairness deficit.

## Conclusion

The modification to optimization scheduling is to *invert the objective*: instead of solving the scheduling problem, use the unsolvability of a fairness-constrained version as a cohort imbalance metric. The seam is that scheduling algorithms are not designed to diagnose structural imbalance; they are designed to find feasible solutions. Forcing infeasibility turns the algorithm into a diagnostic tool.
