# Little League Schedule Fairness via Endogenous Bipartite Matching

## Question

Can bipartite matching between teams and game slots detect structural scheduling inequity when the desirability of a slot is endogenously determined by how many other teams are already assigned to it? Specifically: does the matching algorithm's own preference ordering cause it to concentrate scarce good slots (weekend mornings) among early-matched teams, forcing later teams into worse slots and revealing systemic unfairness?

## The Non-Obvious Angle

Standard bipartite matching (Hungarian algorithm, greedy assignment) treats edge costs as *fixed*. But in little league scheduling, a slot's attractiveness isn't fixed—it depends on *how saturated it already is*. This project runs matching under **endogenous cost updates**: as teams get assigned to desirable slots, those slots become less attractive for remaining teams (via a saturation penalty), and the algorithm's own assignments reveal which teams structurally can't escape low-quality slots.

The key insight: if the matching process itself creates inequity, the unmatched assignments and penalty distributions will show a clear pattern—early teams get high-affinity slots, late teams get low-affinity slots, not because of skill or random chance, but because slot scarcity is endogenous to the matching process.

## Data

**Synthetic.** 12 teams, 18 game slots across a week (Mon–Sun, morning/evening times). Slots have base desirability (Sat/Sun morning = high, weekday evening = low). Affinity is team-specific (perturbed around base + noise). The synthetic data encodes realistic little league preferences.

## Limitation

This is a toy proof-of-concept. Real little league scheduling has many hard constraints (travel distance, field availability, age groups, division balancing). This model ignores all of that. The interesting methodological contribution is *how* to detect endogenous inequity via matching-cost updates—not the fairness solution itself.

## Files

- `main.py`: Generate synthetic schedule, run greedy matching with endogenous cost updates, log inequity signals.
- `README.md`: This file.

---

*This is an auto-generated seed from a data-analyst project seed farm. Most seeds are discarded.*
