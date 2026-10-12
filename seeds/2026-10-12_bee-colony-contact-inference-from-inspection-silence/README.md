# Bee Colony Contact-Network Recovery from Inspection-Note Silence

## Question
**Can we infer which bee colonies are functionally connected (share disease or foraging pathways) by treating simultaneous *absence* from inspection notes as a network signal, and does this reveal critical colonies for biosecurity monitoring?**

## Non-Obvious Angle
Standard contact networks (e.g., epidemiology) are built from observed interactions or co-presence. Beekeeping data is fundamentally *sparse and episodic*: inspections are irregular and notes are incomplete. This project inverts the graph construction: **edges form when two colonies are jointly absent during overlapping time windows**, on the hypothesis that shared stressors (disease, forage failure) suppress symptom reporting and delay inspections on both simultaneously.

This forces a real modification to network algorithms:
- Normal graph analysis looks for *what is observed*.
- Here we build edges from *what is not observed* (absence patterns).
- Centrality metrics then reveal critical colonies not by symptom co-occurrence, but by structural position in the co-silence graph.

## Data
**Synthetic.** Generates 90 days of sparse inspection notes (2–3 colonies inspected per day) across 7 colonies. Edges in the inferred network represent colony pairs with co-absence in ≥2 consecutive 7-day windows.

## Limitation
This is a proof-of-concept. Real validation would require:
1. Actual apiary inspection logs (sparse, unstructured notes).
2. Ground-truth disease / foraging data (e.g., mite counts, nearby forage phenology) to confirm that co-absence really signals functional coupling.
3. Temporal resolution: if inspection frequency itself is endogenous (failing colonies get inspected more often), we'd need causal inference to separate confounding.

## What This Seed Demonstrates
- Graph inference from *co-absence* rather than co-occurrence.
- Betweenness centrality as a biosecurity metric (not just epidemiology).
- How sparse domain data shapes the technique differently than textbook network analysis.

---
*Auto-generated seed for data-analyst project farm. This is exploratory work.*
