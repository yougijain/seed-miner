# Youth Chess Tournament Pairing Feasibility Collapse Under ELO-Stratification

## Question

Can we detect hidden skill-tier clustering in a youth chess tournament by measuring when round-robin pairing optimization becomes infeasible under fairness constraints (opponent ELO rating limits)? Does the "collapse point" reveal tournament composition problems?

## Core Insight

Standard round-robin scheduling is a solved problem. But when you add a seemingly-reasonable constraint—"every opponent must be within K ELO points of your rating each round"—the optimizer doesn't just fail gracefully. It fails *diagnostically*. 

- **Tight ELO tolerance** → few rounds feasible
- **Loose ELO tolerance** → many rounds feasible  
- **The collapse point** → reveals whether players form skill clusters too far apart to pair fairly

Instead of asking "can we schedule this?", we ask "what player structure breaks this scheduling approach?" This inverts the technique from solver to probe.

## Data

**Synthetic.** Generated inline with 20 players in 3 known ELO clusters:
- Cluster 0: 8 players ~1200 ELO (K-8 beginners)
- Cluster 1: 7 players ~1600 ELO (junior club players)  
- Cluster 2: 5 players ~2000 ELO (competitive juniors)

This structure is deliberate: the clusters are separated such that tight ELO-fairness constraints will expose the gap.

## Expected Result

As ELO tolerance tightens:
1. Early rounds remain feasible (intra-cluster pairings work)
2. Around ±75–100 ELO, rounds collapse (inter-cluster gaps hurt feasibility)
3. Graph fragmentation metric rises (compatible opponent graph breaks into components)

The feasibility curve is diagnostic of tournament imbalance.

## Limitation

This is a **proof-of-concept** on a toy dataset with synthetic structure. Real youth tournaments have hundreds of players, multiple age groups, and rating inflation that may weaken the signal. The technique also assumes a greedy (not optimal) pairing algorithm for tractability; true constraint satisfaction would require a solver library.

## What's Real vs. Synthetic

- **Synthetic:** Player population and ELO distribution (generated to have visible clusters)
- **Real technique:** Bin-packing-like greedy scheduling under feasibility constraints
- **Real signal:** The feasibility collapse *does* occur at different tolerances depending on player distribution shape

---

*Auto-generated data-analyst project seed. Part of a public seed farm for exploratory analytics.*
