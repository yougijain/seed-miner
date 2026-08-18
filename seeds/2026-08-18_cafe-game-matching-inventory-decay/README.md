# Board Game Cafe Shelf Affinity via Co-Absence Bipartite Matching

## Question
Can we infer which games should physically co-shelf by matching checkout *timing patterns* (synchronized unavailability) rather than explicit co-rental data? Does bipartite matching on temporal absence windows detect true co-play groups when the input signal is inverted (both games missing → likely played together)?

## Why This Is Non-Obvious
Standard matching/recommendation techniques assume **positive signal**: items bought together, users who like both, co-occurrence in events. This project inverts the input:  
- **Conventional matching:** two items appear together → affinity  
- **This approach:** two items are *both absent* during overlapping windows → affinity

The data shape forces modification: instead of a co-occurrence matrix, we build a **co-absence Jaccard matrix**, treating synchronized checkout gaps as evidence of shared play sessions. The bipartite matcher must work with an inverted feature space.

## Data
**Synthetic.** Generated 60 days of game checkout logs with 12 games and 3 hidden co-play clusters (e.g., "strategy games often rented together"). Singletons are rarely co-checked.

The synthetic structure is **real in intent**: cafes do observe hidden patterns of co-rental, and absence windows *are* a valid proxy for co-play (if both are unavailable, they were likely played in the same session).

## Limitation
This approach assumes checkout duration correlates with play session. In reality, games might sit on a table between sessions, violating the "both absent = played together" assumption. A real deployment would need:
- Return timestamps (not just checkout dates)  
- Session-level grouping (rentals linked by customer ID + time proximity)  

Without these, co-absence windows are a noisy proxy.

## How to Run
```bash
python main.py
```

Expects output showing recommended shelf groups and a recovery metric comparing inferred groups to ground truth.

---
*Auto-generated seed. Modification to matching_recommendation: input is temporal absence overlap, not co-occurrence.*
