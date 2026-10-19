# Board Game Café: Contention Detection via Bin-Packing Feasibility Collapse

## Question
**Can we detect which board games have hidden demand incompatibilities (player pool overlap, demand-time collisions) by observing *when bin-packing of table-slot schedules fails*, rather than from explicit co-checkout data?**

## Why This Angle?
Optimization scheduling (bin-packing) is normally used *forward*: given constraints, find the best packing.  
This seed inverts it: *watch where packing fails* as a diagnostic signal.  

In a board game café, some games create temporal contention—they share players or demand-surge windows. A naive approach would cluster checkout co-occurrence. Instead, we measure: "does this game's presence cause more packing failures than its absence would?"  

This forces bin-packing to work as a *constraint-violation detector*, not a utilization maximizer.

## Data
**Synthetic.** 30 days × 6 two-hour slots per day × 7 games with realistic demand structure:  
- Catan + SevenWonders: both peak midday (incompatible)
- Codenames + Splendor: both peak evening (incompatible)
- Others: low baseline demand

Each checkout records game, day, slot, and duration (0.5–2 hours).  
No explicit "incompatibility" labels—the signal is purely structural (packing stress).

## Method
1. For each game, isolate its checkout events across time slots.
2. For each slot it appears in, measure: can we bin-pack *all* games in that slot into ≤4 tables (2-hour capacity)?
3. Compare packing feasibility *with* vs *without* the focal game.
4. "Contamination score" = (failures_with - failures_without) / appearances.

High contamination means the game acts as a contention flash-point.

## Limitation
- Bin-packing assumes symmetric conflict (any game can conflict with any other). Real cafés have non-symmetric affinity (hardcore vs casual player pools).
- Synthetic data is hand-tuned; real checkout logs would need validation that packing failures actually correlate with player-experience friction.
- Does not recommend *solutions* (e.g., dedicated tables for incompatible games), only diagnosis.

## Run
```bash
python main.py
```

---
*This is an auto-generated seed for the data-analyst project farm. Most seeds are discarded.*
