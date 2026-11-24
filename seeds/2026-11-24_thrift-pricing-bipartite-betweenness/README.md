# Thrift Store Pricing as Donor-Type Bipartite Flow: Detecting Hidden Inventory Rules via Network Betweenness Collapse

## Question
Can graph **betweenness centrality** detect hidden thrift-store pricing regimes by measuring when item-category price-tier bridges collapse under endogenous donation-composition shifts?

## The Non-Obvious Angle
Graph centrality metrics (betweenness, closeness, etc.) assume **stable network topology**. But thrift store pricing networks are endogenously destabilized by **upstream donor-mix shocks**: when the composition of donated items shifts (e.g., fewer clothing donations, more furniture), the store triggers hidden inventory management rules (dynamic pricing, category bunching, clearance strategies).

This causes **betweenness centrality to collapse** in category nodes that previously acted as bridges between donation sources and price tiers. The collapse *itself*—not the absolute centrality value—becomes a **regime-change signal**, inverting the technique from "find hub nodes" to "find when hubs stop working."

## Data & Synthetic Structure
**Synthetic.** Generates 100 days of thrift store item transactions with:
- Pre-shift (days 0–50): 70% clothing donations → priced cheap ($3.99), 30% furniture → expensive ($19.99).
- Post-shift (days 50–100): 30% clothing → priced mid ($8.99), 70% furniture → priced cheap ($7.99).
- The store applies a *hidden rule-flip* at day 50.

The network structure is a bipartite directed graph:
```
Day (donation source) → Category (inventory bucket) → Price Tier (output)
```

Edge weights = item counts flowing through each path.

## Limitation
In real data, donor-mix shifts would be **confounded** with seasonal trends, economic cycles, and actual demand elasticity. This synthetic setup isolates the pure effect. Also, real thrift stores have dozens of categories and complex, multi-stage pricing logic; this uses only clothing/furniture to keep the network small and readable.

## Expected Output
If the network is working correctly:
- **Pre-shift**: Categories show moderate, stable betweenness (they successfully bridge donors to price tiers).
- **Post-shift**: Betweenness *collapses* because the donor mix reversal breaks prior category-tier relationships.
- The collapse magnitude is a **detection signal** for the regime change.

---
*Auto-generated seed. For exploratory analysis only.*