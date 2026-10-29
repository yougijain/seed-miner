# Farmers Market Stall-Conflict Detection via Bin-Packing Infeasibility Cascades

## Question

Can we detect which farmers-market vendors have **hidden operational dependencies** (shared power lines, water access, misting equipment, or foot-traffic cannibalization) by measuring how often weekly stall assignments become **bin-packing infeasible** when those vendors are simultaneously present?

## The Non-Obvious Angle

Standard bin-packing optimization treats all items as independent and aims to *minimize* the number of bins used. This project **reverses that flow**: it uses bin-packing *failure* as a signal that reveals latent affinity constraints.

### Why This Matters
Farmers-market vendors have real undeclared co-constraints:
- Beverage vendors share heavy water and power demands
- Prepared-food vendors all need 240V hookups
- Produce vendors require shared misting systems for freshness

When two vendors from the same affinity group are scheduled together, they exceed available power/water infrastructure, causing the packing algorithm to need more stall slots than are physically available. By detecting *which vendor pairs* repeatedly co-occur in weeks where packing becomes infeasible, we can reverse-engineer the hidden dependency graph.

## Data

**Synthetic.** 20 vendors with realistic stall footprints (8–25 m²) and three hidden affinity groups (beverages, prepared-food, produce) plus independent vendors. 50 weeks of simulated attendance (15–18 vendors per week). Hidden conflict cost: when two vendors from the same affinity group co-attend, they are effectively "larger" (require more infrastructure), making bin-packing fail.

This synthetic structure is necessary because real farmers-market data with stall conflicts, power specs, and attendance logs is not publicly available. The structure is realistic: it mimics real operational constraints and produces genuine infeasibility patterns.

## Limitation

- **No real data.** Validation requires either (a) access to actual farmers-market stall-conflict reports, or (b) domain expert review of detected pairs.
- **Greedy packing simplification.** The code uses a simple greedy bin-packing heuristic; a real system would use optimal packing solvers, but the insight (failure patterns reveal affinity) holds.
- **Affinity is binary.** The synthetic data assumes two vendors either conflict or don't; real constraints may be graded (e.g., weak shared power, strong foot-traffic cannibalization).

## How to Run

```bash
python main.py
```

Outputs:
- Console report of top vendor pairs co-occurring in infeasible weeks
- `results.json` with structured results

## Expected Output

You should see:
- ~20–25% of weeks marked as infeasible (need >5 stall slots for 15–18 vendors)
- Top vendor pairs are almost exclusively from the same hidden affinity group
- The reverse bin-packing signal is working: failures reliably co-locate dependent vendors

---

*This is an auto-generated seed from the data-analyst project farm. It demonstrates how optimization_scheduling (bin-packing) can be inverted to perform dependency inference when the scheduling technique fails on hidden domain constraints.*
