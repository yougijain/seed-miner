# Garden Plot Yield Scheduling: Detecting Endogenous Planting Shifts via Optimization Infeasibility

## Question
Can a bin-packing scheduler for garden plot assignments **detect which gardeners endogenously shift their planting dates** in response to prior-season yields by measuring when weekly plot-rotation feasibility *collapses* under their participation patterns?

## Why This Matters (The Non-Obvious Angle)

Standard garden plot scheduling treats gardeners as having **fixed preferences**: each gardener plants in week X, duration Y, needs plot type Z. The scheduler's job is to pack them into available plots.

But in reality, **gardeners adapt endogenously**: if they had a great yield last season, they plant earlier the next year (copying success). This compresses the planting window across the community, which breaks the bin-packing assumption that preference distributions stay stationary.

This project reverses the problem: instead of "Can we pack these fixed preferences?", we ask **"Can we infer hidden adaptive behavior from packing failures?"**

When conflicts spike, it's not because plots disappeared—it's because gardeners synchronized their timing in response to past outcomes. The optimization scheduler becomes a **behavioral inference tool**.

## Data

**Synthetic.** Generated inline in `main.py`.

- 12 gardeners × 3 seasons × 20-week windows
- Gardeners with high prior-season yield endogenously plant 1 week earlier in the next season
- Bin-packing assigns to 3 plots per week
- Tracks: planting week, duration, yield, packing conflicts

**Why synthetic:** Real community garden plot logs rarely record both detailed planting dates AND yield outcomes. We generate data with the *correct structure* (endogenous timing compression actually produces feasibility collapse in bin-packing) so the signal is detectable.

## Limitation

This is a **proof-of-concept**, not a full causal inference.  Real world has confounds (weather, soil quality, shared resources). The scheduler detects *feasibility collapse* as a symptom, not proof of endogeneity. Would need instrumental variables (e.g., weather shocks that break the adaptation link) to fully separate endogenous shifts from other timing drivers.

## Expected Output

- Feasibility test per season (success, # conflicts, plot utilization)
- Conflict delta across seasons (should spike if endogenous adaptation occurs)
- Gardener timing distributions (should show tighter clustering in later seasons)
- Interpretation: feasibility collapse → endogenous behavior signal

---

*This is an auto-generated data-analyst project seed. See `main.py` for full implementation.*