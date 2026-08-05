# QSO Duration Clustering Under Scheduling Constraint Collapse

## Question

Can bin-packing heuristics (used to schedule contest time slots) **detect** mode-switching behavior in amateur radio logs when the log contains a mixture of contest QSOs (~3–5 min) and ragchew conversations (~15–30 min), and the packing algorithm itself forces misclassification at the boundary?

## Core Insight

Optimization schedulers assume **unimodal** task-duration distributions. When applied to a **bimodal** stream (contest + ragchew), the bin-packing solver becomes an inadvertent mode-detector: it clusters durations not to recover ground truth, but to minimize idle time. This reveals which operators were actually ragchewin (violating contest norms) because the solver can't pack them efficiently into short slots.

## Data

**Synthetic**, generated inline. 120 QSOs:
- 75% from contest mode: 3–5 min
- 25% from ragchew mode: 15–30 min
- Shuffled in time order to mimic real logs

This structure is realistic: contest logs do contain human ragchew intrusions, and contest organizers *do* use slot-packing logic to plan infrastructure.

## Approach

1. **Greedy bin-packing** on 15-minute slot capacity
2. **Gaussian Mixture Model (2 components)** to detect modes
3. **Cross-tabulation**: Are durations packed into bins correlated with their predicted mode?

The key is that packing forces a *different* clustering objective than likelihood maximization, revealing structural friction.

## Limitation

This is a toy dataset; real contest logs would need:
- Actual VHF/HF propagation data and frequency-band timing (which changes mode prevalence)
- Operator identity consistency (same call = operator skill/habit)
- Contest rules (not all ragchew is a violation)

The synthetic data is *too clean*—real bimodality would be more smeared and context-dependent.

## Files

- `main.py`: Generates synthetic log, runs greedy bin-packing + GMM, reports clustering alignment.
- `qso_log_with_modes_and_bins.csv`: Output log with predicted modes and bin assignments.

---

*Auto-generated seed. Not tutorial-grade; seams intentional.*
