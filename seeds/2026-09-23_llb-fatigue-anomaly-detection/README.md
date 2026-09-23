# Little League Schedule Anomalies via Fatigue State Clustering

## Question
Can we detect structurally unfair/anomalous little-league schedules by treating cumulative team rest-day sequences as *paths through a fatigue state-space*, then flagging rounds where teams' rest patterns fracture into disconnected clusters?

## The Non-Obvious Angle
Standard anomaly detectors (Isolation Forest, LOF) work on *feature vectors* and assume data points are independent. Scheduling data is inherently **temporal and path-dependent**: whether a team's rest-day schedule is "normal" depends on the *entire cumulative history*, not individual frames.

This seed forces anomaly detection to:
1. Convert rest-day sequences into **state-space trajectories** (per-game fatigue features)
2. Cluster trajectories *as paths*, not as isolated points
3. Flag teams whose state-trajectory deviates from the dominant cluster (e.g., jumps between clusters, lands in noise)

This is a genuine modification: standard point-based anomaly detection fails here because the signal is *structural path divergence*, not point outliers.

## Data
**Synthetic** (no public little-league scheduling data readily available). Generated inline:
- 12 teams, 15-round schedules
- "Normal" teams: smooth, consistent rest intervals (1–3 days between games)
- **Anomaly 1**: Brutal bunching (games crammed into 2–3 consecutive days with long gaps elsewhere)
- **Anomaly 2**: Excessive rest (5+ day gaps creating uneven competitive rhythm)

The anomalies encode real scheduling problems: bunching causes fatigue cascades, excessive rest kills team rhythm.

## Limitation
This is a proof-of-concept on synthetic data. Real little-league schedules would need:
- Actual fixture data (venue, opponent, day-of-week constraints)
- Home/away travel impact on true "rest quality"
- Multi-team league-wide fairness assessment (not per-team anomalies)

The core technique (state-space trajectory clustering for temporal anomaly detection) is sound but limited by data availability and domain complexity.

## How to Run
```bash
python main.py
```

Output: Detected anomalous team IDs and accuracy vs. ground truth.

---
*This is an auto-generated seed. Use as exploratory scratch work.*
