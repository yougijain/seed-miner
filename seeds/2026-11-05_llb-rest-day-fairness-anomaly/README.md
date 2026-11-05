# Little League Scheduling Fairness via Rest-Day Anomaly Detection

## Question

Can anomaly detection uncover structural unfairness in little league schedules by identifying teams with anomalously **synchronized rest-day patterns**, where the anomaly is not scarcity but rather suspicious clustering into coordinated sub-schedules?

## The Non-Obvious Angle

Traditional anomaly detection finds *rare* or *isolated* points. Here we invert it: a **fair** schedule should have rest days distributed independently across teams. An **unfair** schedule reveals itself through *high synchronization*—teams clustered into coordinated rest blocks (e.g., "Block A rests odd weeks, Block B rests even weeks"). 

The technique must reframe: instead of "find teams that rest alone," we ask "find teams that rest *together too much*." This forces clustering-based anomaly detection to work backward from correlation strength rather than detection of rare events.

## Data

**Synthetic.** Generates two 8-team × 10-week schedules:
- **Fair schedule:** Rest days evenly distributed → low inter-team similarity.
- **Unfair schedule:** Teams partitioned into two synchronized blocks → high inter-team similarity.

The structure captures the real problem: unfairness manifests as emergent clustering in temporal rest patterns, not outliers in game counts.

## Limitation

This is a proof-of-concept on synthetic data with perfect clusters. Real schedules contain noise, partial constraints, and multiple fairness dimensions (travel distance, opponent strength, etc.). The method would need: (1) real game-and-rest data, (2) threshold calibration on historical fair/unfair schedules, and (3) integration with domain knowledge (e.g., geographic clustering is legitimate, but skill-based clustering is not).

## How It Works

1. Extract each team's binary rest pattern across weeks.
2. Compute pairwise cosine similarity of rest sequences.
3. Apply DBSCAN clustering on the rest patterns.
4. Measure within-cluster cohesion: high similarity = anomalous synchronization.
5. Flag clusters as potential fairness violations.

## Files

- `main.py` — Generates schedules, runs anomaly detection, outputs results.
- `results.json` — Cluster assignments and fairness scores.

## Run

```bash
python main.py
```

## Self-Assessment

This seed has genuine legs. The inversion of anomaly detection from "find rare" to "find correlated" is non-trivial and forces a real technique modification. The domain (little league fairness) is credible and unsolved. The synthetic data faithfully encodes the problem structure. A real implementation would require domain data and calibration, but the conceptual crack is authentic.

---

*This is an auto-generated project seed from the Seed Farm. Designed for exploration.*
