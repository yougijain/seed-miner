# Temporal Hypergraph Collapse: Brood-Care Anomaly Detection

## Question
Can we detect early colony disease by comparing observed nurse→brood contact networks against a null model of symmetric diffusion, where standard graph analysis fails because the temporal, causal structure must be explicitly engineered?

## The Seam
Standard network analysis (centrality, clustering, etc.) treats edges as undirected snapshots. Brood disease propagation is **causal and directed**: nurses transfer pathogens to brood, not vice versa. Moreover, the signal (disease avoidance) manifests as a *temporal asymmetry*: diseased brood cells receive care in rare, bursty windows, then are avoided.

The technique modification: **collapse the temporal hyperedge sequence into a time-lagged directed digraph**, where an edge u→v means u (nurse) contacted v (brood) within a sliding window. This forces explicit modeling of causality and reveals asymmetries in in-degree (receptivity) that undirected snapshots would obscure.

## Data
**Synthetic**. 100 timesteps of nurse-brood contact records. 
- Normal period (0–60): 8 nurses uniformly visit 12 brood cells.
- Anomaly period (60–100): brood cells B0–B3 are avoided (rare, bursty contact); healthy brood receive steady care.

The anomaly is injected to simulate disease-driven colony avoidance behavior.

## Signal
In-degree (number of distinct nurses visiting a brood cell) drops sharply for diseased cells after t=60, while healthy cells remain stable. Undirected snapshots would flatten this signal.

## Limitations
- Synthetic data; pattern must be validated on real apiary sensor logs (e.g., RFID hive recordings).
- Assumes nurses are the primary pathogen vector (true for many bee diseases, but not universal).
- Does not validate that detected asymmetry is *disease* vs. other causes (queen preference, cell location, etc.).

## Running
```bash
python main.py
```

**Auto-generated seed**, 2025-01-XX.
