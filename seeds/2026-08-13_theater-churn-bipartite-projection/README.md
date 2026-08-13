# Theater Subscriber Churn via Temporal Bipartite Projection Collapse

## Question
Can we detect season-ticket churn *before* cancellation by measuring when the bipartite graph of subscriber→show attendance collapses into disconnected components?

## Non-Obvious Angle
Traditional churn prediction uses RFM scoring or propensity models that treat each subscriber independently. This seed asks whether **network cohesion itself** (the connected component structure of a subscriber-subscriber projection built from show co-attendance) is a leading indicator.

The tension: Graph algorithms (connected components, density, centrality) assume a *static* network. But subscription renewal is fundamentally *temporal*—we observe season N and must predict cancellation in season N+1. Churners' attendance becomes idiosyncratic (they skip shows others attend), causing the bipartite projection to fragment. The technique must detect *fragmentation under temporal missingness*, not just structure.

## Why This Matters
The bipartite projection method collapses when you try to distinguish "no edge because they didn't co-attend" from "no edge because one will churn next season." The graph is static; churn is dynamic. Rebuilding the projection each season lets us detect *instability*: a cohesive S1 network shatters in S2 as churners scatter.

## Dataset
**Synthetic.** 40 subscribers × 20 shows/season × 3 seasons. 25% of subscribers are churners who gradually reduce attendance in season 2, then disappear in season 3. Non-churners maintain ~85% attendance.

## What's Real vs. Limitation
- **Real:** The bipartite projection and fragmentation metrics are correct graph-theoretic measures.
- **Real:** The signal (fragmentation before churn) exists by construction in the synthetic data.
- **Limitation:** We use synthetic data because theater box-office data with matched subscriber IDs + attendance + renewal flags is proprietary and rarely public.
- **Limitation:** This is a **proof-of-concept**. A real application would need: (1) multi-season historical data with cancellation labels, (2) validation that projection fragmentation predicts churn better than RFM, (3) handling of subscription tier diversity.
- **Limitation:** Network effects (word-of-mouth, peer influence on loyalty) are not modeled; we're only testing whether *attendance co-patterns* reveal structure.

## The Seam
The uncomfortable fit: most graph-analysis libraries assume you're studying a network *as-is*. Here, you're studying *network stability across temporal windows* to infer something about individuals (churn). The projection itself is lossy—two churners might never co-attend and thus be invisible neighbors. That's the real domain challenge, not just applying standard graph metrics.

---
*Auto-generated seed from community_theater_box_office × graph_network_analysis.*
