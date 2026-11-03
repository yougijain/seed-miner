# Colony Disease Propagation Inference via Temporal Co-Silence Bipartite Matching

## Question

Can we infer hidden disease-transmission pathways between bee colonies by running bipartite matching on *when colonies jointly fall silent* (stop reporting) specific health metrics in inspection notes, rather than on explicitly observed symptoms?

## The Non-Obvious Angle

Bipartite matching is typically designed to find optimal pairings between two groups based on *positive preference signals* (e.g., vendor-customer affinity, job candidate suitability). This seed inverts the input:

- **Standard matching:** match colonies with similar observed health profiles → find colonies with matching diseases.
- **This seed:** match colonies whose inspection notes *jointly go missing* on disease metrics in the same time windows → infer colonies that share epidemiological risk even if they never report overt symptoms together.

The pathology is that when a disease silently propagates through a colony network, colonies may stop reporting metrics (due to beekeeper neglect, colony collapse, or information loss) in a **temporally correlated way** that reveals the underlying transmission network. Matching on co-silence forces the algorithm to treat temporal *absence* as evidence of connection.

## Data

**Synthetic.** Generated 12 colonies, 20 inspections (weekly), 5 health metrics (pollen stores, brood pattern, flight activity, mite signs, disease spores). Ground truth: 3 hidden disease clusters (C01–C04, C05–C08, C09–C12). Within each cluster, colonies jointly fall silent on disease metrics (mite_signs, disease_spores) for a 4-week window offset by cluster ID.

This is synthetic because:
1. Real beekeeper inspection datasets are typically proprietary and sparse.
2. The ground truth clusters need to be known to validate whether matching recovers them.
3. The co-silence pattern (the signal) must be present and engineered to be strong enough to be detectable.

## Limitations

1. **Bipartite assumption is artificial:** Colonies don't naturally partition into left/right groups. The split here (C01–C06 vs C07–C12) is arbitrary; a real deployment would need domain knowledge or hierarchical clustering to define partitions.
2. **Silence ≠ disease:** Colonies going silent on reporting could be due to beekeeper neglect, not transmission. The inference conflates epidemiological proximity with record-keeping patterns.
3. **Scale and sparsity:** Real inspection data is often very sparse and irregular. Matching on few co-silent events is noisy.
4. **Temporal structure ignored:** The algorithm does not model *direction* of transmission or propagation delay; it only counts co-occurrence.

## Seed Status

**Auto-generated exploratory seed.** The core insight—that bipartite matching can be inverted to work on absence-correlation rather than presence—is sound. Whether this actually outperforms simpler co-silence clustering or network reconstruction methods would require real data and careful baseline comparison.
