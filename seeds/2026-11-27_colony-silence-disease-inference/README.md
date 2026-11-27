# Colony Health Inference from Inspection Note Temporal Silence Patterns

## Question
**Can we infer hidden disease transmission pathways between bee colonies by detecting which colonies jointly *stop mentioning* specific health metrics in synchronized time windows?**

## The Non-Obvious Angle

Standard NLP on inspection logs extracts *what was written*: mention of varroa mites, wing deformity, disease names. This project inverts the technique: it treats **temporal absence of metric mentions** as the signal.

When two colonies are epidemiologically linked (share transmission pathways), their beekeepers independently note the same health problems in roughly the same time window. As a colony develops disease, the beekeeper stops casually mentioning routine metrics (pollen stores, flight activity) because they're focused on the problem. The key insight: **synchronized silence is rarer than synchronized mention and signals shared risk.**

The technical modification: instead of using NLP to extract entities/topics, we use it to track *which metrics are conspicuously absent* across a time-series, then cluster colonies by overlap in these absence patterns.

## Data

**Synthetic.** Generated inline (`main.py`) with realistic inspection-note structure:
- 20 colonies, weekly inspections over 52 weeks
- 8 health metrics per inspection (brood_pattern, pollen_stores, etc.)
- Three "linked pairs" of colonies are injected with synchronized silence: around day 180–250, both colonies in a pair jointly stop mentioning varroa_mites, wing_deformity, comb_disease (85% silence rate), simulating a shared disease event.
- Baseline: unlinked colonies have ~30% independent silence per metric per week (random missingness).

The synthetic structure has the *right property*: linked colonies truly share a synchronized absence signature that unlinked pairs do not.

## Limitation

This is a proof-of-concept on synthetic data. Real inspection logs would need:
- Actual beekeeper note text (not structured labels)
- Noise: some metrics are skipped for administrative reasons, not biological
- Domain expertise to distinguish meaningful silence from routine variation
- Validation against known transmission events (apiary proximity, shared equipment, etc.)

The technique is *not* production-ready for epidemiology, but it demonstrates a genuine inversion of NLP methodology: extracting absence patterns as the primary signal rather than presence.

## Running

```bash
python main.py
```

Output: ranked pairs of colonies by synchronized silence score. The three linked pairs should appear in the top 10, proving that the technique can recover hidden transmission pathways from metric absence patterns alone.

---

**This is an auto-generated data-analyst project seed.** It is exploratory scratch work intended to test a novel pairing of domain and technique. Most seeds will be discarded.
