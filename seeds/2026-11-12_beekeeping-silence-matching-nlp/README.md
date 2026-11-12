# Colony Health Inference from Inspection-Note Co-Silence Matching

## Question
Can we infer hidden disease/stress transmission pathways between bee colonies by detecting when inspection notes *jointly stop mentioning* specific health metrics in synchronized time windows, using NLP entity extraction inverted to track temporal silence instead of presence?

## Key Modification to NLP-on-Logs

Standard NLP entity extraction on beekeeping logs answers: "Which health metrics are present in this record?"

This project inverts that:
1. **Extract entities normally** (parse each inspection note for health metric mentions).
2. **Track first *absence***: For each (colony, metric) pair, identify the week when that metric stopped being mentioned.
3. **Build co-silence graph**: Detect pairs of colonies whose silence on the same metric started within a narrow timeframe (±2 weeks), indicating they may share a transmission pathway or environmental stressor.
4. **Cluster on silence edges**: Use graph connectivity of co-silence relationships to infer hidden colony clusters.

This forces NLP from operating on "what was written" (presence signal) to operating on "what stopped being written" (absence signal), which is non-obvious because:
- Inspection logs are sparse by nature; silence is ambiguous (missing visit vs. normal vs. outbreak).
- Entity extraction naturally prioritizes positive mentions; reversing it to track synchronized *gaps* requires dual timelines (extraction + absence onset).
- The signal lives in **temporal correlation of silence**, not in mention frequency.

## Data

**Synthetic, generated inline.** The dataset has:
- 15 colonies, 60 weekly inspection records (~8 months).
- 7 health metrics (brood, pollen_stores, honey_stores, varroa_signs, disease_signs, flight_activity, queen_sighting).
- 3 hidden clusters (A, B, C) where:
  - **Cluster A** (colonies C01–C05): jointly enter silence on `brood` + `pollen_stores` at week 20.
  - **Cluster B** (colonies C06–C10): jointly enter silence on `varroa_signs` + `disease_signs` at week 25.
  - **Cluster C** (colonies C11–C15): normal reporting throughout.

The synthetic silence is *engineered* to create a discoverable signal; real beekeeping logs would be messier but the structure is plausible: nutrient stress or disease spreads through apiaries, causing inspectors to stop reporting metrics when the problem is acute (or the colonies collapse).

## Limitations

1. **Synthetic data**: Real inspection logs are unstructured, irregular, and written by humans with inconsistent detail. This demo uses clean, templated metrics.
2. **Silence ambiguity**: A metric stopping being mentioned could mean "the colony died," "the inspector got lazy," or "the problem is so obvious it's unwritten." We cannot distinguish these causes from logs alone.
3. **No ground truth causality**: Co-silence suggests *association*, not causation. Colonies might silence on the same metric because they share a pathogen, or because they both got inspected by a trainee who didn't record that detail.
4. **Temporal resolution**: Weekly granularity is coarse; real pathogen spread or nutrition stress might play out over days.

## How to Run

```bash
python main.py
```

Output:
- Detected co-silence pairs (colonies with synchronized silence on specific metrics).
- Inferred clusters from the co-silence graph.
- Ground-truth cluster membership (for validation).

## Result Interpretation

If the inferred clusters match the hidden clusters (A, B, C), it shows that **silence patterns alone can recover hidden colony associations**, proving the inversion of NLP works: temporal co-absence reveals structure that presence-based metrics would miss.

---

*This is an auto-generated seed for exploratory analysis. It is not a production tool.*
