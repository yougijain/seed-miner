# Colony Foraging Collapse Detection via Temporal Co-Absence Clustering

## Question
Can we detect latent colony foraging-system collapse by clustering on *when specific symptoms stop appearing together* in beekeeper inspection notes, rather than when they appear?

## Angle: NLP Working Backward
Standard NLP on logs extracts what *is* mentioned. Here, we force it to work backward from **temporal co-absence**: when a cluster of symptoms (pollen stores, brood pattern, flight activity) all vanish from the notes *simultaneously* across multiple inspections, that silence signals a hidden bottleneck—not because the symptoms are semantically linked in text, but because their joint disappearance is epidemiologically meaningful.

This is non-obvious because:
- Traditional symptom analysis looks at *co-mention* (what's discussed together).
- We flip it: co-*absence* in time windows reveals which systems decoupled from inspection records simultaneously.
- The technique had to adapt: TF-IDF vectorization normally highlights what's present; here we invert it to find zero-vectors within windows.

## Data
**Synthetic** (20 colonies, 40 inspections each over ~6 months).
- Colonies 0–4, 10–19: normal random symptom reporting.
- Colonies 5–9: foraging collapse window (inspections 20–30) where pollen stores, brood pattern, and flight activity all stop appearing together (silent co-absence).

**Why synthetic?** Real beekeeper logs are private and domain-specific. This synthetic data has the *right structure*: the signal (temporal co-absence of a symptom cluster in a subset of colonies) actually exists and is detectable.

## Limitation
This is a proof-of-concept. Real validation would require:
- Actual inspection logs (with ground-truth colony outcomes, e.g., colony death/recovery).
- Validation that co-absence patterns actually *precede* known foraging-system failures.
- Domain expert confirmation that pollen–brood–flight co-absence is epidemiologically meaningful.

Without that, co-absence clustering is a *signal hypothesis generator*, not a diagnostic tool.

## Output
The script identifies the top absence signatures by frequency and affected-colony count, then checks whether the expected foraging-collapse colonies cluster together under the hypothesis.

---
*Auto-generated seed. This is exploratory code; most seeds are discarded.*
