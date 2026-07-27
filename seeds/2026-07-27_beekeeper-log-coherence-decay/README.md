# Beekeeper Log Coherence Decay: Fatigue and Colony Stress Inference

## Question
Can the *linguistic coherence drift* in beekeeper inspection logs—measured as vocabulary richness and token-sequence stability—serve as an early signal of colony stress or inspector fatigue, independent of explicit observations recorded?

## Why This Matters
Beekeeper logs are typically analyzed for *what they say* (observed symptoms, queen presence, food stores). This project flips the lens: **how** logs are written (repetitiveness, vocabulary consistency, level of detail) may reflect both external stressors (colony decline, disease) and internal ones (inspector fatigue, time pressure). NLP on logs typically assumes stable vocabulary and style; here, the *deviation* from that stability becomes the signal.

## Approach
1. **Synthetic logs** mimicking a 30-day inspection season (early: detailed; mid: deteriorating; late: fragmented).
2. **Coherence metric**: Type/Token Ratio (TTR) in rolling 5-log windows measures lexical diversity.
3. **Decay detection**: Linear fit to richness trajectory identifies whether coherence declines over the season.
4. **Detail signal**: Count of numeric values and observational phrases as a coarse proxy for inspection depth.

## Data
- **Synthetic**: 30 synthetic log entries with intentional coherence decay baked in to simulate realistic fatigue + colony stress patterns.
- **Limitation**: Real beekeeper logs are not publicly available in bulk (privacy, niche domain). The synthetic data is designed to have the *right statistical structure* (coherence actually decays) so the technique isn't tested on noise.

## Key Finding
The decay slope (richness change per rolling window) should be negative if logs deteriorate as expected. This *could* correlate with actual colony decline in real data—a hypothesis for future validation.

## Real vs. Synthetic
- **Synthetic**: All log entries and the decay pattern.
- **Real potential**: Apply to actual beekeeper logs if a source (e.g., citizen science platform, regional beekeeping association) becomes available.
- **Limitation**: No real-world validation; signal is constructed, not discovered.

## Files
- `main.py`: Generate synthetic logs, compute coherence metrics, detect decay.

## Run
```bash
python main.py
```

---
*This is an auto-generated project seed for exploratory data-analyst work.*
