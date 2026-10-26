# Community Theater Box Office Anomaly Detection via Revenue-Per-Seat Variance Collapse

## Question
Can we detect structural show failures (cast illness, technical problems, declining audience word-of-mouth) by watching the *variance* of revenue-per-seat *collapse* during a run, using anomaly detection to flag unusually *stable* pricing patterns rather than absolute revenue outliers?

## Non-Obvious Angle
Standard box-office anomaly detection looks for absolute volume spikes or crashes. This project flips it: we anomaly-detect on **variance collapse** in the per-seat revenue stream.

### Why This Matters
- **Declining shows**: Only core fans show up → attendance and pricing become highly predictable → LOW variance
- **Normal shows**: Mix of demographics, discount timing, group sales → moderate variance
- **Viral shows**: Word-of-mouth creates chaotic demand → HIGH variance

The seam: Isolation Forest is designed to flag outlier *values*. Applying it to rolling *variance* forces the technique to detect stability patterns, not extreme numbers. This is a genuine modification to how the algorithm is typically deployed.

## Data
**Synthetic.** 5 shows × 8 nights, with three quality patterns:
- 2 normal shows (baseline variance)
- 1 declining show (cast illness simulated as low, stable variance)
- 1 viral show (word-of-mouth boom: high, unstable variance)
- 1 normal show (control)

Each night has:
- Seats sold (drawn from normal distribution, variance depends on show quality)
- Revenue per seat (drawn from normal distribution, variance depends on show quality)
- Rolling 3-night standard deviation of revenue-per-seat

## Technique
1. Compute rolling standard deviation of revenue-per-seat within each show (3-night window)
2. Scale variance values
3. Apply Isolation Forest with contamination=0.2 to detect anomalously *low* or *high* variance patterns
4. Inspect which shows have the highest anomaly rate

## Limitation
In real box-office data, variance collapse could be caused by many unrelated factors (pricing systems changes, POS software updates, seasonal patterns). This project assumes variance patterns are a meaningful signal of show health, which is **speculative** and would need domain validation. The synthetic data is engineered to make the signal pop; real data would be noisier and require causal adjustment to isolate show quality from confounders (day-of-week, weather, competing events).

## Expected Output
Decline shows flag as high anomaly rate due to variance collapse. Normal shows remain at baseline. Viral shows may also flag if growth-induced variance is extreme enough.

---
*This is an auto-generated seed from a data-analyst seed farm. It is meant to be exploratory scratch work.*
