# Community Theater Intermission Pacing Anomalies

## Question
Can anomaly detection on *concession transaction inter-arrival timing distributions* during intermissions reveal hidden show problems (technical failures, cast injuries, emergency dismissals) that would otherwise go unnoticed in raw ticket/revenue data?

## The Non-Obvious Angle
Standard anomaly detection (Isolation Forest, LOF) is built to find **rare or isolated points** in feature space. But intermission concession data inverts this problem: 

- **Normal intermission**: ~40–50 concession transactions spread roughly uniformly across 15 minutes → **high variance** in inter-arrival intervals
- **Anomalous intermission**: 
  - Tech failure (audience held in seats) → transactions **clustered in the last 2 min** → **collapsed variance**
  - Cast injury (show extended, intermission cut short) → transactions **front-loaded** → **skewed distribution**
  - Emergency (fire alarm) → transactions **abruptly stop** → **reduced transaction count**

The technique must work *backward* from uniformity/variance collapse to diagnose problems, not forward from rare outliers.

## Data
**Synthetic**, 12 shows over 4 weeks. Each show has ~40–50 concession transactions logged with precise timestamps during intermission. Shows 5, 9, and 11 are injected with anomalies (tech hold, shortened intermission, early exit).

**Why synthetic?** Real theater concession data is proprietary. The synthetic structure faithfully models the core anomaly pattern: when something goes wrong in a show, concession-purchase *timing patterns* change dramatically, not just volume.

## Limitations
1. **Relies on high-frequency transaction logging** — many theaters only total concessions by intermission, not by transaction timestamp.
2. **Confounders** — Saturday vs. weekday, show type (musical vs. drama), genre popularity all affect transaction variance independently of problems.
3. **Scale** — 12 shows is toy data; real signal would require 50+ shows and careful time-series controls.
4. **No causal link** — detecting timing anomalies doesn't prove causation; needs post-hoc incident reports to validate.

## How to Run
```bash
python main.py
```

Output:
- Table of per-show interval statistics and anomaly scores
- Precision/recall vs. injected anomalies
- Interpretation of which mechanism (variance collapse, count drop, skew) triggered each alert

---

*This is an auto-generated seed from a data-analyst project farm. Most seeds are exploratory; many won't scale.*
