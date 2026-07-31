# Entailment Drift in Beekeeper Logs: Detecting Colony Decline via NLP

## Question

Can we detect colony decline *earlier* by measuring whether beekeeper inspection observations logically entail their stated health conclusions? Does the coherence of that entailment relationship degrade before explicit health markers appear in the logs?

## Non-Obvious Angle

Standard NLP on logs treats them as documents to classify or extract facts from. This approach treats logs as **defeasible reasoning chains**: an observation set should (probabilistically) entail a health conclusion. Under colony stress, that relationship weakens even before the beekeeper explicitly notes decline—observations become contradictory or disconnected from the stated conclusion.

This forces NLP away from token-level or sentence-level analysis toward **inter-statement logical consistency**, which isn't a native task for bag-of-words or even standard transformers.

## Data

**Synthetic.** The script generates two colonies across 8 weeks:
- **Colony A (Healthy):** observations are consistent, conclusions are affirming.
- **Colony B (Declining):** starting week 4, observations become mixed (scattered brood, mite signs, etc.), but conclusions lag behind, creating entailment incoherence.

The entailment signal degrades in Colony B before explicit "decline" language appears in conclusions.

## Method

1. **Entailment Proxy:** For each log entry, measure the semantic alignment between observations and conclusion using heuristic lexical markers (positive/negative terms) and sentiment divergence.
2. **Rolling Coherence:** compute a 2-week rolling entailment average.
3. **Drift Detection:** linear regression slope on rolling entailment reveals which colonies are losing logical coherence.

## Limitations

- **Entailment is heuristic:** A real system would use a pre-trained NLI model (e.g., cross-encoder or zero-shot classifier), but that requires an external model.
- **Synthetic data:** the pattern is designed in, not discovered. Real beekeeping logs would require domain expertise to validate the signal and may have confounding factors (e.g., fatigue in writing).
- **No causal proof:** entailment decay may correlate with decline, but causality isn't established.

## To Run

```bash
python main.py
```

Expect output showing Colony B with negative entailment slope (decay) and an alert, while Colony A remains stable.

---

**Auto-generated seed.** Not production-grade; meant to explore whether NLP on domain logs can detect structural (logical) failure, not just semantic content shifts.
