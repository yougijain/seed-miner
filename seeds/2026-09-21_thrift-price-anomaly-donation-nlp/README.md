# Thrift Store Pricing Anomaly Detection via Donation Note NLP

## Question
Can we detect thrift store pricing errors by using NLP on donor donation notes to infer expected item condition, then flagging items whose recorded condition-based price diverges from what the notes predicted?

## Why This Is Non-Obvious
Standard NLP-on-logs applies text analysis to extract explicit features (category, sentiment, intent). Here, NLP does the opposite: it *infers a latent feature* (donor-assessed condition) that was never recorded, then uses that inference as a forensic tool to detect inconsistency in a different domain signal (pricing). The technique is forced to work backwards from noisy donor text to reverse-engineer the pricing logic—turning NLP into an audit mechanism rather than a classification system.

## Data
**Synthetic.** 150 donation records with:
- `donor_note`: freeform text describing item condition (e.g., "never worn, no tags, from estate")
- `recorded_condition`: categorical label (mint/good/fair/worn) assigned by staff
- `recorded_price`: actual price tag based on recorded condition
- Ground truth: whether the recorded price actually matches the donor-implied condition

The data is generated so that 85% of items are priced consistently, and 15% have auditor errors where price doesn't match recorded condition or implied condition.

## Method
1. **NLP feature extraction**: TF-IDF + NMF topic modeling on donor notes to extract 4 latent "condition topics."
2. **Condition inference**: Map NMF topics to condition classes (mint/good/fair/worn) by examining top words.
3. **Mismatch scoring**: Compute condition mismatch (recorded vs. NLP-inferred) and price mismatch (recorded price vs. NLP-expected price).
4. **Anomaly detection**: Isolation Forest on [condition_mismatch, price_mismatch, recorded_price] to flag anomalous items.

## Limitation
This is **proof-of-concept on synthetic data**. Real thrift donor notes might:
- Contain less structured condition language (more ambiguous/sarcastic text)
- Have category-specific condition vocabularies (clothes vs. electronics) that NMF doesn't separate
- Include confounders (price negotiation, bulk lots) that break the assumption that price should follow condition

In production, you'd need:
- Annotated donor notes to train a supervised condition classifier
- Multi-label topics per donor (condition + category) to avoid topic collapse
- A more sophisticated price model (e.g., regression on condition + category + era)

## Output
The script prints the top 10 detected anomalies with:
- Donor note text
- Recorded vs. NLP-inferred condition
- Recorded vs. NLP-expected price
- Condition and price mismatch scores
- Recall and precision against ground-truth pricing errors

---
**This is an auto-generated data-analyst project seed.** It's exploratory; most seeds are discarded. See the root repo for context.
