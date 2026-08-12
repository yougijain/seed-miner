# Sighting Rarity Inference from Sparse Fieldnote NLP

## Research Question

Can NLP on freeform birdwatching fieldnotes infer rarity class (common/uncommon/rare) for species **without explicit labels or count data**—instead learning rarity from *linguistic patterns* such as uncertainty hedges, detail density, and confirmation-seeking language?

## The Non-Obvious Angle

Traditional NLP on birding logs extracts species names, locations, and timestamps. This project inverts the task:

- **Hypothesis**: Watchers unconsciously write differently about rare birds. Rare sightings are *detailed* (justify the ID), *confirmatory* (call it in, take photos), and *confident* ("definitely it"). Common birds are written *fluidly* and *briefly*—no justification needed.
- **Technique seam**: NLP isn't designed to mine *linguistic style* as an implicit signal for a categorical outcome. Standard classifiers assume features directly describe the target. Here, linguistic markers are a *proxy* for observer confidence, which is *correlated with* rarity but doesn't directly mention it.

## Data

**Synthetic**, 10 fieldnotes covering common, uncommon, and rare sightings. The dataset is small but deliberately structured:
- Common: brief, fluent ("Saw robin. Red breast.")  
- Uncommon: moderate detail, explicit uncertainty ("Possible warbler...might be...uncertain")
- Rare: high detail, confirmation language ("SCARLET TANAGER! Photos. eBird submitted.")

Each note was hand-authored to embed the linguistic patterns that *should* correlate with rarity.

## Limitation

- **Tiny dataset**: accuracy is not generalizable. Real validation requires 100+ fielded notes with known rarity labels.  
- **Proxy confounding**: detail density and confirmation-seeking are proxies for observer confidence, not rarity itself. A very experienced birder might describe a rare bird *briefly* if certain; a novice might hedge a common sighting. Real deployment would need domain adaptation.
- **No ground truth rarity**: actual rarity is species-dependent, not fieldnote-dependent. The model learns "what does linguistic uncertainty look like," not "what is objectively rare."

## Method

1. Extract linguistic features:  
   - Uncertainty markers ("possible", "might", "?")  
   - Confirmation-seeking language ("photos", "submitted", "expert")  
   - Detail density (count of body-part + color words)  
   - Text length and negation  

2. Combine into a heuristic score (weighted sum).  

3. Classify: rare (high detail + confirmation − uncertainty), uncommon (moderate), common (low).  

## Output

Confusion matrix and per-class feature means showing whether the linguistic patterns separate rarity classes.

---

**Status**: Auto-generated seed. Real signal is latent in text style, not content. Useful as a proof-of-concept that rarity *inference* is possible; real application needs eBird-scale data and cross-validation.
