import re
import pandas as pd
from collections import Counter

# Synthetic fieldnotes with realistic structure
fieldnotes = [
    ("common", "Saw robin on lawn. Red breast. Left after 2min."),
    ("common", "Robins in yard. Multiple. Normal."),
    ("uncommon", "Possible warbler near oak. Yellow underparts. Checked field guide twice. Might be Cape May but uncertain. Saw for ~5 seconds."),
    ("uncommon", "Think I saw a redstart—at least, orange on wings, black back. Flew before I could confirm. Wasn't 100% sure."),
    ("rare", "Pileated woodpecker???! Large black body, white neck stripe, red crest—unmistakable if correct. Watched 3+ min, got photos on phone. Incredible. Called it in to local hotline."),
    ("rare", "This *might* be a Prothonotary warbler but I'm hesitant. Golden head, bluish wings. Only saw it once, could be misidentification. Need expert opinion. Audio recording taken."),
    ("common", "Chickadee. Heard and saw. Black cap."),
    ("uncommon", "Possible cedar waxwing—yellow tail band, silky plumage, crest. But light was poor. Tentatively recorded it. Could be wrong."),
    ("rare", "SCARLET TANAGER! Brilliant red male, black wings—this is definitely it, not a summer tanager. Photos, audio, eBird submitted. Multiple observers present."),
    ("uncommon", "Warblers (2 types?). Orange on one, unclear on other. Brief view. Uncertain identifications."),
]

def extract_features(text):
    """Extract linguistic markers of uncertainty and detail density."""
    text_lower = text.lower()
    
    # Uncertainty markers
    uncertain_words = ["possible", "might", "think", "uncertain", "tentatively", "unsure", "hesitant", "could", "?"]
    uncertainty_count = sum(text_lower.count(w) for w in uncertain_words)
    
    # Confirmation-seeking language
    confirm_phrases = ["checked", "photos", "audio", "submitted", "called", "observer", "expert"]
    confirmation_count = sum(text_lower.count(p) for p in confirm_phrases)
    
    # Detail density: number of descriptive adjectives and body-part mentions
    details = ["red", "black", "white", "yellow", "blue", "crest", "wing", "breast", "tail", "back", "stripe", "underpart", "plumage", "head"]
    detail_count = sum(text_lower.count(d) for d in details)
    
    # Length as proxy for investment
    text_length = len(text.split())
    
    # Negation of confidence
    negations = text_lower.count("not") + text_lower.count("n't")
    
    return {
        "uncertainty": uncertainty_count,
        "confirmation": confirmation_count,
        "detail_density": detail_count,
        "text_length": text_length,
        "negation": negations
    }

# Build dataset
data = []
for rarity_class, note in fieldnotes:
    features = extract_features(note)
    features["text"] = note
    features["rarity_true"] = rarity_class
    data.append(features)

df = pd.DataFrame(data)

# Simple heuristic classifier based on feature patterns
def classify_rarity(row):
    """Infer rarity from linguistic markers."""
    # Rare sightings: high detail, high confirmation-seeking, *low* uncertainty (certain rare ID)
    # Uncommon: moderate detail, moderate uncertainty
    # Common: low detail, low uncertainty, low confirmation
    
    score = (
        row["detail_density"] * 0.4 +
        row["confirmation"] * 0.3 +
        row["text_length"] * 0.15 -
        row["uncertainty"] * 0.1 +
        row["negation"] * 0.05
    )
    
    if score > 3.5:
        return "rare"
    elif score > 1.5:
        return "uncommon"
    else:
        return "common"

df["rarity_predicted"] = df.apply(classify_rarity, axis=1)

# Evaluate
accuracy = (df["rarity_predicted"] == df["rarity_true"]).mean()
confusion = pd.crosstab(df["rarity_true"], df["rarity_predicted"], margins=True)

print("Feature Engineering from Fieldnote Uncertainty Patterns")
print("=" * 60)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"\nConfusion Matrix:\n{confusion}")
print(f"\nFeature Importance (sample correlations):")
for col in ["uncertainty", "confirmation", "detail_density", "text_length"]:
    common_mean = df[df["rarity_true"] == "common"][col].mean()
    rare_mean = df[df["rarity_true"] == "rare"][col].mean()
    print(f"  {col:20s}: common={common_mean:5.2f}, rare={rare_mean:5.2f}")

print(f"\nSample Predictions:")
for idx, row in df.head(6).iterrows():
    print(f"  True: {row['rarity_true']:10s} | Pred: {row['rarity_predicted']:10s} | Text: {row['text'][:50]}...")
