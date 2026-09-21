import re
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

np.random.seed(42)

# Generate synthetic donation logs with condition hints embedded in donor notes
donations = []
prices_by_condition = {"mint": 12, "good": 7, "fair": 4, "worn": 2}
condition_keywords = {
    "mint": ["never worn", "pristine", "new", "sealed", "unworn"],
    "good": ["barely used", "clean", "light wear", "gently"],
    "fair": ["used", "visible wear", "some stains", "thread loose"],
    "worn": ["heavy wear", "damaged", "broken", "torn", "stained"]
}

for i in range(150):
    true_condition = np.random.choice(list(prices_by_condition.keys()), p=[0.15, 0.35, 0.35, 0.15])
    keywords = np.random.choice(condition_keywords[true_condition], 1)[0]
    noise_words = np.random.choice(["from estate", "box lot", "no tags", "mixed"], 1)[0]
    donor_note = f"{keywords}. {noise_words}. Donated by {np.random.choice(['estate', 'thrift', 'household'])}"
    
    # Most items priced consistently; some have anomalous pricing (auditor error or intentional marking)
    if np.random.rand() < 0.85:
        recorded_price = prices_by_condition[true_condition]
    else:
        # Pricing anomaly: price doesn't match condition
        recorded_price = np.random.choice(list(prices_by_condition.values()))
    
    recorded_condition = [k for k, v in prices_by_condition.items() if v == recorded_price][0]
    
    donations.append({
        "item_id": f"item_{i}",
        "recorded_condition": recorded_condition,
        "recorded_price": recorded_price,
        "donor_note": donor_note,
        "true_condition": true_condition  # Ground truth for evaluation
    })

df = pd.DataFrame(donations)

# Extract condition from donor notes using TF-IDF + NMF topic modeling
vectorizer = TfidfVectorizer(max_features=20, stop_words="english", ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["donor_note"])

# NMF to extract 4 latent "condition topics"
nmf = NMF(n_components=4, random_state=42, init="random", max_iter=300)
topics = nmf.fit_transform(tfidf_matrix)
feature_names = vectorizer.get_feature_names_out()

# Map topics to condition classes by examining top words
topic_to_condition = {}
for topic_idx in range(4):
    top_indices = nmf.components_[topic_idx].argsort()[-3:][::-1]
    top_words = feature_names[top_indices]
    # Heuristic: map by keyword similarity to known condition keywords
    score_by_cond = defaultdict(float)
    for word in top_words:
        for cond, keywords in condition_keywords.items():
            if any(kw in word or word in kw for kw in keywords):
                score_by_cond[cond] += 1
    topic_to_condition[topic_idx] = max(score_by_cond, key=score_by_cond.get, default="fair")

# Infer NLP-predicted condition from dominant topic
df["nlp_inferred_condition_topic"] = topics.argmax(axis=1)
df["nlp_inferred_condition"] = df["nlp_inferred_condition_topic"].map(topic_to_condition)

# Create mismatch score: does the recorded_condition match the NLP-inferred condition?
# Encode condition severity as ordinal
condition_order = {"mint": 3, "good": 2, "fair": 1, "worn": 0}
df["recorded_severity"] = df["recorded_condition"].map(condition_order)
df["nlp_severity"] = df["nlp_inferred_condition"].map(condition_order)
df["condition_mismatch"] = np.abs(df["recorded_severity"] - df["nlp_severity"])

# Also compute price vs. NLP-inferred price disparity
df["nlp_expected_price"] = df["nlp_inferred_condition"].map(prices_by_condition)
df["price_mismatch"] = np.abs(df["recorded_price"] - df["nlp_expected_price"])

# Create anomaly feature vector: condition_mismatch + price_mismatch + price-level
X = df[["condition_mismatch", "price_mismatch", "recorded_price"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest to detect anomalies
iso_forest = IsolationForest(contamination=0.15, random_state=42)
df["anomaly_score"] = iso_forest.fit_predict(X_scaled)
df["is_anomaly"] = df["anomaly_score"] == -1

# Evaluate against ground truth (pricing that disagrees with true condition)
df["true_price_anomaly"] = df["recorded_price"] != df["true_condition"].map(prices_by_condition)

anomalies = df[df["is_anomaly"]].sort_values("price_mismatch", ascending=False).head(10)

print("\n=== Thrift Price Anomaly Detection via Donation Note NLP ===")
print(f"Total items: {len(df)}")
print(f"Detected anomalies: {df['is_anomaly'].sum()}")
print(f"True pricing anomalies in ground truth: {df['true_price_anomaly'].sum()}")
print(f"\nTop anomalies (by NLP-predicted price mismatch):")
print(anomalies[["item_id", "donor_note", "recorded_condition", "nlp_inferred_condition", "recorded_price", "nlp_expected_price", "condition_mismatch", "price_mismatch"]].to_string())
print(f"\nRecall (detected true anomalies / total true): {(anomalies['true_price_anomaly'].sum() / df['true_price_anomaly'].sum() * 100):.1f}%")
print(f"Precision (true anomalies in detected / detected): {(anomalies['true_price_anomaly'].sum() / len(anomalies) * 100):.1f}%")
