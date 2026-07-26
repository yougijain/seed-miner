import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

np.random.seed(42)

# Synthetic thrift inventory: items with true condition grade (unobserved)
# and price (observed) plus corrupted description (simulating OCR/data-entry noise).
ntems = 200
true_conditions = np.repeat([1, 2, 3, 4, 5], 40)  # 1=poor, 5=excellent
prices = true_conditions * 8 + np.random.normal(0, 5, ntems)
prices = np.clip(prices, 2, 50)

# Simulate noisy text description via character-level corruption.
# We'll encode condition as a text label, then corrupt it.
text_labels = {1: 'poor', 2: 'fair', 3: 'good', 4: 'very good', 5: 'excellent'}
true_labels = [text_labels[c] for c in true_conditions]

def corrupt_text(label, corruption_rate=0.4):
    """Randomly delete/swap characters to simulate OCR/data-entry errors."""
    label = list(label)
    for _ in range(max(1, int(len(label) * corruption_rate))):
        if label:
            i = np.random.randint(0, len(label))
            if np.random.rand() < 0.5:
                label.pop(i)
            else:
                label[i] = chr(ord('a') + np.random.randint(0, 26))
    return ''.join(label)

corrupted_labels = [corrupt_text(l) for l in true_labels]

# Feature engineering: extract "condition signal" from corrupted text via
# simple heuristic (word length + character frequency as proxy for original label).
def text_to_condition_signal(text):
    """Heuristic: longer text + more vowels ~ better condition (rough proxy)."""
    vowels = sum(1 for c in text.lower() if c in 'aeiou')
    return len(text) + 0.5 * vowels + np.random.normal(0, 0.5)

condition_signal = np.array([text_to_condition_signal(t) for t in corrupted_labels])

# Cluster on price + noisy condition signal.
X = np.column_stack([prices, condition_signal])
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Evaluate: do clusters align with true condition grades?
df = pd.DataFrame({
    'price': prices,
    'true_condition': true_conditions,
    'corrupted_label': corrupted_labels,
    'condition_signal': condition_signal,
    'cluster': clusters
})

# Compute purity: fraction of items whose cluster's majority label matches their true condition.
cluster_true_condition_mode = df.groupby('cluster')['true_condition'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
df['cluster_mode'] = df['cluster'].map(cluster_true_condition_mode)
purity = (df['true_condition'] == df['cluster_mode']).mean()

print(f"Clustering purity (noisy input): {purity:.3f}")
print(f"\nCluster centers (price, condition_signal):")
print(kmeans.cluster_centers_)
print(f"\nSample of clustered items:")
print(df[['price', 'true_condition', 'corrupted_label', 'cluster']].head(15))

results = {
    'purity': float(purity),
    'n_clusters': 5,
    'n_items': ntems,
    'question': 'Do market-meaningful price clusters emerge despite corrupted condition metadata?',
    'finding': 'Purity = {:.3f}: clustering finds partial structure despite noise; pure recovery impossible without ground truth.'.format(purity)
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{results['finding']}")