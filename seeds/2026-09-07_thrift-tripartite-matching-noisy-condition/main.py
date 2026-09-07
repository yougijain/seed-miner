import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import json

np.random.seed(42)

# Generate synthetic thrift inventory with latent categories encoded in price
categories = ['clothing', 'books', 'electronics', 'furniture', 'dishes']
prices_by_cat = {
    'clothing': np.random.uniform(2, 15, 40),
    'books': np.random.uniform(0.5, 8, 40),
    'electronics': np.random.uniform(5, 60, 40),
    'furniture': np.random.uniform(20, 150, 40),
    'dishes': np.random.uniform(1, 25, 40)
}

data = []
for cat in categories:
    for i, price in enumerate(prices_by_cat[cat]):
        # True condition: high quality if near upper quartile of category
        true_qual = 'excellent' if price > np.percentile(prices_by_cat[cat], 75) else 'good'
        # Corrupt condition label: random flips + OCR noise
        labels = ['excellent', 'good', 'fair', 'poor']
        corruption_prob = 0.4
        noisy_cond = np.random.choice(labels) if np.random.rand() < corruption_prob else true_qual
        data.append({
            'item_id': f'{cat}_{i}',
            'category': cat,
            'price': price,
            'true_condition': true_qual,
            'noisy_condition': noisy_cond
        })

df = pd.DataFrame(data)
print(f"Inventory: {len(df)} items, {df['category'].nunique()} latent categories")
print(f"Price ranges by category:\n{df.groupby('category')['price'].agg(['min', 'max', 'mean'])}
")

# BASELINE: Condition-based matching (fails due to noise)
from sklearn.feature_extraction.text import TfidfVectorizer
cond_vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 2)).fit_transform(df['noisy_condition'])
nn_cond = NearestNeighbors(n_neighbors=4, metric='cosine').fit(cond_vec)
dists_cond, indices_cond = nn_cond.kneighbors(cond_vec)

# Score: % of neighbors in same true category (baseline)
baseline_score = 0
for i, neighbors in enumerate(indices_cond):
    true_cat = df.iloc[i]['category']
    match_count = sum(1 for j in neighbors[1:] if df.iloc[j]['category'] == true_cat)
    baseline_score += match_count / 3  # 3 neighbors (excluding self)
baseline_score /= len(df)
print(f"Baseline (noisy condition matching): {baseline_score:.3f} category purity")

# ADAPTED MATCHING: Price-quantile matching (robust to noisy condition)
# Match items within category-specific price quantiles
scaler = StandardScaler()
price_scaled = scaler.fit_transform(df[['price']])

adapted_score = 0
for i in range(len(df)):
    true_cat = df.iloc[i]['category']
    # Find 3 nearest in price space
    dists = np.abs(price_scaled - price_scaled[i]).flatten()
    neighbors = np.argsort(dists)[1:4]  # exclude self
    match_count = sum(1 for j in neighbors if df.iloc[j]['category'] == true_cat)
    adapted_score += match_count / 3
adapted_score /= len(df)
print(f"Adapted (price-quantile matching): {adapted_score:.3f} category purity")

# Analysis: Which categories benefit most from price-based matching?
print("\nPer-category improvement (price vs. condition matching):")
for cat in categories:
    mask = df['category'] == cat
    cat_df = df[mask].reset_index(drop=True)
    cat_indices = np.where(mask)[0]
    
    # Condition baseline for this category
    cond_score = sum(
        sum(1 for j in indices_cond[idx][1:] if df.iloc[j]['category'] == cat) / 3
        for idx in cat_indices
    ) / len(cat_df)
    
    # Price adapted for this category
    price_score = sum(
        sum(1 for j in np.argsort(np.abs(price_scaled - price_scaled[idx]).flatten())[1:4]
            if df.iloc[j]['category'] == cat) / 3
        for idx in cat_indices
    ) / len(cat_df)
    
    print(f"  {cat:15s}: condition={cond_score:.3f}, price={price_score:.3f}, delta={price_score - cond_score:+.3f}")

# Output summary
result = {
    'condition': 'deliberately corrupted (40% random flip)',
    'baseline_purity': float(baseline_score),
    'adapted_purity': float(adapted_score),
    'improvement': float(adapted_score - baseline_score),
    'insight': 'Price structure encodes category membership; when condition text is noisy, price-quantile matching recovers latent categories more reliably than noisy text matching.'
}
print(f"\n{json.dumps(result, indent=2)}")