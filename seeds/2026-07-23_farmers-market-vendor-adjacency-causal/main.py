import pandas as pd
import numpy as np
from itertools import product
import json

np.random.seed(42)

# Synthetic farmers market data: 8 vendor stalls in fixed grid (2x4),
# observed across 14 market days (2 weeks, Sat/Sun repeats).
# SPATIAL CONFOUNDING: high-traffic corner stalls earn more, but also draw more foot traffic.
# We want: causal effect of being a weekend day on sales, accounting for location spillover.

vendors = [
    ('A', 0, 0),  # corner (high traffic baseline)
    ('B', 0, 1),
    ('C', 0, 2),
    ('D', 0, 3),
    ('E', 1, 0),  # adjacent to corner
    ('F', 1, 1),
    ('G', 1, 2),
    ('H', 1, 3),  # far corner (low traffic baseline)
]

# Distance-based traffic score: corner (0,0) and (0,3) and (1,0) are high-flow.
traffic_baseline = {}
for name, x, y in vendors:
    dist_to_corners = min(
        abs(x - 0) + abs(y - 0),
        abs(x - 0) + abs(y - 3),
        abs(x - 1) + abs(y - 0),
        abs(x - 1) + abs(y - 3)
    )
    traffic_baseline[name] = 100 - dist_to_corners * 15

data = []
for day_idx in range(14):
    day_name = ['Sat', 'Sun'][day_idx % 2]
    is_weekend = 1 if day_name in ['Sat', 'Sun'] else 0
    
    # Market-wide foot traffic: weekend has baseline +30% more traffic
    market_traffic = 1.0 + (0.3 if is_weekend else 0.0) + np.random.normal(0, 0.05)
    
    for vendor_name, x, y in vendors:
        # Causal path: is_weekend -> sales (direct + via traffic)
        # Confounder: location -> traffic (spatial effect)
        # Direct effect of location: baseline traffic
        location_effect = traffic_baseline[vendor_name]
        
        # Adjacency spillover (CONFOUNDING): if neighbor is busy, I get spillover
        neighbor_traffic = 0
        for other_name, ox, oy in vendors:
            if other_name != vendor_name:
                dist = abs(x - ox) + abs(y - oy)
                if dist == 1:  # adjacent
                    neighbor_traffic += traffic_baseline[other_name] * 0.15 / market_traffic
        
        # Sales generation
        base_sales = 50 + location_effect + neighbor_traffic * 20
        traffic_multiplier = market_traffic
        noise = np.random.normal(0, 15)
        sales = base_sales * traffic_multiplier + noise
        
        data.append({
            'vendor': vendor_name,
            'day_idx': day_idx,
            'day': day_name,
            'is_weekend': is_weekend,
            'sales': max(0, sales),
            'location_x': x,
            'location_y': y,
            'location_traffic_baseline': traffic_baseline[vendor_name]
        })

df = pd.DataFrame(data)
print("\n=== RAW SUMMARY ===")
print(df.groupby('is_weekend')['sales'].mean())
print(f"Naive weekend effect (t-test difference): {df[df['is_weekend']==1]['sales'].mean() - df[df['is_weekend']==0]['sales'].mean():.2f}")

# STANDARD APPROACH (covariate adjustment): add location_traffic_baseline
from sklearn.linear_model import LinearRegression

X_naive = df[['is_weekend']].values
y = df['sales'].values
model_naive = LinearRegression().fit(X_naive, y)

X_adjusted = df[['is_weekend', 'location_traffic_baseline']].values
model_adjusted = LinearRegression().fit(X_adjusted, y)

print("\n=== CAUSAL ESTIMATES ===")
print(f"Naive (unadjusted) weekend effect: {model_naive.coef_[0]:.2f}")
print(f"Adjusted for location baseline effect: {model_adjusted.coef_[0]:.2f}")
print(f"Location baseline coefficient: {model_adjusted.coef_[1]:.2f}")

# BACKDOOR STRATIFICATION: try to separate causal flow by location tier
df['location_tier'] = pd.cut(df['location_traffic_baseline'], bins=2, labels=['Low', 'High'])

print("\n=== STRATIFIED ANALYSIS (backdoor adjustment by location tier) ===")
strata_effects = []
for tier in ['Low', 'High']:
    subset = df[df['location_tier'] == tier]
    effect = subset[subset['is_weekend']==1]['sales'].mean() - subset[subset['is_weekend']==0]['sales'].mean()
    strata_effects.append({'tier': tier, 'effect': effect, 'n': len(subset)})
    print(f"{tier} traffic locations: weekend effect = {effect:.2f}")

# Weighted average (pool strata)
pooled_effect = sum(s['effect'] * s['n'] for s in strata_effects) / len(df)
print(f"Pooled stratified effect: {pooled_effect:.2f}")

print("\n=== INTERPRETATION ===")
print("The spatial confounding (adjacency spillover) creates a collider-like pattern:")
print("- High-traffic locations naturally see more weekend boost (true causal effect)")
print("- But covariate adjustment can over/under-correct if location is a collider")
print("- Stratification attempts to recover the 'true' causal effect within homogeneous groups")
print("\nLimitation: we can't truly isolate causal effects when treatment (day) is global")
print("and confounders are spatial/local. This is why RCT or natural experiments (weather)")
print("would be needed for definitive inference.")
