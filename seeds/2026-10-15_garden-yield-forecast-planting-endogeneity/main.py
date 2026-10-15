import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import json

np.random.seed(42)

# Synthetic: 20 gardeners, 3 years of yield + planting-date records
# True signal: 8 gardeners adapt planting date to prior yield (endogenous);
# 12 plant on fixed calendar (exogenous control).

gardeners = []
for gid in range(20):
    endogenous = gid < 8  # First 8 adapt planting to prior yields
    baseline_plant_doy = 80 + np.random.randint(-5, 5)  # ~mid-March, day-of-year
    
    ts_yields = []
    ts_plant_doys = []
    
    prior_yield = 50
    for year in range(36):  # 3 years * 12 months for monthly granularity
        if year > 0:
            if endogenous:
                # Shift planting by ±5 days per 10-unit yield deviation
                shift = (prior_yield - 50) / 10 * 5
                plant_doy = baseline_plant_doy + shift + np.random.normal(0, 2)
            else:
                plant_doy = baseline_plant_doy + np.random.normal(0, 1)
        else:
            plant_doy = baseline_plant_doy
        
        # Yield influenced by planting date + noise
        yield_val = 50 + (baseline_plant_doy - plant_doy) * 0.8 + np.random.normal(0, 5)
        yield_val = max(20, yield_val)
        
        ts_yields.append(yield_val)
        ts_plant_doys.append(plant_doy)
        prior_yield = yield_val
    
    gardeners.append({
        'gardener_id': gid,
        'endogenous': endogenous,
        'yields': ts_yields,
        'plant_doys': ts_plant_doys
    })

# Fit ARIMA(1,0,1) to each gardener's yield series, extract residuals
results = []
for g in gardeners:
    try:
        model = ARIMA(g['yields'], order=(1, 0, 1))
        fit = model.fit()
        residuals = fit.resid.values
        
        # Extract features: residual autocorrelation structure
        # If endogenous, residuals will show systematic trend (unexplained by ARIMA)
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
        residual_trend = np.polyfit(range(len(residuals)), residuals, 1)[0]
        residual_std = np.std(residuals)
        resid_mean_abs = np.mean(np.abs(residuals))
        
        results.append({
            'gardener_id': g['gardener_id'],
            'endogenous': g['endogenous'],
            'aic': fit.aic,
            'lag1_autocorr': lag1_corr,
            'residual_trend': residual_trend,
            'residual_std': residual_std,
            'resid_mean_abs': resid_mean_abs
        })
    except:
        pass

df_results = pd.DataFrame(results)

# Cluster on residual structure features
# Hypothesis: endogenous gardeners have residuals with systematic trends
X = df_results[['lag1_autocorr', 'residual_trend', 'resid_mean_abs']].values
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_results['cluster'] = kmeans.fit_predict(X)

# Which cluster has higher residual_trend magnitude? (Indicates systematic bias in ARIMA)
cluster_profiles = df_results.groupby('cluster').agg({
    'residual_trend': 'mean',
    'endogenous': 'mean',
    'aic': 'mean'
})

print("Cluster Profiles (detecting endogeneity via residual structure):")
print(cluster_profiles)
print()

# Validate: which cluster contains more truly-endogenous gardeners?
for cluster_id in [0, 1]:
    mask = df_results['cluster'] == cluster_id
    endogenous_pct = df_results[mask]['endogenous'].mean()
    print(f"Cluster {cluster_id}: {endogenous_pct:.1%} endogenous gardeners")

print()
print("Detection accuracy (endogenous gardeners correctly separated):")
predicted_endogenous = df_results[df_results['cluster'] == 1]['endogenous'].mean() > 0.5
true_endogenous_count = df_results['endogenous'].sum()
print(f"  Expected: ~8 endogenous, Found in likely-endogenous cluster: ", end="")
print(f"{(df_results[df_results['cluster']==1]['endogenous'].sum())}")
