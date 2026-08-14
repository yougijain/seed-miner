import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic data: 40 plots, 12 weeks of yield measurements
n_plots = 40
n_weeks = 12

data = []
for plot_id in range(1, n_plots + 1):
    # Most plots: stable yield + small noise
    if plot_id <= 35:
        base_yield = np.random.uniform(8, 18)
        noise = np.random.normal(0, 0.5, n_weeks)
        yields = base_yield + noise
    # Plots 36-38: destabilizing microclimate (variance grows mid-season)
    elif plot_id <= 38:
        base_yield = np.random.uniform(10, 16)
        week_seq = np.arange(1, n_weeks + 1)
        noise = np.random.normal(0, 0.3 + 0.15 * (week_seq / n_weeks), n_weeks)
        yields = base_yield + noise
    # Plots 39-40: pest/disease outbreak (sudden variance spike)
    else:
        base_yield = np.random.uniform(10, 16)
        noise = np.concatenate([
            np.random.normal(0, 0.3, 6),
            np.random.normal(0, 2.5, 6)
        ])
        yields = base_yield + noise
    
    data.append({"plot_id": plot_id, "yields": list(yields)})

df_raw = pd.DataFrame(data)

# Feature engineering: rolling variance (the KEY insight—detect variance instability, not mean)
features = []
for _, row in df_raw.iterrows():
    yields = np.array(row["yields"])
    # Compute rolling variance over 3-week windows
    rolling_vars = [
        np.var(yields[max(0, i-2):i+1])
        for i in range(len(yields))
    ]
    # Aggregate statistics: mean variance, variance of variance, coefficient of variation
    features.append({
        "plot_id": row["plot_id"],
        "mean_variance": np.mean(rolling_vars),
        "variance_of_variance": np.var(rolling_vars),
        "cv_variance": np.std(rolling_vars) / (np.mean(rolling_vars) + 1e-6),
        "max_variance": np.max(rolling_vars),
    })

df_features = pd.DataFrame(features)

# Isolation Forest on temporal heteroskedasticity
iforest = IsolationForest(
    contamination=0.12,
    random_state=42,
    n_estimators=100
)

X = df_features[["mean_variance", "variance_of_variance", "cv_variance", "max_variance"]].values
anomalies = iforest.fit_predict(X)
scores = iforest.score_samples(X)

df_features["anomaly_label"] = anomalies
df_features["anomaly_score"] = scores

# Merge back raw yields for inspection
df_results = df_features.merge(df_raw, on="plot_id")
df_results["anomaly_flag"] = df_results["anomaly_label"] == -1

# Report
print("\n=== TEMPORAL HETEROSKEDASTICITY ANOMALY DETECTION ===")
print(f"Detected {df_results['anomaly_flag'].sum()} anomalous plots (of {len(df_results)})")
print("\nAnomalous plots (variance instability):")
anomalous = df_results[df_results["anomaly_flag"]].sort_values("anomaly_score")
for _, row in anomalous.iterrows():
    print(f"  Plot {int(row['plot_id'])}: "
          f"mean_var={row['mean_variance']:.2f}, "
          f"var_of_var={row['variance_of_variance']:.3f}, "
          f"cv_var={row['cv_variance']:.2f}")

print("\nNormal plots (stable variance):")
normal = df_results[~df_results["anomaly_flag"]].sort_values("anomaly_score").head(5)
for _, row in normal.iterrows():
    print(f"  Plot {int(row['plot_id'])}: "
          f"mean_var={row['mean_variance']:.2f}, "
          f"var_of_var={row['variance_of_variance']:.3f}, "
          f"cv_var={row['cv_variance']:.2f}")

print("\n=== KEY LIMITATION ===")
print("Isolation Forest assumes i.i.d. points. Garden plots are NOT i.i.d.—")
print("adjacent plots share soil, water, pests. The algorithm cannot leverage")
print("spatial structure, so it may conflate rare-but-stable plots with")
print("genuinely unstable ones. A spatial-aware method (e.g., Local Outlier Factor")
print("with spatial distance kernels) would be more appropriate but is left as future work.")
