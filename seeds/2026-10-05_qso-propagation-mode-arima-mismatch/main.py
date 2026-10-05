import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Synthetic QSO log with hidden mode-switching
np.random.seed(42)

# Build a time series where operator switches between contest-mode (rapid, ~30 QSOs/hour)
# and ragchew-mode (slow, ~8 QSOs/hour) based on (unobserved) propagation band quality
n_hours = 200
hours = np.arange(n_hours)

# Ground truth: hidden mode indicator (0=ragchew, 1=contest)
# Switches based on synthetic "band quality" that we do NOT observe
band_quality = np.sin(hours / 40) + 0.3 * np.random.normal(size=n_hours)
mode = (band_quality > 0).astype(int)

# QSO counts: mode-dependent with noise
qso_counts = np.where(
    mode == 1,
    np.random.poisson(30, n_hours),  # contest mode
    np.random.poisson(8, n_hours)    # ragchew mode
)

df = pd.DataFrame({
    'hour': hours,
    'qso_count': qso_counts,
    'true_mode': mode  # for validation only
})

# Fit ARIMA(1,1,1) — standard contest-log forecasting approach
model = ARIMA(df['qso_count'], order=(1, 1, 1))
results = model.fit()
df['forecast'] = results.fittedvalues
df['residual'] = results.resid.values

# Large residuals indicate mode-switches (ARIMA assumes stationarity; mode-switching violates it)
residual_abs = np.abs(df['residual'].fillna(0))

# Cluster residual magnitudes: high-residual windows should separate from low
residual_clusters = KMeans(n_clusters=2, random_state=42).fit_predict(
    residual_abs.values.reshape(-1, 1)
)

# Inferred mode: high-residual cluster should correlate with true mode-switches
df['inferred_cluster'] = residual_clusters

# Check: does high-residual cluster align with true mode transitions?
mode_transition_mask = np.abs(np.diff(df['true_mode'].values, prepend=df['true_mode'].iloc[0])) > 0
df['is_transition'] = mode_transition_mask.astype(int)

# Metrics
high_residual_cluster = residual_clusters[residual_abs.argmax()]
detected_transitions = (residual_clusters == high_residual_cluster).astype(int)

print("=== QSO Propagation Mode Inference via ARIMA Residual Clustering ===")
print(f"Sample: {len(df)} hours of QSO activity")
print(f"\nTrue mode transitions in log: {df['is_transition'].sum()}")
print(f"High-residual cluster detections: {(residual_clusters == high_residual_cluster).sum()}")
print(f"\nARIMA model AIC: {results.aic:.2f}")
print(f"Mean |residual|: {residual_abs.mean():.2f}")
print(f"Std |residual|: {residual_abs.std():.2f}")

# Alignment: how many true transitions fall into high-residual cluster?
true_transition_in_high = (
    df.loc[df['is_transition'] == 1, 'inferred_cluster'] == high_residual_cluster
).mean()
print(f"\nTrue transitions caught in high-residual cluster: {true_transition_in_high:.1%}")

# Show sample window
window = df.iloc[50:80]
print(f"\n--- Window hours 50-80 ---")
print(window[['hour', 'qso_count', 'forecast', 'residual', 'true_mode', 'inferred_cluster']].to_string(index=False))
