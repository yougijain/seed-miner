import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic: 12 plots, 8 seasons, with deliberate crop-rotation structural breaks
plots = np.arange(1, 13)
seasons = np.arange(0, 8)
data = []

for plot_id in plots:
    soil_quality = np.random.uniform(5, 9)  # fixed per plot
    rainfall_base = np.random.uniform(40, 80)  # mm, varies per season
    
    for season in seasons:
        rainfall = rainfall_base + np.random.normal(0, 5)
        
        # Crop type: rotate every 2 seasons
        crop_phase = (season // 2) % 3
        crops = ['tomato', 'legume', 'leafy']
        crop = crops[crop_phase]
        
        # STRUCTURAL BREAK: yield response to soil_quality inverts between crop phases
        if crop_phase == 0:  # tomato: high soil = high yield
            yield_val = 15 + 2.5 * soil_quality + 0.3 * rainfall + np.random.normal(0, 1.5)
        elif crop_phase == 1:  # legume: low soil quality fine (fixes nitrogen)
            yield_val = 20 - 1.2 * soil_quality + 0.25 * rainfall + np.random.normal(0, 1.5)
        else:  # leafy: moderate soil, heavy rainfall dependence
            yield_val = 12 + 0.8 * rainfall + np.random.normal(0, 1.5)
        
        yield_val = max(5, yield_val)
        data.append({
            'plot_id': plot_id,
            'season': season,
            'soil_quality': soil_quality,
            'rainfall': rainfall,
            'crop': crop,
            'yield': yield_val
        })

df = pd.DataFrame(data)
print("Data shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))

# Split: train on seasons 0-5, forecast season 6-7
train_df = df[df['season'] <= 5].copy()
test_df = df[df['season'] > 5].copy()

print(f"\n--- BASELINE: Naive Stationary Forecast (ignores rotation) ---")
# Fit linear model on all training data, ignoring crop type
X_train = train_df[['soil_quality', 'rainfall']].values
y_train = train_df['yield'].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model_naive = LinearRegression()
model_naive.fit(X_train_scaled, y_train)

X_test = test_df[['soil_quality', 'rainfall']].values
X_test_scaled = scaler.transform(X_test)
y_pred_naive = model_naive.predict(X_test_scaled)
mae_naive = np.mean(np.abs(y_pred_naive - test_df['yield'].values))
print(f"MAE (naive, stationary): {mae_naive:.3f}")
print(f"Predictions (first 4): {y_pred_naive[:4]}")
print(f"Actuals (first 4): {test_df['yield'].values[:4]}")

print(f"\n--- ROTATION-AWARE: Forecast using crop-phase-specific models ---")
# Train separate model per crop phase
phase_models = {}
for phase in range(3):
    phase_data = train_df[train_df['season'] // 2 == phase]
    if len(phase_data) > 1:
        X_p = phase_data[['soil_quality', 'rainfall']].values
        y_p = phase_data['yield'].values
        X_p_scaled = scaler.fit_transform(X_p)
        m = LinearRegression()
        m.fit(X_p_scaled, y_p)
        phase_models[phase] = (m, scaler)

# Forecast using correct phase model
y_pred_aware = []
for idx, row in test_df.iterrows():
    phase = (row['season'] // 2) % 3
    if phase in phase_models:
        model, sc = phase_models[phase]
        X_row = np.array([[row['soil_quality'], row['rainfall']]])
        X_row_scaled = sc.transform(X_row)
        pred = model.predict(X_row_scaled)[0]
    else:
        pred = y_train.mean()
    y_pred_aware.append(pred)

y_pred_aware = np.array(y_pred_aware)
mae_aware = np.mean(np.abs(y_pred_aware - test_df['yield'].values))
print(f"MAE (rotation-aware): {mae_aware:.3f}")
print(f"Predictions (first 4): {y_pred_aware[:4]}")
print(f"Actuals (first 4): {test_df['yield'].values[:4]}")
print(f"\nImprovement: {100 * (mae_naive - mae_aware) / mae_naive:.1f}% MAE reduction")
print(f"\nNote: Naive model fails because it averages over incompatible regimes.")
print(f"Rotation-aware model detects regime and selects appropriate coefficients.")
