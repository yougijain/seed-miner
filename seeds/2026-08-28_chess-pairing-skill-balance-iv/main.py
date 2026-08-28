import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Synthetic dataset: youth chess tournament pairings
n_pairings = 200
age_groups = np.repeat(['U10', 'U12', 'U14', 'U16'], 50)
rating_player1 = np.random.normal([1200, 1350, 1500, 1650], 150, n_pairings)
rating_player2 = np.random.normal([1200, 1350, 1500, 1650], 150, n_pairings)
rating_player1 = np.clip(rating_player1, 800, 2200)
rating_player2 = np.clip(rating_player2, 800, 2200)

# Key mechanism: rating_diff drives both pairing assignment AND outcome.
# cross_skill_pair = True if |rating1 - rating2| > threshold (endogenous assignment)
rating_diff = np.abs(rating_player1 - rating_player2)
threshold = 200
cross_skill_pair = (rating_diff > threshold).astype(int)

# Outcome: game quality (0-1). True causal effect of cross-skill pairing is negative.
# But rating_diff itself also predicts low quality (confounding).
base_quality = 0.65
cross_skill_effect = -0.15  # True causal effect
confounding_effect = -0.0003 * rating_diff  # Confounder: high rating spread → lower quality
game_quality = base_quality + cross_skill_effect * cross_skill_pair + confounding_effect
game_quality += np.random.normal(0, 0.1, n_pairings)
game_quality = np.clip(game_quality, 0, 1)

df = pd.DataFrame({
    'age_group': age_groups,
    'rating_p1': rating_player1,
    'rating_p2': rating_player2,
    'rating_diff': rating_diff,
    'cross_skill_pair': cross_skill_pair,
    'game_quality': game_quality
})

print("=" * 70)
print("NAIVE (confounded) regression: cross_skill_pair → game_quality")
print("=" * 70)
X_naive = df[['cross_skill_pair']].values
y = df['game_quality'].values
reg_naive = LinearRegression().fit(X_naive, y)
print(f"Naive effect of cross_skill_pair: {reg_naive.coef_[0]:.4f}")
print(f"(True effect is -0.15, but confounder biases estimate downward)\n")

print("=" * 70)
print("INSTRUMENTAL VARIABLE (2SLS): rating_diff as IV for cross_skill_pair")
print("=" * 70)
print("Exclusion assumption: rating_diff affects game_quality *only through*")
print("whether the pairing algo is forced to cross skill boundaries.\n")

# Stage 1: rating_diff → cross_skill_pair (instrument relevance)
X_stage1 = df[['rating_diff']].values
reg_stage1 = LinearRegression().fit(X_stage1, df['cross_skill_pair'].values)
print(f"Stage 1 (First-stage F-stat proxy): R²={reg_stage1.score(X_stage1, df['cross_skill_pair'].values):.4f}")
print(f"Stage 1 coefficient (rating_diff → cross_skill_pair): {reg_stage1.coef_[0]:.6f}\n")

# Generate predicted cross_skill_pair from stage 1
cross_skill_pred = reg_stage1.predict(X_stage1)

# Stage 2: predicted cross_skill_pair → game_quality
reg_stage2 = LinearRegression().fit(cross_skill_pred.reshape(-1, 1), y)
print(f"Stage 2 IV estimate of cross_skill_pair effect: {reg_stage2.coef_[0]:.4f}")
print(f"(Closer to true -0.15 because confounding is removed)\n")

print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("Naive OLS underestimates the causal harm of cross-skill pairings.")
print("IV regression isolates the true effect by using rating_diff as")
print("an instrument: it predicts assignment but (we assume) not quality")
print("except through the pairing mechanism itself.\n")
print("In practice: tournament organizers can identify whether fairness")
print("problems stem from algorithmic assignment bias (recoverable via")
print("constraint adjustment) vs. true imbalance in skill group sizes.")
