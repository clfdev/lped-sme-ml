import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Consistent random state
RANDOM_STATE = 4

# Load dataset
df = pd.read_excel('Dataset3.xlsx')

# Feature selection
features = ['Interatomic distance', 'Atomic charge A (MK)', 'Atomic charge B (MK)']
target = 'LPED'

# Prepare data
X = df[features]
y = df[target]

# Define Original Regularized Gradient Boosting model
original_rgb_model = make_pipeline(
    StandardScaler(), 
    GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
)

# Function to evaluate cross-validation scores for different random states
def evaluate_random_states(model, X, y, n_splits=4, random_states=range(10)):
    results = []
    
    for random_state in random_states:
        # Create KFold with current random_state
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Calculate R² scores
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        # Calculate RMSE scores (negative MSE then take square root)
        neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        # Store results
        results.append({
            'random_state': random_state,
            'r2_scores': r2_scores,
            'mean_r2': r2_scores.mean(),
            'std_r2': r2_scores.std(),
            'rmse_scores': rmse_scores,
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std()
        })
    
    return results

# Evaluate RGB model with different random states
print("Evaluating Regularized Gradient Boosting with different random states...")
rgb_results = evaluate_random_states(original_rgb_model, X, y)

# Print detailed results for each random state
for result in rgb_results:
    print(f"\nRandom State: {result['random_state']}")
    print(f"R² scores: {[f'{score:.4f}' for score in result['r2_scores']]}")
    print(f"Mean R²: {result['mean_r2']:.4f} (±{result['std_r2']:.4f})")
    print(f"RMSE scores: {[f'{score:.4f}' for score in result['rmse_scores']]}")
    print(f"Mean RMSE: {result['mean_rmse']:.4f} (±{result['std_rmse']:.4f})")

# Create summary table
summary_df = pd.DataFrame({
    'Random State': [r['random_state'] for r in rgb_results],
    'Mean R²': [r['mean_r2'] for r in rgb_results],
    'R² Std Dev': [r['std_r2'] for r in rgb_results],
    'Mean RMSE': [r['mean_rmse'] for r in rgb_results],
    'RMSE Std Dev': [r['std_rmse'] for r in rgb_results]
})

# Find best random state based on R²
best_random_state_index = summary_df['Mean R²'].idxmax()
best_random_state = summary_df.loc[best_random_state_index, 'Random State']
best_result = rgb_results[best_random_state_index]

print("\nSummary Table (sorted by Mean R²):")
print(summary_df.sort_values('Mean R²', ascending=False).to_string(index=False))

print("\nBest Random State Performance:")
print(f"Random State: {int(best_random_state)}")
print(f"R² scores: {[f'{score:.4f}' for score in best_result['r2_scores']]}")
print(f"Mean R²: {best_result['mean_r2']:.4f}")
print(f"RMSE scores: {[f'{score:.4f}' for score in best_result['rmse_scores']]}")
print(f"Mean RMSE: {best_result['mean_rmse']:.4f}")

# Create comparison visualization for random states
plt.figure(figsize=(12, 8))

# Plot mean R² with error bars
plt.subplot(2, 1, 1)
random_states = [r['random_state'] for r in rgb_results]
mean_r2 = [r['mean_r2'] for r in rgb_results]
std_r2 = [r['std_r2'] for r in rgb_results]

plt.errorbar(random_states, mean_r2, yerr=std_r2, fmt='o-', capsize=5, linewidth=2, markersize=10)
plt.axhline(y=np.mean(mean_r2), color='r', linestyle='--', label=f'Average: {np.mean(mean_r2):.4f}')
plt.xlabel('Random State', fontsize=14)
plt.ylabel('Mean R² Score', fontsize=14)
plt.title('R² Performance Across Random States', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot mean RMSE with error bars
plt.subplot(2, 1, 2)
mean_rmse = [r['mean_rmse'] for r in rgb_results]
std_rmse = [r['std_rmse'] for r in rgb_results]

plt.errorbar(random_states, mean_rmse, yerr=std_rmse, fmt='o-', capsize=5, linewidth=2, markersize=10)
plt.axhline(y=np.mean(mean_rmse), color='r', linestyle='--', label=f'Average: {np.mean(mean_rmse):.4f}')
plt.xlabel('Random State', fontsize=14)
plt.ylabel('Mean RMSE', fontsize=14)
plt.title('RMSE Performance Across Random States', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('rgb_random_state_performance.png', dpi=300)
plt.show()

# Additionally, show the distribution of R² scores across folds with the best random state
plt.figure(figsize=(10, 6))
best_r2_scores = best_result['r2_scores']
plt.bar(range(len(best_r2_scores)), best_r2_scores, color='steelblue')
plt.axhline(y=best_result['mean_r2'], color='r', linestyle='--', 
            label=f'Mean R²: {best_result["mean_r2"]:.4f}')
plt.xlabel('Fold', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.title(f'R² Scores Across Folds (Random State={int(best_random_state)})', fontsize=16)
plt.xticks(range(len(best_r2_scores)), [f'Fold {i+1}' for i in range(len(best_r2_scores))])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('rgb_best_folds_r2.png', dpi=300)
plt.show()