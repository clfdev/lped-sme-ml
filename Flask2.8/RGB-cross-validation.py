import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

# Load the new dataset with the engineered features
file_path = 'Dataset3.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Create a column called Inverse interatomic distance
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create a column called Lennard_Jones interatomic distance
# Lennard-Jones potential parameters
epsilon = 1
sigma = 1

# Calculate Lennard-Jones potential
data['Lennard_Jones interatomic distance'] = 4 * epsilon * (
    (sigma * data['Inverse interatomic distance']) ** 12 - 
    (sigma * data['Inverse interatomic distance']) ** 6
)

# Drop the index column, categorical variable, and the original interatomic distance column
data_cleaned = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'])

# Separate target (Y) and features (X)
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

# Use selected features
selected_features = ['Atomic charge A (MK)','Atomic charge B (MK)', 'Lennard_Jones interatomic distance']
X_selected = data_cleaned[selected_features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

# Test different random states
best_random_state = None
best_mean_r2 = -np.inf
results = []

# Define the hyperparameters for GradientBoostingRegressor
n_estimators = 100
learning_rate = 0.1
max_depth = 3

for random_state in range(10):  # Test random_states from 0 to 9
    # Initialize Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42  # Fixed random state for the model itself
    )
    
    # Perform cross-validation with current random_state
    cv = KFold(n_splits=4, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(model, X_scaled, Y, cv=cv, scoring='r2')
    cv_neg_mse = cross_val_score(model, X_scaled, Y, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_neg_mse)
    
    # Calculate means and stds
    mean_r2 = cv_r2.mean()
    std_r2 = cv_r2.std()
    mean_rmse = cv_rmse.mean()
    std_rmse = cv_rmse.std()
    
    # Store results
    results.append({
        'random_state': random_state,
        'r2_scores': cv_r2.round(4),
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'rmse_scores': cv_rmse.round(4),
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse
    })
    
    # Update best random_state if current one is better
    if mean_r2 > best_mean_r2:
        best_mean_r2 = mean_r2
        best_random_state = random_state

# Print results for all random states
print("\nGradient Boosting Results for different random states:")
print("=" * 80)
print(f"Model parameters: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
print("=" * 80)
for result in results:
    print(f"\nRandom State: {result['random_state']}")
    print(f"R² scores for each fold: {result['r2_scores']}")
    print(f"Mean R²: {result['mean_r2']:.4f} (+/- {result['std_r2'] * 2:.4f})")
    print(f"RMSE scores for each fold: {result['rmse_scores']}")
    print(f"Mean RMSE: {result['mean_rmse']:.4f} (+/- {result['std_rmse'] * 2:.4f})")

# Print summary of best random state
print("\nBest Random State Summary:")
print("=" * 80)
best_result = next(result for result in results if result['random_state'] == best_random_state)
print(f"Best Random State: {best_random_state}")
print(f"Best Mean R²: {best_result['mean_r2']:.4f}")
print(f"R² scores for best random state: {best_result['r2_scores']}")
print(f"RMSE scores for best random state: {best_result['rmse_scores']}")

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'random_state': [r['random_state'] for r in results],
    'mean_r2': [r['mean_r2'] for r in results],
    'std_r2': [r['std_r2'] for r in results],
    'mean_rmse': [r['mean_rmse'] for r in results],
    'std_rmse': [r['std_rmse'] for r in results]
})

print("\nSummary Statistics:")
print("=" * 80)
print(summary_df.sort_values(by='mean_r2', ascending=False))

# Train the final model using the best random state
best_cv = KFold(n_splits=4, shuffle=True, random_state=best_random_state)
final_model = GradientBoostingRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=42
)

# Get feature importances from the final model
final_model.fit(X_scaled, Y)
feature_importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances from Final Model:")
print("=" * 80)
print(feature_importances)

# Calculate total importance to get percentages
total_importance = feature_importances['Importance'].sum()
feature_importances['Percentage'] = (feature_importances['Importance'] / total_importance * 100).round(2)

print("\nFeature Importance Percentages:")
print("=" * 80)
for idx, row in feature_importances.iterrows():
    print(f"{row['Feature']}: {row['Percentage']}%")