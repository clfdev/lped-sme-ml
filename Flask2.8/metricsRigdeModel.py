import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Define Ridge Regression model with regularization
ridge_model = make_pipeline(
    StandardScaler(), 
    Ridge(alpha=1.0, random_state=42)  # Alpha is the regularization strength
)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Function to evaluate model metrics
def evaluate_model_metrics(model, X_train, X_test, y_train, y_test):
    # Fit model
    model.fit(X_train, y_train)
    
    # Get predictions on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics on training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics on test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Organize metrics into DataFrames
    train_metrics = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Training Set': [train_mse, train_rmse, train_mae, train_r2]
    })
    
    test_metrics = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Test Set': [test_mse, test_rmse, test_mae, test_r2]
    })
    
    # Combine into one DataFrame
    metrics_df = pd.merge(train_metrics, test_metrics, on='Metric')
    
    # Calculate percentage difference between train and test
    metrics_df['Difference (%)'] = ((metrics_df['Test Set'] - metrics_df['Training Set']) / 
                                   metrics_df['Training Set'] * 100)
    
    # Format the metrics DataFrame for better readability
    formatted_metrics = metrics_df.copy()
    formatted_metrics['Training Set'] = formatted_metrics['Training Set'].apply(lambda x: f"{x:.4f}")
    formatted_metrics['Test Set'] = formatted_metrics['Test Set'].apply(lambda x: f"{x:.4f}")
    formatted_metrics['Difference (%)'] = formatted_metrics['Difference (%)'].apply(lambda x: f"{x:.2f}%")
    
    return {
        'metrics_df': metrics_df,
        'formatted_metrics': formatted_metrics,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

# Evaluate model with default alpha
print("Evaluating Ridge Regression with alpha=1.0")
ridge_results = evaluate_model_metrics(ridge_model, X_train, X_test, y_train, y_test)

print("\nModel Performance Metrics:")
print("="*80)
print(ridge_results['formatted_metrics'].to_string(index=False))

# Visualize predictions on test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ridge_results['y_test_pred'], alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.title('Ridge Regression: Actual vs Predicted', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ridge_actual_vs_predicted.png', dpi=300)
plt.show()

# Conduct cross-validation for more robust metrics
print("\nConducting Cross-Validation for More Robust Metrics")
print("="*80)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
y_cv_pred = cross_val_predict(ridge_model, X, y, cv=cv)

cv_mse = mean_squared_error(y, y_cv_pred)
cv_rmse = np.sqrt(cv_mse)
cv_mae = mean_absolute_error(y, y_cv_pred)
cv_r2 = r2_score(y, y_cv_pred)

print(f"Cross-Validation MSE:  {cv_mse:.4f}")
print(f"Cross-Validation RMSE: {cv_rmse:.4f}")
print(f"Cross-Validation MAE:  {cv_mae:.4f}")
print(f"Cross-Validation R²:   {cv_r2:.4f}")

# Calculate residuals
residuals = y_test - ridge_results['y_test_pred']

# Plot residuals
plt.figure(figsize=(12, 5))

# Residuals vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(ridge_results['y_test_pred'], residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs Predicted Values', fontsize=14)
plt.grid(True, alpha=0.3)

# Histogram of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=10, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Residual Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of Residuals', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_residual_analysis.png', dpi=300)
plt.show()

# Try different alpha values to find optimal regularization
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
alpha_results = {}

for alpha in alphas:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=42))
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    alpha_results[alpha] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }

# Find best alpha based on test R²
best_alpha = max(alpha_results.items(), key=lambda x: x[1]['test_r2'])[0]

# Create comparison table
alpha_comparison = pd.DataFrame({
    'Alpha': list(alpha_results.keys()),
    'Training R²': [alpha_results[a]['train_r2'] for a in alphas],
    'Test R²': [alpha_results[a]['test_r2'] for a in alphas],
    'Test MSE': [alpha_results[a]['test_mse'] for a in alphas],
    'Test RMSE': [alpha_results[a]['test_rmse'] for a in alphas],
    'Test MAE': [alpha_results[a]['test_mae'] for a in alphas]
})

print("\nRegularization Parameter Comparison:")
print("="*80)
pd.set_option('display.float_format', '{:.4f}'.format)
print(alpha_comparison)

print(f"\nBest Alpha Value: {best_alpha}")
print(f"Best Test R²: {alpha_results[best_alpha]['test_r2']:.4f}")
print(f"Corresponding Test RMSE: {alpha_results[best_alpha]['test_rmse']:.4f}")

# Plot metrics vs alpha
plt.figure(figsize=(12, 8))

# R² vs alpha
plt.subplot(2, 2, 1)
plt.semilogx(alphas, [alpha_results[a]['train_r2'] for a in alphas], 'o-', label='Training R²')
plt.semilogx(alphas, [alpha_results[a]['test_r2'] for a in alphas], 'o-', label='Test R²')
plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'Best Alpha: {best_alpha}')
plt.xlabel('Alpha (regularization strength)', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('R² vs Alpha', fontsize=14)
plt.legend()
plt.grid(True)

# MSE vs alpha
plt.subplot(2, 2, 2)
plt.semilogx(alphas, [alpha_results[a]['test_mse'] for a in alphas], 'o-')
plt.axvline(x=best_alpha, color='r', linestyle='--')
plt.xlabel('Alpha (regularization strength)', fontsize=12)
plt.ylabel('Test MSE', fontsize=12)
plt.title('Test MSE vs Alpha', fontsize=14)
plt.grid(True)

# RMSE vs alpha
plt.subplot(2, 2, 3)
plt.semilogx(alphas, [alpha_results[a]['test_rmse'] for a in alphas], 'o-')
plt.axvline(x=best_alpha, color='r', linestyle='--')
plt.xlabel('Alpha (regularization strength)', fontsize=12)
plt.ylabel('Test RMSE', fontsize=12)
plt.title('Test RMSE vs Alpha', fontsize=14)
plt.grid(True)

# MAE vs alpha
plt.subplot(2, 2, 4)
plt.semilogx(alphas, [alpha_results[a]['test_mae'] for a in alphas], 'o-')
plt.axvline(x=best_alpha, color='r', linestyle='--')
plt.xlabel('Alpha (regularization strength)', fontsize=12)
plt.ylabel('Test MAE', fontsize=12)
plt.title('Test MAE vs Alpha', fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('ridge_metrics_vs_alpha.png', dpi=300)
plt.show()

# Evaluate final model with best alpha
final_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, random_state=42))
final_results = evaluate_model_metrics(final_model, X_train, X_test, y_train, y_test)

print("\nFinal Model (Best Alpha) Performance Metrics:")
print("="*80)
print(final_results['formatted_metrics'].to_string(index=False))

# Extract and visualize coefficients from the best model
final_model.fit(X_train, y_train)
ridge_coefs = final_model.named_steps['ridge'].coef_
# Need to transform the coefficients back to original scale
scaler = final_model.named_steps['standardscaler']
scaled_coefs = ridge_coefs / scaler.scale_

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': scaled_coefs
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nModel Coefficients (sorted by absolute value):")
print("="*80)
for idx, row in coef_df.iterrows():
    print(f"{row['Feature']}: {row['Coefficient']:.6f}")

# Visualize coefficients
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value', fontsize=14)
plt.title('Ridge Regression Coefficients', fontsize=16)
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('ridge_coefficients.png', dpi=300)
plt.show()