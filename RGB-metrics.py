import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
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

# Define Regularized Gradient Boosting model
regularized_gb_model = make_pipeline(
    StandardScaler(), 
    GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,  # Reduced learning rate
        max_depth=2,         # Reduced tree depth
        min_samples_split=5, # Require more samples to split
        min_samples_leaf=2,  # Require more samples in leaves
        subsample=0.8,       # Use only 80% of samples per tree
        random_state=42
    )
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

# Evaluate regularized gradient boosting model
print("Evaluating Regularized Gradient Boosting")
gb_results = evaluate_model_metrics(regularized_gb_model, X_train, X_test, y_train, y_test)

print("\nRegularized Gradient Boosting Performance Metrics:")
print("="*80)
print(gb_results['formatted_metrics'].to_string(index=False))

# Visualize predictions on test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, gb_results['y_test_pred'], alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.title('Regularized Gradient Boosting: Actual vs Predicted', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gb_actual_vs_predicted.png', dpi=300)
plt.show()

# Conduct cross-validation for more robust metrics
print("\nConducting Cross-Validation for More Robust Metrics")
print("="*80)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
y_cv_pred = cross_val_predict(regularized_gb_model, X, y, cv=cv)

cv_mse = mean_squared_error(y, y_cv_pred)
cv_rmse = np.sqrt(cv_mse)
cv_mae = mean_absolute_error(y, y_cv_pred)
cv_r2 = r2_score(y, y_cv_pred)

print(f"Cross-Validation MSE:  {cv_mse:.4f}")
print(f"Cross-Validation RMSE: {cv_rmse:.4f}")
print(f"Cross-Validation MAE:  {cv_mae:.4f}")
print(f"Cross-Validation R²:   {cv_r2:.4f}")

# Calculate residuals
residuals = y_test - gb_results['y_test_pred']

# Plot residuals
plt.figure(figsize=(12, 5))

# Residuals vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(gb_results['y_test_pred'], residuals, alpha=0.7)
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
plt.savefig('gb_residual_analysis.png', dpi=300)
plt.show()

# Extract feature importances
gb = regularized_gb_model.named_steps['gradientboostingregressor']
gb.fit(X_train, y_train)
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': gb.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print("="*80)
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance', fontsize=14)
plt.title('Feature Importances (Regularized Gradient Boosting)', fontsize=16)
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('gb_feature_importances.png', dpi=300)
plt.show()

# Compare with Ridge Regression results
print("\nComparing Regularized Gradient Boosting with Ridge Regression:")
print("="*80)

# Define Ridge model with best alpha from previous analysis
ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1, random_state=42))
ridge_results = evaluate_model_metrics(ridge_model, X_train, X_test, y_train, y_test)

# Create comparison table
comparison_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Ridge (Test)': [float(ridge_results['formatted_metrics']['Test Set'][i]) for i in range(4)],
    'Gradient Boosting (Test)': [float(gb_results['formatted_metrics']['Test Set'][i]) for i in range(4)]
})

# Calculate percentage improvement
comparison_df['Improvement (%)'] = ((comparison_df['Gradient Boosting (Test)'] - comparison_df['Ridge (Test)']) / 
                                    comparison_df['Ridge (Test)'] * 100)

# Format the comparison DataFrame
formatted_comparison = comparison_df.copy()
formatted_comparison['Ridge (Test)'] = formatted_comparison['Ridge (Test)'].apply(lambda x: f"{x:.4f}")
formatted_comparison['Gradient Boosting (Test)'] = formatted_comparison['Gradient Boosting (Test)'].apply(lambda x: f"{x:.4f}")
formatted_comparison['Improvement (%)'] = formatted_comparison['Improvement (%)'].apply(lambda x: f"{x:.2f}%")

print(formatted_comparison.to_string(index=False))