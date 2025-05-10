import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

# Validate dataset size
num_samples = X.shape[0]
min_required_samples = 50  # Adjust as needed

if num_samples < min_required_samples:
    raise ValueError(f"Dataset is too small ({num_samples} samples). At least {min_required_samples} are required.")

# Split data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Function to create learning curves and evaluate models
def analyze_model(model, model_name, X, y, X_train, X_test, y_train, y_test):
    # Set up cross-validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    
    # Adjust train sizes based on dataset size - fewer points for clearer spacing
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    # Compute MSE learning curve
    train_sizes_abs, train_scores_mse, test_scores_mse = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='neg_mean_squared_error'
    )
    
    # Convert negative MSE to positive
    train_mse = -train_scores_mse
    test_mse = -test_scores_mse
    
    # Compute R² learning curve
    _, train_scores_r2, test_scores_r2 = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='r2'
    )
    
    # Calculate means and standard deviations
    train_mse_mean = np.mean(train_mse, axis=1)
    train_mse_std = np.std(train_mse, axis=1)
    test_mse_mean = np.mean(test_mse, axis=1)
    test_mse_std = np.std(test_mse, axis=1)
    
    train_r2_mean = np.mean(train_scores_r2, axis=1)
    train_r2_std = np.std(train_scores_r2, axis=1)
    test_r2_mean = np.mean(test_scores_r2, axis=1)
    test_r2_std = np.std(test_scores_r2, axis=1)
    
    # Create MSE learning curve plot
    plt.figure(figsize=(12, 6))
    plt.title(f"Learning Curves (MSE) - {model_name}", fontsize=18)
    plt.xlabel("Training examples", fontsize=16)
    plt.ylabel("Mean Squared Error", fontsize=16)
    
    plt.fill_between(train_sizes_abs, train_mse_mean - train_mse_std,
                     train_mse_mean + train_mse_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_mse_mean - test_mse_std,
                     test_mse_mean + test_mse_std, alpha=0.1, color='g')
    
    plt.plot(train_sizes_abs, train_mse_mean, 'o-', color='r', label='Training score', markersize=8)
    plt.plot(train_sizes_abs, test_mse_mean, 'o-', color='g', label='Cross-validation score', markersize=8)
    
    plt.xticks(train_sizes_abs, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'{model_name.replace(" ", "_")}_MSE_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create R² learning curve plot
    plt.figure(figsize=(12, 6))
    plt.title(f"Learning Curves (R²) - {model_name}", fontsize=18)
    plt.xlabel("Training examples", fontsize=16)
    plt.ylabel("R² Score", fontsize=16)
    
    plt.fill_between(train_sizes_abs, train_r2_mean - train_r2_std,
                     train_r2_mean + train_r2_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_r2_mean - test_r2_std,
                     test_r2_mean + test_r2_std, alpha=0.1, color='g')
    
    plt.plot(train_sizes_abs, train_r2_mean, 'o-', color='r', label='Training score', markersize=8)
    plt.plot(train_sizes_abs, test_r2_mean, 'o-', color='g', label='Cross-validation score', markersize=8)
    
    plt.xticks(train_sizes_abs, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'{model_name.replace(" ", "_")}_R2_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Fit model and get predictions
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Create metrics table
    metrics = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Training Set': [train_mse, train_rmse, train_mae, train_r2],
        'Test Set': [test_mse, test_rmse, test_mae, test_r2]
    })
    
    metrics['Difference (%)'] = ((metrics['Test Set'] - metrics['Training Set']) / 
                               metrics['Training Set'] * 100)
    
    # Format the metrics
    formatted_metrics = metrics.copy()
    formatted_metrics['Training Set'] = formatted_metrics['Training Set'].apply(lambda x: f"{x:.4f}")
    formatted_metrics['Test Set'] = formatted_metrics['Test Set'].apply(lambda x: f"{x:.4f}")
    formatted_metrics['Difference (%)'] = formatted_metrics['Difference (%)'].apply(lambda x: f"{x:.2f}%")
    
    # Calculate overfitting metrics
    mse_ratio = test_mse / train_mse if train_mse > 0 else float('inf')
    r2_gap = train_r2 - test_r2
    
    # Print results
    print("\n" + "="*80)
    print(f"MODEL ANALYSIS - {model_name}")
    print("="*80)
    
    print("\nModel Performance Metrics:")
    print(formatted_metrics.to_string(index=False))
    
    print("\nOverfitting Indicators:")
    print(f"MSE Ratio (Test/Train): {mse_ratio:.2f}x")
    print(f"R² Gap (Train - Test): {r2_gap:.4f}")
    
    # Analyze convergence from learning curves
    final_train_cv_mse_gap = test_mse_mean[-1] - train_mse_mean[-1]
    final_train_cv_r2_gap = train_r2_mean[-1] - test_r2_mean[-1]
    
    is_converging_mse = abs(test_mse_mean[-1] - train_mse_mean[-1]) < abs(test_mse_mean[0] - train_mse_mean[0])
    is_converging_r2 = abs(train_r2_mean[-1] - test_r2_mean[-1]) < abs(train_r2_mean[0] - test_r2_mean[0])
    
    print("\nLearning Curve Analysis:")
    print(f"Final CV-Train MSE Gap: {final_train_cv_mse_gap:.4f}")
    print(f"Final CV-Train R² Gap: {final_train_cv_r2_gap:.4f}")
    print(f"MSE curves are {'converging' if is_converging_mse else 'not converging'}")
    print(f"R² curves are {'converging' if is_converging_r2 else 'not converging'}")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'{model_name}: Actual vs Predicted (R² = {test_r2:.4f})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_actual_vs_predicted.png', dpi=300)
    plt.show()
    
    return {
        'model': model,
        'train_sizes': train_sizes_abs,
        'train_mse_mean': train_mse_mean,
        'test_mse_mean': test_mse_mean,
        'train_r2_mean': train_r2_mean,
        'test_r2_mean': test_r2_mean,
        'metrics': metrics,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'mse_ratio': mse_ratio,
        'r2_gap': r2_gap
    }

# Define several refined Gradient Boosting models
models = {
    "Original Regularized GB": make_pipeline(
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
    ),
    
    "Further Regularized GB": make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.02,     # Reduced further
            max_depth=1,            # Even smaller trees
            min_samples_split=5,
            min_samples_leaf=3,     # Increased
            subsample=0.7,          # Reduced
            random_state=42
        )
    ),
    
    "GB with Early Stopping": make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(
            n_estimators=500,           # More trees, but will be limited by early stopping
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            validation_fraction=0.2,    # Use 20% of training data for validation
            n_iter_no_change=10,        # Stop if no improvement for 10 iterations
            tol=0.001,                  # Tolerance for improvement
            random_state=42
        )
    ),
    
    "Balanced GB": make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,         # Middle ground
            max_depth=2,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.85,             # Slight increase
            max_features=0.8,           # Use subset of features per tree
            random_state=42
        )
    )
}

# Analyze each model
results = {}
for name, model in models.items():
    print(f"\nAnalyzing {name}...")
    results[name] = analyze_model(model, name, X, y, X_train, X_test, y_train, y_test)

# Create comparison table
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Test R²': [results[m]['test_r2'] for m in results],
    'Test MSE': [results[m]['test_mse'] for m in results],
    'MSE Ratio': [results[m]['mse_ratio'] for m in results],
    'R² Gap': [results[m]['r2_gap'] for m in results]
}).sort_values(by='Test R²', ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
pd.set_option('display.float_format', '{:.4f}'.format)
print(comparison)

# Find best model
best_model_name = comparison.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\nBest model: {best_model_name}")
print(f"Test R²: {comparison.iloc[0]['Test R²']}")
print(f"Test MSE: {comparison.iloc[0]['Test MSE']}")

# Plot comparison of learning curves (R²)
plt.figure(figsize=(12, 8))
plt.title("Comparison of Learning Curves (R²)", fontsize=18)
plt.xlabel("Training examples", fontsize=16)
plt.ylabel("Cross-validation R² Score", fontsize=16)

for name in results:
    plt.plot(results[name]['train_sizes'], results[name]['test_r2_mean'], 'o-', label=name)

plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curve_r2_comparison.png', dpi=300)
plt.show()

# Plot comparison of learning curves (MSE)
plt.figure(figsize=(12, 8))
plt.title("Comparison of Learning Curves (MSE)", fontsize=18)
plt.xlabel("Training examples", fontsize=16)
plt.ylabel("Cross-validation MSE", fontsize=16)

for name in results:
    plt.plot(results[name]['train_sizes'], results[name]['test_mse_mean'], 'o-', label=name)

plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curve_mse_comparison.png', dpi=300)
plt.show()

# Extract and visualize feature importances from best model
if hasattr(best_model.named_steps['gradientboostingregressor'], 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.named_steps['gradientboostingregressor'].feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances (Best Model):")
    print("="*80)
    print(feature_importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'])
    plt.xlabel('Importance', fontsize=14)
    plt.title(f'Feature Importances - {best_model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig('best_model_feature_importances.png', dpi=300)
    plt.show()