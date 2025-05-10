import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

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

# Adjust train sizes based on dataset size
train_sizes = np.linspace(0.1, 1.0, min(6, num_samples // 10))

# Define Ridge Regression model with regularization
ridge_model = make_pipeline(
    StandardScaler(), 
    Ridge(alpha=1.0, random_state=42)  # Alpha is the regularization strength
)

# Split data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Function to analyze learning curves for both MSE and R²
def analyze_learning_curves(model, X, y, model_name, random_state=RANDOM_STATE):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)
    
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
    
    # Create figure for both metrics
    plt.figure(figsize=(12, 10))
    
    # Plot MSE learning curves
    plt.subplot(2, 1, 1)
    plt.title(f"Learning Curves (MSE) - {model_name}", fontsize=16)
    plt.xlabel("Training examples", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    
    plt.fill_between(train_sizes_abs, train_mse_mean - train_mse_std,
                     train_mse_mean + train_mse_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_mse_mean - test_mse_std,
                     test_mse_mean + test_mse_std, alpha=0.1, color='g')
    
    plt.plot(train_sizes_abs, train_mse_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes_abs, test_mse_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    
    # Plot R² learning curves
    plt.subplot(2, 1, 2)
    plt.title(f"Learning Curves (R²) - {model_name}", fontsize=16)
    plt.xlabel("Training examples", fontsize=14)
    plt.ylabel("R² Score", fontsize=14)
    
    plt.fill_between(train_sizes_abs, train_r2_mean - train_r2_std,
                     train_r2_mean + train_r2_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_r2_mean - test_r2_std,
                     test_r2_mean + test_r2_std, alpha=0.1, color='g')
    
    plt.plot(train_sizes_abs, train_r2_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes_abs, test_r2_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_learning_curves.png', dpi=300)
    plt.show()
    
    # Fit model and calculate final performance metrics
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_final_mse = mean_squared_error(y_train, y_train_pred)
    train_final_r2 = r2_score(y_train, y_train_pred)
    
    test_final_mse = mean_squared_error(y_test, y_test_pred)
    test_final_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate overfitting metrics
    mse_ratio = test_final_mse / train_final_mse if train_final_mse > 0 else float('inf')
    r2_gap = train_final_r2 - test_final_r2
    
    # Print analysis
    print("\n" + "="*50)
    print(f"LEARNING CURVE ANALYSIS - {model_name}")
    print("="*50)
    
    print("\n1. Final Cross-Validation Performance:")
    print(f"   Final CV MSE: {test_mse_mean[-1]:.4f} (±{test_mse_std[-1]:.4f})")
    print(f"   Final CV R²: {test_r2_mean[-1]:.4f} (±{test_r2_std[-1]:.4f})")
    
    print("\n2. Final Train/Test Performance:")
    print(f"   Training MSE: {train_final_mse:.4f}")
    print(f"   Test MSE: {test_final_mse:.4f}")
    print(f"   Training R²: {train_final_r2:.4f}")
    print(f"   Test R²: {test_final_r2:.4f}")
    
    print("\n3. Overfitting Metrics:")
    print(f"   MSE Ratio (Test/Train): {mse_ratio:.2f}x")
    print(f"   R² Gap (Train - Test): {r2_gap:.4f}")
    
    # Calculate learning curve characteristics
    final_cv_train_mse_gap = test_mse_mean[-1] - train_mse_mean[-1]
    final_cv_train_r2_gap = train_r2_mean[-1] - test_r2_mean[-1]
    
    # Calculate if curves are converging
    is_converging_mse = abs(test_mse_mean[-1] - train_mse_mean[-1]) < abs(test_mse_mean[0] - train_mse_mean[0])
    is_converging_r2 = abs(train_r2_mean[-1] - test_r2_mean[-1]) < abs(train_r2_mean[0] - test_r2_mean[0])
    
    print("\n4. Learning Curve Characteristics:")
    print(f"   Final CV-Train MSE Gap: {final_cv_train_mse_gap:.4f}")
    print(f"   Final CV-Train R² Gap: {final_cv_train_r2_gap:.4f}")
    print(f"   MSE curves are {'converging' if is_converging_mse else 'not converging'}")
    print(f"   R² curves are {'converging' if is_converging_r2 else 'not converging'}")
    
    # Assess model based on gaps
    if mse_ratio < 1.5 and r2_gap < 0.15:
        balance_assessment = "Well-balanced model with good generalization"
    elif mse_ratio < 2.5 and r2_gap < 0.3:
        balance_assessment = "Moderately balanced model with acceptable generalization"
    else:
        balance_assessment = "Shows signs of overfitting, may need further regularization"
    
    print(f"\n5. Model Balance Assessment: {balance_assessment}")
    
    # Compare with curve endpoint
    if abs(test_final_r2 - test_r2_mean[-1]) > 0.1:
        print(f"\nNote: There's a significant difference between cross-validation R² ({test_r2_mean[-1]:.4f}) "
              f"and test set R² ({test_final_r2:.4f}), which may indicate sample variability.")
    
    return {
        'train_sizes': train_sizes_abs,
        'train_mse_mean': train_mse_mean,
        'test_mse_mean': test_mse_mean,
        'train_r2_mean': train_r2_mean,
        'test_r2_mean': test_r2_mean,
        'final_test_r2': test_final_r2,
        'final_test_mse': test_final_mse
    }

# Run analysis for Ridge Regression
ridge_results = analyze_learning_curves(ridge_model, X, y, "Ridge Regression")

# Plot predictions vs actual values
plt.figure(figsize=(12, 5))

# Training set predictions
y_train_pred = ridge_model.predict(X_train)
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.title(f'Training Set (R² = {r2_score(y_train, y_train_pred):.4f})', fontsize=14)

# Test set predictions
y_test_pred = ridge_model.predict(X_test)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.title(f'Test Set (R² = {r2_score(y_test, y_test_pred):.4f})', fontsize=14)

plt.tight_layout()
plt.savefig('ridge_regression_predictions.png', dpi=300)
plt.show()

# Try different alpha values to examine regularization impact
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
results = {}

for alpha in alphas:
    print(f"\nEvaluating Ridge Regression with alpha={alpha}")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=42))
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results[alpha] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': train_r2 - test_r2
    }
    
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Gap: {train_r2 - test_r2:.4f}")

# Create alpha comparison plot
alphas_list = list(results.keys())
train_r2_values = [results[a]['train_r2'] for a in alphas_list]
test_r2_values = [results[a]['test_r2'] for a in alphas_list]
gaps = [results[a]['gap'] for a in alphas_list]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.semilogx(alphas_list, train_r2_values, 'o-', label='Training R²')
plt.semilogx(alphas_list, test_r2_values, 'o-', label='Test R²')
plt.xlabel('Alpha (regularization strength)', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('R² vs Regularization Strength', fontsize=14)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogx(alphas_list, gaps, 'o-')
plt.xlabel('Alpha (regularization strength)', fontsize=12)
plt.ylabel('R² Gap (Train - Test)', fontsize=12)
plt.title('Overfitting vs Regularization Strength', fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('ridge_alpha_comparison.png', dpi=300)
plt.show()

# Identify optimal alpha
best_alpha = min(results.items(), key=lambda x: -x[1]['test_r2'])[0]
print(f"\nOptimal Alpha: {best_alpha}")
print(f"Best Test R²: {results[best_alpha]['test_r2']:.4f}")
print(f"Training R² at Best Alpha: {results[best_alpha]['train_r2']:.4f}")
print(f"R² Gap at Best Alpha: {results[best_alpha]['gap']:.4f}")