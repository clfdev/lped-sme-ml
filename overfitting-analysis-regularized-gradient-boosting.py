import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

# Define Regularized Gradient Boosting model
regularized_model = make_pipeline(
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

# Split data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Function to plot learning curve and analyze overfitting
def analyze_overfitting(model, X_train, y_train, X_test, y_test):
    # Compute learning curve
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=train_sizes,
        scoring='neg_mean_squared_error'
    )
    
    # Convert negative MSE to positive
    train_mse = -train_scores
    test_mse = -test_scores
    
    # Calculate R² scores for each fold and size
    train_r2 = []
    test_r2 = []
    
    for i, size in enumerate(train_sizes_abs):
        # Calculate mean R² for training folds
        train_r2_fold = []
        test_r2_fold = []
        
        for j in range(len(train_scores[i])):
            # Placeholder for R² values - will estimate based on MSE improvement
            train_r2_fold.append(1 - (train_mse[i][j] / np.var(y_train)))
            test_r2_fold.append(1 - (test_mse[i][j] / np.var(y_train)))
        
        train_r2.append(train_r2_fold)
        test_r2.append(test_r2_fold)
    
    # Calculate means and standard deviations
    train_mse_mean = np.mean(train_mse, axis=1)
    train_mse_std = np.std(train_mse, axis=1)
    test_mse_mean = np.mean(test_mse, axis=1)
    test_mse_std = np.std(test_mse, axis=1)
    
    train_r2_mean = np.mean(train_r2, axis=1)
    train_r2_std = np.std(train_r2, axis=1)
    test_r2_mean = np.mean(test_r2, axis=1)
    test_r2_std = np.std(test_r2, axis=1)
    
    # Fit the model on full training data and evaluate
    model.fit(X_train, y_train)
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics on training and test sets
    train_final_mse = mean_squared_error(y_train, y_train_pred)
    train_final_r2 = r2_score(y_train, y_train_pred)
    train_final_rmse = np.sqrt(train_final_mse)
    
    test_final_mse = mean_squared_error(y_test, y_test_pred)
    test_final_r2 = r2_score(y_test, y_test_pred)
    test_final_rmse = np.sqrt(test_final_mse)
    
    # Calculate overfitting ratios
    mse_overfitting_ratio = test_final_mse / train_final_mse if train_final_mse > 0 else float('inf')
    r2_gap = train_final_r2 - test_final_r2
    
    # Create figure for MSE Learning Curve
    plt.figure(figsize=(12, 8))
    
    # Plot MSE learning curves
    plt.subplot(2, 1, 1)
    plt.title("Learning Curves (MSE)", fontsize=16)
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
    plt.title("Learning Curves (R²)", fontsize=16)
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
    plt.savefig('learning_curves_both_metrics.png', dpi=300)
    plt.show()
    
    # Plot predictions vs actual values
    plt.figure(figsize=(12, 5))
    
    # Training set predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title(f'Training Set (R² = {train_final_r2:.4f})', fontsize=14)
    
    # Test set predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title(f'Test Set (R² = {test_final_r2:.4f})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('train_test_predictions.png', dpi=300)
    plt.show()
    
    # Create detailed overfitting analysis report
    print("\n" + "="*50)
    print("OVERFITTING ANALYSIS REPORT")
    print("="*50)
    
    print("\n1. Final Model Performance Metrics:")
    print(f"   Training MSE: {train_final_mse:.4f}")
    print(f"   Test MSE:     {test_final_mse:.4f}")
    print(f"   Training R²:  {train_final_r2:.4f}")
    print(f"   Test R²:      {test_final_r2:.4f}")
    
    print("\n2. Overfitting Indicators:")
    print(f"   MSE Ratio (Test/Train): {mse_overfitting_ratio:.2f}x")
    print(f"   R² Gap (Train - Test): {r2_gap:.4f}")
    
    # Analyze overfitting level
    if mse_overfitting_ratio < 1.2 and r2_gap < 0.1:
        overfitting_level = "Minimal"
        details = "The model generalizes very well with almost identical performance on training and test data."
    elif mse_overfitting_ratio < 2 and r2_gap < 0.2:
        overfitting_level = "Low"
        details = "The model shows good generalization with only slight differences between training and test performance."
    elif mse_overfitting_ratio < 3 and r2_gap < 0.3:
        overfitting_level = "Moderate"
        details = "Some overfitting is present, but the model still generalizes reasonably well."
    elif mse_overfitting_ratio < 5 and r2_gap < 0.4:
        overfitting_level = "Significant"
        details = "Substantial overfitting is occurring. Consider further regularization."
    else:
        overfitting_level = "Severe"
        details = "The model is memorizing the training data and fails to generalize. Strong regularization is needed."
    
    print(f"\n3. Overfitting Assessment: {overfitting_level}")
    print(f"   {details}")
    
    # Training curve analysis
    final_train_cv_mse_gap = test_mse_mean[-1] - train_mse_mean[-1]
    final_train_cv_r2_gap = train_r2_mean[-1] - test_r2_mean[-1]
    
    print("\n4. Learning Curve Analysis:")
    print(f"   Final CV-Train MSE Gap: {final_train_cv_mse_gap:.4f}")
    print(f"   Final CV-Train R² Gap: {final_train_cv_r2_gap:.4f}")
    
    # Calculate if curves are converging
    is_converging_mse = (test_mse_mean[-1] - train_mse_mean[-1]) < (test_mse_mean[0] - train_mse_mean[0])
    is_converging_r2 = (train_r2_mean[-1] - test_r2_mean[-1]) < (train_r2_mean[0] - test_r2_mean[0])
    
    print(f"   MSE curves are {'converging' if is_converging_mse else 'not converging'}")
    print(f"   R² curves are {'converging' if is_converging_r2 else 'not converging'}")
    
    if is_converging_mse and is_converging_r2:
        convergence_trend = "The model shows healthy convergence with more data. Additional data is likely to further reduce overfitting."
    elif not is_converging_mse and not is_converging_r2:
        convergence_trend = "The gap between training and validation performance is not closing with more data. Structural changes to the model may be needed."
    else:
        convergence_trend = "Mixed convergence signals. The model may benefit from both additional data and further tuning."
    
    print(f"   {convergence_trend}")
    
    # Final verdict
    print("\n5. Final Verdict:")
    if overfitting_level in ["Minimal", "Low"]:
        if is_converging_mse and is_converging_r2:
            verdict = "The model is well-balanced with minimal overfitting. It's ready for production use."
        else:
            verdict = "The model performs well but could be further improved with additional data."
    elif overfitting_level == "Moderate":
        if is_converging_mse and is_converging_r2:
            verdict = "Some overfitting is present. Consider collecting more data or slight increases in regularization."
        else:
            verdict = "Moderate overfitting that isn't improving with more data. Consider stronger regularization."
    else:
        verdict = "Significant overfitting detected. The model needs structural changes and stronger regularization before deployment."
    
    print(f"   {verdict}")
    
    return {
        'train_mse': train_final_mse,
        'test_mse': test_final_mse,
        'train_r2': train_final_r2,
        'test_r2': test_final_r2,
        'mse_ratio': mse_overfitting_ratio,
        'r2_gap': r2_gap,
        'overfitting_level': overfitting_level
    }

# Run the analysis
overfitting_results = analyze_overfitting(regularized_model, X_train, y_train, X_test, y_test)

# Extract and display feature importances
gb_model = regularized_model.named_steps['gradientboostingregressor']
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Feature Importances ---")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance', fontsize=14)
plt.title('Feature Importances', fontsize=16)
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()