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

# Function to create learning curve with analytical analysis
def plot_learning_curve(model, X, y, model_name, random_state=RANDOM_STATE):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)
    
    # Compute learning curve
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='neg_mean_squared_error'
    )
    
    # Convert negative MSE to positive
    train_scores = -train_scores
    test_scores = -test_scores
    
    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curve - {model_name}", fontsize=16)
    plt.xlabel("Training Examples", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)

    # Plot learning curves
    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    plt.xlim([0, max(train_sizes_abs) + 10])
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{model_name.replace(" ", "_")}LearningCurve.png', dpi=300)

    # Analytical Analysis
    print("\n--- Learning Curve Analysis ---")
    print(f"Training MSE at max size: {train_scores_mean[-1]:.4f}")
    print(f"Cross-validation MSE at max size: {test_scores_mean[-1]:.4f}")
    
    # Calculate overfitting ratio
    overfitting_ratio = test_scores_mean[-1] / train_scores_mean[-1] if train_scores_mean[-1] > 0 else float('inf')
    print(f"Overfitting ratio (CV/Train MSE): {overfitting_ratio:.2f}x")
    
    # Calculate improvements and diminishing returns
    improvements = []
    for i in range(1, len(test_scores_mean)):
        improvement = test_scores_mean[i-1] - test_scores_mean[i]
        per_sample = improvement / (train_sizes_abs[i] - train_sizes_abs[i-1])
        improvements.append({
            'interval': f"{int(train_sizes_abs[i-1])}-{int(train_sizes_abs[i])}",
            'improvement': improvement,
            'per_sample': per_sample
        })
    
    # Check if improvements are diminishing
    is_diminishing = all(improvements[i]['per_sample'] > improvements[i+1]['per_sample'] 
                        for i in range(len(improvements)-1))
    
    print("\nImprovement per sample:")
    for imp in improvements:
        print(f"  {imp['interval']}: {imp['improvement']:.4f} (Per sample: {imp['per_sample']:.6f})")
    print(f"Shows clear pattern of diminishing returns: {is_diminishing}")
    
    # Calculate performance gap
    performance_gap = test_scores_mean[-1] - train_scores_mean[-1]
    print(f"\nPerformance gap (CV - Training): {performance_gap:.4f}")
    
    # Projection of potential improvement
    if len(test_scores_mean) >= 3:
        recent_improvements = np.diff(test_scores_mean[-3:])
        avg_improvement_rate = np.mean(recent_improvements) / (train_sizes_abs[-1] - train_sizes_abs[-2])
        
        projected_samples = [75, 100, 150]
        current_samples = train_sizes_abs[-1]
        
        print("\nProjected MSE with more samples:")
        for samples in projected_samples:
            additional_samples = samples - current_samples
            if additional_samples > 0:
                projected_improvement = avg_improvement_rate * additional_samples
                projected_mse = test_scores_mean[-1] - projected_improvement
                percent_improvement = (projected_improvement / test_scores_mean[-1]) * 100
                print(f"  {samples} samples: {projected_mse:.4f} (Estimated improvement: {percent_improvement:.2f}%)")
    
    # Recommendation based on analysis
    if is_diminishing and overfitting_ratio < 2:
        recommendation = ("The model shows good balance and is approaching diminishing returns. "
                         "Focus on feature engineering rather than collecting more data.")
    elif is_diminishing and overfitting_ratio >= 2:
        recommendation = ("The model shows evidence of overfitting despite diminishing returns. "
                         "Consider further regularization or feature selection.")
    else:
        recommendation = ("The model may benefit from additional data collection, "
                         "though regularization parameters appear to be effective.")
    
    print("\nRecommendation:")
    print(recommendation)

    return plt, train_sizes_abs, train_scores_mean, test_scores_mean

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

# Generate learning curve
plt.close('all')  # Close any existing plots
fig, train_sizes, train_scores, test_scores = plot_learning_curve(
    regularized_model, X, y, "Regularized Gradient Boosting"
)

# Show the plot
plt.show()

# Final Model Performance Analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
regularized_model.fit(X_train, y_train)
y_pred = regularized_model.predict(X_test)

# Calculate final performance metrics
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)
final_rmse = np.sqrt(final_mse)

print("\n--- Final Model Performance ---")
print(f"Mean Squared Error (MSE): {final_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {final_rmse:.4f}")
print(f"R² Score: {final_r2:.4f}")

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
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()