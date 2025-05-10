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
    plt.figure(figsize=(12, 8))
    plt.title(f"Learning Curve - {model_name}", fontsize=18)
    plt.xlabel("Training Examples", fontsize=16)
    plt.ylabel("Mean Squared Error", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Plot learning curves
    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    plt.xlim([0, max(train_sizes_abs) + 10])
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{model_name.replace(" ", "_")}LearningCurve.png', dpi=300, bbox_inches='tight')

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
    
    # Recommendations based on overfitting ratio
    if overfitting_ratio > 20:
        model_status = "Severe overfitting"
        recommendation = ("The model shows severe overfitting. Prioritize regularization techniques like reducing "
                         "max_depth, increasing min_samples_leaf, and using a smaller learning rate.")
    elif overfitting_ratio > 5:
        model_status = "Significant overfitting"
        recommendation = ("The model shows significant overfitting. Consider applying regularization and early stopping. "
                         "Adding more data alone is unlikely to solve the problem.")
    elif overfitting_ratio > 2:
        model_status = "Moderate overfitting"
        recommendation = ("The model shows moderate overfitting. A balanced approach of regularization and "
                         "potentially more data collection may help.")
    else:
        model_status = "Good balance"
        recommendation = ("The model shows a good balance between training and validation performance. "
                         "Fine-tuning and potentially more data may further improve results.")
    
    print(f"\nModel Status: {model_status}")
    print("Recommendation:")
    print(recommendation)

    return plt, train_sizes_abs, train_scores_mean, test_scores_mean

# Compare different model configurations
def compare_models():
    # Define model configurations
    models = {
        "Default Gradient Boosting": make_pipeline(
            StandardScaler(), 
            GradientBoostingRegressor(random_state=42)
        ),
        
        "Regularized Gradient Boosting": make_pipeline(
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
        ),
        
        "Gradient Boosting with Early Stopping": make_pipeline(
            StandardScaler(), 
            GradientBoostingRegressor(
                n_estimators=500,    # More trees, but early stopping will limit this
                learning_rate=0.1,
                max_depth=3,
                validation_fraction=0.2,
                n_iter_no_change=10, # Stop if no improvement for 10 iterations
                tol=0.001,           # Tolerance for improvement
                random_state=42
            )
        ),
        
        "Fully Optimized Gradient Boosting": make_pipeline(
            StandardScaler(), 
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                validation_fraction=0.2,
                n_iter_no_change=10,
                tol=0.001,
                random_state=42
            )
        )
    }
    
    results = {}
    
    # Generate learning curves for each model
    for name, model in models.items():
        print(f"\n{'='*80}\nAnalyzing: {name}\n{'='*80}")
        plt.close('all')
        _, train_sizes, train_scores, test_scores = plot_learning_curve(model, X, y, name)
        
        results[name] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'final_cv_mse': test_scores[-1],
            'train_cv_gap': abs(train_scores[-1] - test_scores[-1])
        }
    
    # Compare final results
    print("\n\n" + "="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Final CV MSE': [results[model]['final_cv_mse'] for model in results],
        'Train/CV Gap': [results[model]['train_cv_gap'] for model in results],
        'Overfitting Ratio': [results[model]['final_cv_mse'] / results[model]['train_scores'][-1] 
                             if results[model]['train_scores'][-1] > 0 else float('inf') 
                             for model in results]
    }).sort_values('Final CV MSE')
    
    print(comparison_df)
    
    # Plot comparison of learning curves
    plt.figure(figsize=(14, 8))
    plt.title("Learning Curves Comparison", fontsize=18)
    plt.xlabel("Training Examples", fontsize=16)
    plt.ylabel("Cross-validation MSE", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    for name, result in results.items():
        plt.plot(result['train_sizes'], result['test_scores'], 'o-', label=name)
    
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print best model recommendation
    best_model = comparison_df.iloc[0]['Model']
    print(f"\nBest performing model: {best_model}")
    print(f"CV MSE: {comparison_df.iloc[0]['Final CV MSE']:.4f}")
    print(f"Overfitting ratio: {comparison_df.iloc[0]['Overfitting Ratio']:.2f}x")
    
    # Final evaluation on a separate test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    best_model_pipeline = models[best_model]
    best_model_pipeline.fit(X_train, y_train)
    y_pred = best_model_pipeline.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\nFinal test set performance ({best_model}):")
    print(f"MSE: {test_mse:.4f}")
    print(f"R²: {test_r2:.4f}")
    
    # Extract feature importances if available
    try:
        importances = best_model_pipeline.named_steps['gradientboostingregressor'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importances:")
        print(feature_importance_df)
        
        # Visualize feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance', fontsize=14)
        plt.title('Feature Importances', fontsize=16)
        plt.tight_layout()
        plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
        plt.show()
    except:
        print("\nCould not extract feature importances from the model.")

# Run the model comparison
if __name__ == "__main__":
    compare_models()