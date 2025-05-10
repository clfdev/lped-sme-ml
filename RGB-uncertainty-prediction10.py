import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn import metrics
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

# Load the dataset
file_path = 'Dataset3.xlsx'
data = pd.read_excel(file_path)

# Feature engineering
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']
data['Lennard_Jones potential'] = 4 * 1 * (
    (1 * data['Inverse interatomic distance']) ** 12 - 
    (1 * data['Inverse interatomic distance']) ** 6
)

# Clean the data
data_cleaned = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'])

# Split features and target
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

# Split train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=4)

# Create and train the primary model
def create_model(random_state=42):
    return make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state
        )
    )

# Train primary model
model_pipeline = create_model()
model_pipeline.fit(X_train, Y_train)

# Make predictions
Y_pred_train = model_pipeline.predict(X_train)
Y_pred_test = model_pipeline.predict(X_test)

# Calculate metrics
def get_metrics(Y_actual, Y_pred):
    return {
        'r2': metrics.r2_score(Y_actual, Y_pred),
        'mae': metrics.mean_absolute_error(Y_actual, Y_pred),
        'mse': np.mean((Y_actual - Y_pred)**2),
        'rmse': np.sqrt(np.mean((Y_actual - Y_pred)**2))
    }

# Get metrics
train_metrics = get_metrics(Y_train, Y_pred_train)
test_metrics = get_metrics(Y_test, Y_pred_test)

print("\nTraining Set Metrics:")
print(train_metrics)
print("\nTest Set Metrics:")
print(test_metrics)

# Uncertainty Estimation Function
def estimate_calibrated_uncertainty(X_train, Y_train, X_test, Y_test, Y_pred_test):
    # Step 1: Generate ensemble predictions with many models
    n_models = 30  # Increased from 10 to 30 for smoother estimates
    ensemble_preds = []
    
    # First model is our original model's predictions
    ensemble_preds.append(Y_pred_test)
    
    # Train additional models with different seeds
    for i in range(1, n_models):
        model = create_model(random_state=42+i*10)  # Use larger step in seeds for more diversity
        model.fit(X_train, Y_train)
        ensemble_preds.append(model.predict(X_test))
    
    ensemble_preds = np.array(ensemble_preds)
    
    # Calculate raw standard deviation
    uncalibrated_std = np.std(ensemble_preds, axis=0)
    
    # Step 2: Use cross-validation to get calibration data
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # For each fold, store:
    # - True values
    # - Mean predictions
    # - Std of predictions
    cal_data = []
    
    for train_idx, val_idx in kf.split(X_train):
        # Split data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        Y_fold_train, Y_fold_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
        
        # Train models
        fold_preds = []
        for i in range(n_models):
            model = create_model(random_state=42+i*10)
            model.fit(X_fold_train, Y_fold_train)
            fold_preds.append(model.predict(X_fold_val))
        
        fold_preds = np.array(fold_preds)
        fold_mean = np.mean(fold_preds, axis=0)
        fold_std = np.std(fold_preds, axis=0)
        
        # Store calibration data
        for j in range(len(Y_fold_val)):
            cal_data.append({
                'true': Y_fold_val.iloc[j],
                'pred': fold_mean[j],
                'std': fold_std[j],
                'error': abs(fold_mean[j] - Y_fold_val.iloc[j])
            })
    
    # Step 3: Build a non-parametric calibration model
    # Convert to arrays
    uncal_std = np.array([d['std'] for d in cal_data])
    errors = np.array([d['error'] for d in cal_data])
    
    # Sort by std for smoothing
    sort_idx = np.argsort(uncal_std)
    uncal_std_sorted = uncal_std[sort_idx]
    errors_sorted = errors[sort_idx]
    
    # Smooth calibration using isotonic regression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(uncal_std_sorted, errors_sorted)
    
    # Apply calibration to test set
    calibrated_std = ir.predict(uncalibrated_std)
    
    # Create confidence intervals
    lower_bound_68 = Y_pred_test - calibrated_std
    upper_bound_68 = Y_pred_test + calibrated_std
    lower_bound_95 = Y_pred_test - 1.96 * calibrated_std
    upper_bound_95 = Y_pred_test + 1.96 * calibrated_std
    
    # Calculate coverage
    within_68 = np.mean((Y_test.values >= lower_bound_68) & (Y_test.values <= upper_bound_68)) * 100
    within_95 = np.mean((Y_test.values >= lower_bound_95) & (Y_test.values <= upper_bound_95)) * 100
    
    # Calculate relative uncertainty
    rel_uncertainty = (calibrated_std / np.abs(Y_pred_test)) * 100
    avg_rel_uncertainty = np.mean(rel_uncertainty)
    
    return {
        'cal_std': calibrated_std,
        'rel_uncertainty': rel_uncertainty,
        'avg_rel_uncertainty': avg_rel_uncertainty,
        'lower_68': lower_bound_68,
        'upper_68': upper_bound_68,
        'lower_95': lower_bound_95,
        'upper_95': upper_bound_95,
        'coverage_68': within_68,
        'coverage_95': within_95
    }

# Get calibrated uncertainties
uncertainty_results = estimate_calibrated_uncertainty(X_train, Y_train, X_test, Y_test, Y_pred_test)

# Print uncertainty information
print("\nUncertainty Information:")
print(f"68% CI Coverage: {uncertainty_results['coverage_68']:.1f}%")
print(f"95% CI Coverage: {uncertainty_results['coverage_95']:.1f}%")
print(f"Average Relative Uncertainty: {uncertainty_results['avg_rel_uncertainty']:.2f}%")

# Plot calibration curve
def plot_calibration_curve(y_true, y_pred, uncertainties):
    plt.figure(figsize=(10, 6))
    
    # Use more granular confidence levels
    confidence_levels = np.linspace(0.01, 0.99, 40)  # Increased from 30 to 40
    
    # Initialize arrays to store empirical coverage
    empirical_coverage = []
    
    # Calculate empirical coverage for each confidence level
    for conf in confidence_levels:
        # Calculate z-score
        z_score = norm.ppf((1 + conf) / 2)
        
        # Calculate prediction intervals
        lower = y_pred - z_score * uncertainties
        upper = y_pred + z_score * uncertainties
        
        # Count fraction of true values in interval
        in_interval = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverage.append(in_interval)
    
    # Plot calibration curve
    plt.plot(confidence_levels, empirical_coverage, 'bo-', linewidth=2, markersize=5, label='Calibrated Model')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    
    plt.xlabel('Expected Confidence Level', fontsize=14)
    plt.ylabel('Empirical Coverage', fontsize=14)
    plt.title('Uncertainty Calibration Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('calibration_curve.png', dpi=300)
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(np.array(confidence_levels) - np.array(empirical_coverage)))
    return calibration_error

# Plot the calibration curve
calibration_error = plot_calibration_curve(Y_test.values, Y_pred_test, uncertainty_results['cal_std'])
print(f"Calibration Error: {calibration_error:.4f}")

# Plot predictions with uncertainty
plt.figure(figsize=(12, 6))
sort_idx = np.argsort(Y_test.values)
x = np.arange(len(Y_test))

plt.errorbar(x, Y_pred_test[sort_idx], yerr=1.96*uncertainty_results['cal_std'][sort_idx], 
             fmt='o', ecolor='blue', alpha=0.7, capsize=4, label='95% Confidence Interval')
plt.scatter(x, Y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, Y_pred_test[sort_idx], color='blue', label='Predicted Values')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Calibrated Uncertainty\nR² = {test_metrics["r2"]:.4f}, Avg. Uncertainty = {uncertainty_results["avg_rel_uncertainty"]:.2f}%', 
          fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predictions_with_uncertainty.png', dpi=300)