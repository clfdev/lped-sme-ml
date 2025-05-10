import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

# Load the dataset
file_path = 'Dataset3.xlsx'
data = pd.read_excel(file_path)

# Create inverse interatomic distance
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create Lennard-Jones potential
epsilon = 1
sigma = 1
data['Lennard_Jones potential'] = 4 * epsilon * (
    (sigma * data['Inverse interatomic distance']) ** 12 - 
    (sigma * data['Inverse interatomic distance']) ** 6
)

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'])

# Separate features and target
X = data_cleaned.drop(columns=['LPED'])
y = data_cleaned['LPED']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Define the Regularized Gradient Boosting model
def create_model():
    return make_pipeline(
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

# IMPROVED: Function to estimate prediction uncertainty with cross-validation
def estimate_calibrated_uncertainty(X_train, y_train, X_test, y_test, n_splits=5):
    # Step 1: Use cross-validation to get out-of-fold predictions on training data
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_oof_preds = np.zeros_like(y_train)
    train_oof_errors = np.zeros_like(y_train)
    
    for train_idx, val_idx in kf.split(X_train):
        # Split data for this fold
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create and train model
        model = create_model()
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        val_preds = model.predict(X_val_fold)
        
        # Store predictions and errors
        train_oof_preds[val_idx] = val_preds
        train_oof_errors[val_idx] = np.abs(val_preds - y_val_fold)
    
    # Step 2: Train an ensemble of models to get test predictions and uncertainty
    test_preds = []
    
    for i in range(10):  # Use 10 models with different random seeds
        # Train model with different random seed
        model = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42+i  # Different random seed for each model
            )
        )
        model.fit(X_train, y_train)
        
        # Predict on test set
        test_preds.append(model.predict(X_test))
    
    # Convert to numpy array
    test_preds = np.array(test_preds)
    
    # Calculate mean prediction and std for test set
    mean_test_pred = np.mean(test_preds, axis=0)
    uncalibrated_std = np.std(test_preds, axis=0)
    
    # Step 3: Build a calibration model based on errors vs predicted uncertainty
    # Predict standard deviation for training data points
    train_stds = []
    
    for i in range(len(y_train)):
        # Create leave-one-out train/test split
        loo_X_train = X_train.drop(X_train.index[i]).reset_index(drop=True)
        loo_y_train = y_train.drop(y_train.index[i]).reset_index(drop=True)
        loo_X_test = X_train.iloc[[i]].reset_index(drop=True)
        
        # Train multiple models (3 is enough for leave-one-out)
        loo_preds = []
        for j in range(3):
            model = make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=j  # Different random seed
                )
            )
            model.fit(loo_X_train, loo_y_train)
            loo_preds.append(model.predict(loo_X_test)[0])
        
        train_stds.append(np.std(loo_preds))
    
    train_stds = np.array(train_stds)
    
    # Calculate calibration factor using median ratio of errors to predicted stds
    valid_indices = train_stds > 0  # Avoid division by zero
    error_to_std_ratio = train_oof_errors[valid_indices] / train_stds[valid_indices]
    
    # Use median for robustness
    calibration_factor = np.median(error_to_std_ratio)
    
    # Apply calibration factor to test set standard deviations
    calibrated_std = uncalibrated_std * calibration_factor
    
    # Create prediction intervals (68% confidence interval = +/- 1 standard deviation)
    lower_bound_68 = mean_test_pred - calibrated_std
    upper_bound_68 = mean_test_pred + calibrated_std
    
    # Create 95% prediction intervals (95% confidence interval = +/- 1.96 standard deviations)
    lower_bound_95 = mean_test_pred - 1.96 * calibrated_std
    upper_bound_95 = mean_test_pred + 1.96 * calibrated_std
    
    return mean_test_pred, lower_bound_68, upper_bound_68, lower_bound_95, upper_bound_95, calibrated_std, calibration_factor

# Estimate calibrated prediction uncertainty
mean_pred, lower_68, upper_68, lower_95, upper_95, cal_std, cal_factor = estimate_calibrated_uncertainty(
    X_train, y_train, X_test, y_test)

# Calculate metrics
r2 = r2_score(y_test, mean_pred)
rmse = np.sqrt(mean_squared_error(y_test, mean_pred))

# Calculate coverage percentages
within_68_count = sum((y_test.values >= lower_68) & (y_test.values <= upper_68))
within_95_count = sum((y_test.values >= lower_95) & (y_test.values <= upper_95))
coverage_68 = (within_68_count / len(y_test)) * 100
coverage_95 = (within_95_count / len(y_test)) * 100

# Print metrics
print("Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Uncertainty Calibration Factor: {cal_factor:.4f}")
print(f"68% CI Coverage: {within_68_count}/{len(y_test)} ({coverage_68:.1f}%)")
print(f"95% CI Coverage: {within_95_count}/{len(y_test)} ({coverage_95:.1f}%)")

# Visualize predictions with uncertainty
plt.figure(figsize=(12, 6))

# Sort by actual values for better visualization
sort_idx = np.argsort(y_test.values)
x = np.arange(len(y_test))

plt.fill_between(x, lower_95[sort_idx], upper_95[sort_idx], alpha=0.3, color='blue', label='95% Confidence Interval')
plt.fill_between(x, lower_68[sort_idx], upper_68[sort_idx], alpha=0.5, color='blue', label='68% Confidence Interval')
plt.scatter(x, y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, mean_pred[sort_idx], color='blue', label='Mean Prediction')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Calibrated Uncertainty\nR² = {r2:.4f}, 95% Coverage = {coverage_95:.1f}%', 
          fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('improved_calibrated_prediction.png', dpi=300)

# IMPROVED: Function to calculate and plot calibration curve
def plot_calibration_curve(y_true, y_pred, uncertainties):
    plt.figure(figsize=(10, 6))
    
    # Define more granular confidence levels to evaluate
    confidence_levels = np.linspace(0.01, 0.99, 30)
    
    # Initialize arrays to store empirical coverage
    empirical_coverage = []
    
    # Calculate empirical coverage for each confidence level
    for conf in confidence_levels:
        # Calculate appropriate z-score for the confidence level
        # For a two-sided interval with confidence level 'conf'
        z_score = norm.ppf((1 + conf) / 2)
        
        # Calculate prediction intervals
        lower = y_pred - z_score * uncertainties
        upper = y_pred + z_score * uncertainties
        
        # Count fraction of true values falling in the interval
        in_interval = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverage.append(in_interval)
    
    # Plot calibration curve
    plt.plot(confidence_levels, empirical_coverage, 'bo-', linewidth=2, markersize=6, label='Calibrated Model')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    
    plt.xlabel('Expected Confidence Level', fontsize=14)
    plt.ylabel('Empirical Coverage', fontsize=14)
    plt.title('Uncertainty Calibration Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('improved_calibration_curve.png', dpi=300)
    
    # Return the data for further analysis
    return confidence_levels, empirical_coverage

# Plot improved calibration curve
conf_levels, emp_coverage = plot_calibration_curve(y_test.values, mean_pred, cal_std)

# Calculate calibration error metrics
calibration_error = np.mean(np.abs(np.array(conf_levels) - np.array(emp_coverage)))
print(f"\nCalibration Error: {calibration_error:.4f}")
print(f"Ideal Calibration Error: 0.0000")

# Plot sharpness (distribution of uncertainty estimates)
plt.figure(figsize=(10, 6))
plt.hist(cal_std, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Calibrated Uncertainty (Standard Deviation)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Uncertainty Estimates', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uncertainty_distribution.png', dpi=300)